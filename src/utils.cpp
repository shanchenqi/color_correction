#include "utils.h"


using namespace cv;
using namespace std;

namespace
{
    cv::Mat togray_Mat = (cv::Mat_<double>(3, 1) << 0.2126, 0.7152, 0.0722); 
}

// some convection functions
cv::Mat saturate(cv::Mat src, double low, double up)
{
    cv::Mat src_saturation(src.size(), CV_64FC1);
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++) 
        {
            double saturation_ij = 1;
            for (int m = 0; m < 3; m++)
            {
                if (not((src.at<Vec3d>(i, j)[m] < up) && (src.at<Vec3d>(i, j)[m] > low))) 
                {
                    saturation_ij = 0;
                    break;
                }
            }
            src_saturation.at<double>(i, j) = saturation_ij;
        }
    }
    return src_saturation;
}

cv::Mat xyz2grayl(cv::Mat xyz) 
{
    vector<cv::Mat> channels;
    split(xyz, channels);
    return channels[1];
}

cv::Mat xyz2lab(cv::Mat xyz, IO io)
{ 
    vector<double> xyz_ref_white_io = illuminants[io];
    cv::Mat lab(xyz.size(), xyz.type());  
    for (int i = 0; i < xyz.rows; i++) 
    {
        for (int j = 0; j < xyz.cols; j++) 
        {       
            f_xyz2lab(xyz.at<Vec3d>(i, j)[0], xyz.at<Vec3d>(i, j)[1], xyz.at<Vec3d>(i, j)[2],//x,y,z
                     lab.at<Vec3d>(i, j)[0], lab.at<Vec3d>(i, j)[1], lab.at<Vec3d>(i, j)[2], //L,a,b
                     xyz_ref_white_io[0], xyz_ref_white_io[1], xyz_ref_white_io[2]);//Xn, Yn, Zn
        }
    }
    return lab;
}

void f_xyz2lab(double  X, double  Y, double Z, double& L, double& a, double& b, double Xn, double Yn, double Zn)//todo cv引用过来
{
    // CIE XYZ values of reference white point for D65 illuminant
   // static const float Xn = 0.950456f, Yn = 1.f, Zn = 1.088754f;

    // Other coefficients below:
    // 7.787f    = (29/3)^3/(29*4)
    // 0.008856f = (6/29)^3
    // 903.3     = (29/3)^3

    double x = X / Xn, y = Y / Yn, z = Z / Zn;
    auto f = [](double t) { return t > 0.008856 ? std::cbrtl(t) : (7.787 * t + 16.0 / 116.0); };
    double fx = f(x), fy = f(y), fz = f(z);
    L = y > 0.008856 ? (116.0 * std::cbrtl(y) - 16.0) : (903.3 * y);
    a = 500.0 * (fx - fy);
    b = 200.0 * (fy - fz);
}
double  r_revise(double x)
{  
    return x = (x > 6.0 / 29.0) ? pow(x, 3.0) : (x - 16.0 / 116.0) * 3 * powl(6.0 / 29.0, 2);  
}

void f_lab2xyz(double l, double a, double b, double& x, double& y, double& z, double Xn, double Yn, double Zn) 
{
    double Y = (l + 16.0) / 116.0;
    double X = Y + a / 500.0;
    double Z = Y - b / 200.0;
    if (z < 0)
    {
        z = 0;
    }
    x = r_revise(X) * Xn;
    y = r_revise(Y) * Yn;
    z = r_revise(Z) * Zn;
}

cv::Mat lab2xyz(cv::Mat lab, IO io) 
{
    vector<double> xyz_ref_white_io = illuminants[io];
    cv::Mat xyz(lab.size(), lab.type());
    for (int i = 0; i < lab.rows; i++) 
    {
        for (int j = 0; j < lab.cols; j++)
        {
            f_lab2xyz(lab.at<Vec3d>(i, j)[0], lab.at<Vec3d>(i, j)[1], lab.at<Vec3d>(i, j)[2],// l, a, b,
                     xyz.at<Vec3d>(i, j)[0], xyz.at<Vec3d>(i, j)[1], xyz.at<Vec3d>(i, j)[2],//x, y, z, 
                      xyz_ref_white_io[0], xyz_ref_white_io[1], xyz_ref_white_io[2]);// Xn, Yn, Zn);
        }
    }
    return xyz;
}

cv::Mat rgb2gray(cv::Mat rgb)
{
    cv::Mat togray = togray_Mat;
    cv::Mat gray(rgb.size(), CV_64FC1);
    for (int i = 0; i < rgb.rows; i++)
    {
        for (int j = 0; j < rgb.cols; j++)
        {
            double res1 = rgb.at<Vec3d>(i, j)[0] * togray.at<double>(0, 0);
            double res2 = rgb.at<Vec3d>(i, j)[1] * togray.at<double>(1, 0);
            double res3 = rgb.at<Vec3d>(i, j)[2] * togray.at<double>(2, 0);
            gray.at<double>(i, j) = res1 + res2 + res3;
        }
    }
    return gray;
}

cv::Mat xyz2xyz(cv::Mat xyz, IO sio, IO dio) 
{
    if (sio == dio) 
    {
        return xyz;
    }
    else
    {
        cv::Mat cam(IO, IO, string);
        cv::Mat cam_M = cam(sio, dio, "Bradford");
        cv::Mat cam_res(xyz.size(), xyz.type());
        cam_res = mult(xyz, cam_M);
        return cam_res;
    }
}

cv::Mat lab2lab(cv::Mat lab, IO sio, IO dio) 
{
    if (sio == dio) {
        return lab;
    }
    return xyz2lab(xyz2xyz(lab2xyz(lab, sio), sio, dio), dio);
}

// others
double gammaCorrection_f(double f, double gamma) 
{
    double k =( f >= 0 ? pow(f, gamma) : -pow((-f), gamma));
    return k;
}

cv::Mat gammaCorrection(cv::Mat src, double K)
{
    cv::Mat dst(src.size(),src.type());
    for (int row = 0; row < src.rows; row++)
    {
        for (int col = 0; col < src.cols; col++) 
        {
            for (int m=0;m<3;m++) dst.at<Vec3d>(row, col)[m] = gammaCorrection_f(src.at<Vec3d>(row, col)[m], K);
        }
    }
    return dst;
}

cv::Mat mult(cv::Mat xyz, cv::Mat ccm)//reshape(mxnx3c)（）
{
    cv::Mat res(xyz.size(), CV_64FC3);
    for (int i = 0; i < xyz.rows; i++) 
    {
        for (int j = 0; j < xyz.cols; j++) 
        {
            for (int m = 0; m < res.channels(); m++) 
            {
                res.at<Vec3d>(i, j)[m] = 0;
                for (int n = 0; n < xyz.channels(); n++) 
                {
                    res.at<Vec3d>(i, j)[m] += xyz.at<Vec4d>(i, j)[n] * ccm.at<double>(n, m);
                }
            }
        }
    }
    return res;
}

cv::Mat mult3D(cv::Mat xyz, cv::Mat ccm) 
{
    cv::Mat res(xyz.size(), CV_64FC3);
    for (int i = 0; i < xyz.rows; i++) 
    {
        for (int j = 0; j < xyz.cols; j++) 
        {
            for (int m = 0; m < res.channels(); m++)
            {
                res.at<Vec3d>(i, j)[m] = 0;
                for (int n = 0; n < xyz.channels(); n++)
                {
                    res.at<Vec3d>(i, j)[m] += xyz.at<Vec3d>(i, j)[n] * ccm.at<double>(n, m);
                }
            }
        }
    }
    return res;
}
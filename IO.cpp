#include "IO.h"
#include <string>
#include <iostream>

//data from https://en.wikipedia.org/wiki/Standard_illuminant

namespace
{
    vector<double> illuminants_xy_A_2 = { 0.44757, 0.40745 };
    vector<double> illuminants_xy_A_10 = { 0.45117, 0.40594 };
    vector<double> illuminants_xy_D50_2 = { 0.34567, 0.35850 };
    vector<double> illuminants_xy_D50_10 = { 0.34773, 0.35952 };
    vector<double> illuminants_xy_D55_2 = { 0.33242, 0.34743 };
    vector<double> illuminants_xy_D55_10 = { 0.33411, 0.34877 };
    vector<double> illuminants_xy_D65_2 = { 0.31271, 0.32902 };
    vector<double> illuminants_xy_D65_10 = { 0.31382, 0.33100 };
    vector<double> illuminants_xy_D75_2 = { 0.29902, 0.31485 };
    vector<double> illuminants_xy_E_2 = { 1 / 3, 1 / 3 };
    vector<double> illuminants_xy_E_10 = { 1 / 3, 1 / 3 };
    cv::Mat  Von_Kries_Mat = (Mat_<double>(3, 3) << 0.40024, 0.7076, -0.08081,-0.2263, 1.16532, 0.0457,0., 0., 0.91822);
    cv::Mat Bradford_Mat = (Mat_<double>(3, 3) << 0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296);
}

map<IO, vector<double> >  illuminants = getIlluminant();

 IO::IO(string illuminant, int observer) 
 {
    m_illuminant = illuminant;
    m_observer = observer;
}

IO::~IO() {}

bool IO::operator<(const IO& other)const 
{
    return (m_illuminant < other.m_illuminant || ((m_illuminant == other.m_illuminant) && (m_observer < other.m_observer))); 
}

vector<double> xyY2XYZ(double x, double y, double Y) 
{  
    double X;
    double Z;
    X = Y * x / y;
    Z = Y / y * (1 - x - y);
    vector <double>  xyY2XYZ(3);
    xyY2XYZ[0] = X;
    xyY2XYZ[1] = Y;
    xyY2XYZ[2] = Z;
    return xyY2XYZ;
}

map <IO, vector<double>> getIlluminant() 
{
    map<IO, vector<double> >  illuminants_xy;
    illuminants_xy[A_2] = illuminants_xy_A_2;
    illuminants_xy[A_10] = illuminants_xy_A_10;
    illuminants_xy[D50_2] = illuminants_xy_D50_2;
    illuminants_xy[D50_10] = illuminants_xy_D50_10;
    illuminants_xy[D55_2] = illuminants_xy_D55_2;
    illuminants_xy[D55_10] = illuminants_xy_D55_10;
    illuminants_xy[D65_2] = illuminants_xy_D65_2;
    illuminants_xy[D65_10] = illuminants_xy_D65_10;
    illuminants_xy[D75_2] = illuminants_xy_D75_2;
    illuminants_xy[E_2] = illuminants_xy_E_2;
    illuminants_xy[E_10] = illuminants_xy_E_10;
    map <IO, vector<double>>  illuminants_;
    map<IO, vector<double>>::iterator it;
    it = illuminants_xy.begin();
    for (it; it != illuminants_xy.end(); it++)
    {
        double x = it->second[0];
        double y = it->second[1];
        double Y = 1;
        vector<double> res;
        res = xyY2XYZ(x, y, Y);
        illuminants_[it->first] = res;
    }
    illuminants_[D65_2] = { 0.95047, 1.0, 1.08883 };   
    illuminants_[D65_10] = { 0.94811, 1.0, 1.07304 };
    return illuminants_;
}

//chromatic adaption matrices
map <tuple<IO, IO, string>, cv::Mat > CAMs;

//function to get cam
cv::Mat cam(IO sio, IO dio, string method)
{ 
    cv::Mat  Von_Kries = Von_Kries_Mat;
    cv::Mat Bradford = Bradford_Mat;
    map <String, vector< cv::Mat >> MAs;
    MAs["Identity"] = { cv::Mat::eye(cv::Size(3,3),CV_64FC1) , cv::Mat::eye(cv::Size(3,3),CV_64FC1) };
    MAs["Von_Krie"] = { Von_Kries ,Von_Kries.inv() };
    MAs["Bradford"] = { Bradford ,Bradford.inv() };
    if (CAMs.count(make_tuple(dio, sio, method)) == 1)
    {
        return CAMs[make_tuple(dio, sio, method)];
    }
    cv::Mat XYZws = cv::Mat(illuminants[dio]);
    cv::Mat XYZWd = cv::Mat(illuminants[sio]);
    cv::Mat MA = MAs[method][0];
    cv::Mat MA_inv = MAs[method][1];
    cv::Mat MA_res1 = MA * XYZws;
    cv::Mat MA_res2 = MA * XYZWd;
    cv::Mat MA_res3 = MA_res1 / MA_res2;
    cv::Mat me = cv::Mat::eye(cv::Size(3, 3), CV_64FC1);
    for (int i = 0; i < 3; i++) 
    {
        me.at<double>(i, i) = MA_res3.at<double>(i, 0);
    }
    cv::Mat M = MA_inv * (me);
    CAMs[make_tuple(dio, sio, method)] = M;
    return M;
}
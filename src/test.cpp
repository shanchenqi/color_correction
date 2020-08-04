#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <map>
#include <tuple>
#include <cmath>
#include "distance.h"
#include "IO.h"
#include "utils.h"
//#include "colorspace.h"
#include "ccm.h"
#include "colorchecker.h"
#include "linearize.h"



using namespace cv;
using namespace std;

int main() {
    // cout << illuminants.size() << endl;
   // IO sio = IO("D65", 2);
   // IO dio = IO("D65", 10);
   //string method = "Bradford";
   // std::cout << Mat(illuminants[dio]) << endl;
    //   Mat src1 = (Mat_<double>(4, 3) << 50.0000, 2.6772, -79.7751, 50.0000, 3.1571, -77.2803, 7, 8, 9, 10, 11, 12);
      // , 50.0000, 3.1571, -77.2803, 7, 8, 9, 10, 11, 12
     //  Mat src2 = (Mat_<double>(3, 3) << 50.0000, 0.0000, -82.7485, 50.0000, 0.0000, -82.7485, 12, 13, 15);
      // Mat cam(IO sio, IO dio, string method = "Bradford");
      // Mat distance_s(Mat Lab1, Mat Lab2, string distance);
       
    //   Mat cam(IO sio, IO dio, string method = "Bradford");
    //   Mat cam_M = cam(A_2, D65_2, "Bradford");
    //   Mat srcImage;
   //    srcImage = imread("D:/OpenCV/test.jpg");
   //    Mat src1 = srcImage(Range(0, 4), Range(0, 4));
   //    src1.convertTo(src1, CV_64F);
    //   Mat src2 = srcImage(Range(3, 7), Range(4, 8));
     //  src2.convertTo(src2, CV_64F);
       //System.out.println(CvType.typeToString(src));
  //  IO* test_sio = &A_2;
  //  IO* test_dio = &D65_2;
    //  string distance = "distance_de00";
     //string distance = "distance_cmc";
     // string distance = "distance_de76";
     //string distance = "distance_de94";
    //  Mat M(3, 1, CV_64FC3);
    //  M.at<Vec3d>(0, 0) = { 37.986 ,13.555,14.059};
    //  M.at<Vec3d>(1, 0) = { 65.711 ,18.13,17.81 };
    //  M.at<Vec3d>(2, 0) = { 49.927 ,-4.88,17.81 };
     // cout << M.channels() << endl;
      //std::cout << "res.size " << M<< endl;
   // Mat res=cam;
    // cout <<"src / 255  " << (src / 255) << endl;
    Mat res;
   //  res = cam(A_2, D65_2,method );
     //res = xyz2grayl(src / 255);
     //res = xyz2lab(src1/255, D65_2);
     //double low = 180.0;
    // double up = 255.0;
     //cout << src1 << endl;
    // res = saturate(src1,  low,  up);
     //cout << "src1/255"<<src1 / 255 << endl;
    // cout << cam_M << endl;

     //res = distance_s(src1, src2, distance);
    //RGB_Base test;
  //  Mat res = test.cal_M_RGBL2XYZ_base();
    // WideGamutRGB test;
     //test._M_RGBL2XYZ_base=  (Mat_<double>(3, 3) << 50.0000, 2.6772, -79.7751, 50.0000, 3.1571, -77.2803, 7, 8, 9);;
    // cout << "res.size " << res << endl;
    //cout << "res.size " << cam_M << endl;
   //  cout << "res.size " << test.rgbl2rgb(src1/255 ) << endl;
  //   string colorspace = "LAB";
    // Mat Arange_18_24 = (Mat_<int>(1, 7) << 18, 19, 20, 21, 22, 23, 24);

  //   ColorChecker colorchecker_Macbeth = ColorChecker(ColorChecker2005_LAB_D50_2, "LAB", IO("D65", 2), Arange_18_24);
 //    ColorChecker colorchecker_Macbeth_D65_2 = ColorChecker(ColorChecker2005_LAB_D65_2, "LAB", IO("D65", 2), Arange_18_24);

    //cout << test.lab << endl;
    //map <IO, vector<double>> xyz_ref_white = get_xyz_ref_white();
   // vector<double> xyz_ref_white_io = xyz_ref_white[D65_2];
    //cout <<"123  "<< illuminants[D65_2][2] << endl;
    Mat test_src = (Mat_<Vec3d>(24, 1) <<
        Vec3d(50.9, 49.07, 20.62),
        Vec3d(144.35, 142.37, 68.76),
        Vec3d(58.45, 98.21, 76.68),
        Vec3d(47.21, 64.9, 19.75),
        Vec3d(75.94, 107.21, 88.47),
        Vec3d(110.73, 193.01, 103.59),
        Vec3d(169.94, 110.82, 22.24),
        Vec3d(38.24, 74.13, 89.),
        Vec3d(105.75, 63.45, 33.),
        Vec3d(27.06, 33.33, 28.2),
        Vec3d(156.78, 197.28, 44.47),
        Vec3d(188.47, 155.32, 24.02),
        Vec3d(19.35, 42.9, 63.2),
        Vec3d(78.25, 131.45, 44.16),
        Vec3d(74.04, 36.51, 14.55),
        Vec3d(254.54, 251.8, 42.45),
        Vec3d(98.13, 75.17, 67.1),
        Vec3d(44.48, 114.7, 97.86),
        Vec3d(255., 255., 255.),
        Vec3d(254.96, 255., 201.62),
        Vec3d(177.42, 222.95, 122.64),
        Vec3d(95.35, 121.51, 66.48),
        Vec3d(45.4, 59.18, 32.),
        Vec3d(17.68, 23.99, 12.22));
    
   /* ColorChecker colorcheckertest = colorchecker_Macbeth_D65_2;
    string colorspace = "sRGB";
    ColorCheckerMetric test(colorcheckertest, colorspace, D65_2);
    cout <<"test.rgbl"<< test.rgbl << endl;*/
   /* float gamma = 2.2;
    int deg = 3;
    Mat src = ColorChecker2005_LAB_D65_2/100;*/
    //Mat step = (Mat_<double>(3, 3) << 2, 3, 4, 1, -0.1, 3, 0.1, 0.1, 0.1);
    //cout << step << endl;
   //cout<<"mult"<< mult(test_src,step);
   // ColorCheckerMetric cc = test;
   //
   // Linear_gamma test_linearize(gamma, deg, test_src/255,  cc, saturated_threshold);
   //// cout <<"res" <<test_linearize.linearize(src1/255) << endl;
   // string dst_colorspace= "RGB_Base";
   // Mat src_ = src1 / 255;
   // string dst_illuminant = "D65";
   // int dst_observer = 2;
   // Mat dst_whites = test.white_mask;
   // string linear_ = "Linear_gamma";
   // string distance_ = "distance_de00";
   // string  dist_illuminant = "D65";
   // int dist_observer = 10;
   // Mat weights_list ;
   // double weights_coeff =0;
   // bool weights_color=false;
   // string initial_method="least_square";
   // double xtol_ =1e-4;
   // double ftol_ =1e-4;
   // Mat dst;
   // string colorchecker = " ColorChecker2005_LAB_D65_2";
    
  /*  CCM_3x3::CCM_3x3(Mat src_, Mat dst, string dst_colorspace, string dst_illuminant, int dst_observer, Mat dst_whites, string colorchecker, vector<double> saturated_threshold,/
    string colorspace, string linear_, float gamma, float deg, string distance_, string dist_illuminant, int dist_observer, Mat weights_list, double weights_coeff, bool weights_color, string initial_method, double xtol_, double ftol_)
 */   vector<double> saturated_threshold(2);
    saturated_threshold[0] = 0.02;
    saturated_threshold[1] = 0.98;
    Mat src_Mat = test_src/255;
    Mat dst;
    string dst_colorspace = "sRGB";
    string dst_illuminant= "D65";
    int dst_observer =2 ;
    Mat dst_whites;
    string colorchecker = "Macbeth_D65_2";
  //  vector<double> saturated_threshold;
    string colorspace = "AdobeRGB";
 
    string linear_= "Linear_gray_polyfit";
  //  string linear_ = "identity";
   // string linear_ = "Linear_gamma";
    double gamma = 2.2;
    int deg = 2;
    //string distance_ = "de00";
    string distance_ = "rgbl";
    string dist_illuminant = "D65"; 
    int dist_observer = 2;
    Mat weights_list; 
    double weights_coeff = 0;
    bool weights_color = false;
    string initial_method = "least_square";
    //string initial_method = "white_balance";
    string shape = "4x3";
   // CCM_4x3 ccmtest();
    CCM_4x3 ccmtest(src_Mat, dst, dst_colorspace, dst_illuminant, dst_observer, dst_whites, colorchecker, saturated_threshold, colorspace, linear_,
        gamma, deg, distance_, dist_illuminant,dist_observer, weights_list, weights_coeff, weights_color,  initial_method, shape);
    ccmtest.calc(initial_method, distance_);
    ccmtest.value(10000);
    //ccmtest.prepare();
    string imgfile = "D:/OpenCV/input2.png";
    Mat output_image;
    output_image = ccmtest.infer_image(imgfile);
    imwrite("D:/OpenCV/input2_infer"+distance_ +shape+".png", output_image);
    //cout << "ccmtest.src_rgbl_masked" << ccmtest.src_rgbl_masked << endl ;
    /*
    Mat test = (Mat_<double>(6, 1) << 0.37542,0.652,0.61365,0.4986,0.70916,0.40554);
    Mat test_dst = (Mat_<double>(6, 1) << 0.14078353, 0.46619182, 0.58291028, 0.41416074, 0.6622565, 0.32723434);
    Linear testpolyfit;
    res = testpolyfit.polyfit(test, test_dst, 3);
    */
   // cout << "res    oyo    " << res << endl;
    return 0;

}
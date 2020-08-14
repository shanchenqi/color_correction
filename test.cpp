#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <map>
#include <string>
#include "color.h"
#include "time.h"

using namespace cv;
using namespace cv::ccm;
using namespace std;

//#include "distance.h"
//#include "utils.h"

//
//Mat resh(Mat xyz, Mat ccm) {
//    //Mat res(xyz.size(), CV_64FC3);
//    int rows = xyz.rows;
//    xyz = xyz.reshape(1, xyz.rows * xyz.cols);
//    Mat res = xyz*ccm;
//    res = res.reshape(3, rows);
//    return res;
//}
//
//
//Mat mult(Mat xyz, Mat ccm) {
//    //cout<<ccm<<endl;
//    //int c = xyz.channels();
//
//    Mat res(xyz.size(), CV_64FC3);
//    //cout << "3:" << res.size() <<"," <<xyz.rows<<"," <<xyz.cols << endl;
//    for (int i = 0; i < xyz.rows; i++) {
//        for (int j = 0; j < xyz.cols; j++) {
//            for (int m = 0; m < res.channels(); m++) {//
//
//                res.at<Vec3d>(i,j)[m] = 0;
//                for (int n = 0; n < xyz.channels(); n++) {
//                    res.at<Vec3d>(i,j)[m] += xyz.at<Vec3d>(i,j)[n] * ccm.at<double>(n, m);
//                    /*double res1 = xyz.at<Vec3d>(i, j)[0] * ccm.at<double>(0, m);
//                    double res2 = xyz.at<Vec3d>(i, j)[1] * ccm.at<double>(1, m);
//                    double res3 = xyz.at<Vec3d>(i, j)[2] * ccm.at<double>(2, m);
//                    res.at<Vec3d>(i, j)[m] = res1 + res2 + res3;*/
//                }
//
//            }
//            //cout << "j:" << j << endl;
//        }
//        //cout << "i:" << i << endl;
//    }
//    return res;
//}

int main() {
	Mat test = (Mat_<Vec3d>(1, 1) << Vec3d(0.3, 0.2, 0.5));
	cout << "123" << endl;
	Color color(test, sRGB); 
	/// #sRGB, sRGBL = bind(sRGB_);
	Color color_rgb = color.to(sRGB);
	
	cout << color_rgb.colors << endl;
	return 0;

}

#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>
#include "io.h"
#include "distance.h"

using namespace cv;
using namespace std;

void f_xyz2lab(double  X, double  Y, double Z,double& L, double& a, double& b, double Xn, double Yn, double Zn);
double gammaCorrection_f(double  f, double gamma);
cv::Mat  saturate(cv::Mat src, double low, double up);
cv::Mat xyz2grayl(cv::Mat xyz);
cv::Mat xyz2lab(cv::Mat xyz, IO io);
double  r_revise(double x);
void f_lab2xyz(double l, double a, double b, double& x, double& y, double& z, double Xn, double Yn, double Zn);
cv::Mat lab2xyz(cv::Mat lab, IO io);
cv::Mat rgb2gray(cv::Mat rgb);
cv::Mat xyz2xyz(cv::Mat xyz, IO sio, IO dio);
cv::Mat lab2lab(cv::Mat lab, IO sio, IO dio);
cv::Mat gammaCorrection(cv::Mat src, double K);
cv::Mat mult(cv::Mat xyz, cv::Mat ccm);
cv::Mat mult3D(cv::Mat xyz, cv::Mat ccm);
#endif
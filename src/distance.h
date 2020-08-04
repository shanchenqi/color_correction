#ifndef DISTANCE_H
#define DISTANCE_H
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <map>
#include <tuple>
#include <cmath>

using namespace std;
using namespace cv;

#define PI  3.1415926535897

struct LAB
{
	/* Lightness */
	double l;
	/* Color-opponent a dimension */
	double a;
	/* Color-opponent b dimension */
	double b;
};
using LAB = struct LAB;
double deltaECIEDE76(const LAB& lab1, const LAB& lab2);
double toRad(double degree);
double toDeg(double rad);
double deltacECMC(const LAB& lab1, const LAB& lab2, double kL , double kC );
double deltacECIEDE94(const LAB& lab1, const LAB& lab2, double kH = 1.0, double kC = 1.0, double kL = 1.0, double k1 = 0.045, double k2 = 0.015);
double deltacECIEDE2000(const LAB& lab1, const LAB& lab2, double kL = 1.0, double kC = 1.0, double kH = 1.0);
cv::Mat distance_s(cv::Mat Lab1, cv::Mat Lab2, string distance);
#endif
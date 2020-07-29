
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <map>
#include <tuple>
#include <cmath>
#include "distance.h"


using namespace std;
using namespace cv;

double deltaECIEDE76(const LAB& lab1, const LAB& lab2)
{
	double lab_de76 = sqrtl(powl(lab1.l - lab2.l, 2.0) + powl(lab1.a - lab2.a, 2.0) + powl(lab1.b - lab2.b, 2.0));
	return lab_de76;
}

double toRad(double degree)
{
	return degree * (PI / 180.0);
}

double toDeg(double rad) 
{
	return ((180.0 / PI) * rad);
}

double deltacECMC(const LAB& lab1, const LAB& lab2, double kL = 1, double kC = 1)
{
	double dL = lab2.l - lab1.l;
	double da = lab2.a - lab1.a;
	double db = lab2.b - lab1.b;
	double c1 = sqrtl(powl(lab1.a, 2.0) + powl(lab1.b, 2.0));
	double c2 = sqrtl(powl(lab2.a, 2.0) + powl(lab2.b, 2.0));
	double avg_c = (c1 + c2) / 2.0;
	double sqrt_c_pow = sqrtl(powl(avg_c, 7) / (powl(avg_c, 7) + powl(25, 7)));
	double temp_a1 = lab1.a + lab1.a / 2.0 * (1.0 - sqrt_c_pow);
	double h1;
	if (lab1.b == 0 && temp_a1 == 0)
	{
		h1 = 0.0;
	}
	else 
	{
		h1 = atan2l(lab1.b, temp_a1);
		if (h1 < 0.0) h1 += toRad(360);
	}
	double dC = c2 - c1;
	double dH2 = sqrtl(powl(da, 2) + powl(db, 2) - powl(dC, 2));
	double F = powl(c1, 2) / sqrtl(powl(c1, 4) + 1900);
	double T;
	h1 > toRad(164) && h1 <= toRad(345) ? T = 0.56 + abs(0.2 * cosl(h1 + toRad(168))) : T = 0.36 + abs(0.4 * cosl(h1 + toRad(35)));
	double sL;
	lab1.l < 16 ? sL = 0.511 : sL = (0.040975 * lab1.l) / (1.0 + 0.01765 * lab1.l);
	double sC = (0.0638 * c1) / (1.0 + 0.0131 * c1) + 0.638;
	double sH = sC * (F * T + 1.0 - F);
	return sqrtl(powl(dL / (kL * sL), 2.0) + powl(dC / (kC * sC), 2.0) + powl(dH2 / sH, 2.0));
}

double deltacECIEDE94(const LAB& lab1, const LAB& lab2, double kH , double kC, double kL, double k1, double k2)//todo 声明放在头文件
{
	double dl = lab1.l - lab2.l;
	double c1 = sqrtl(powl(lab1.a, 2) + powl(lab1.b, 2));
	double c2 = sqrtl(powl(lab2.a, 2) + powl(lab2.b, 2));
	double dc = c1 - c2;
	double da = lab2.a - lab1.a;
	double db = lab2.b - lab1.b;
	double dh = powl(da, 2) + powl(db, 2) - powl(dc, 2);
	double sc = 1.0 + k1 * c1;
	double sh = 1.0 + k2 * c1;
	int sl = 1;
	double res = powl(dl /(kL * sl), 2) + powl(dc / (kC * sc), 2) + dh / powl(kH * sh, 2);
	return res > 0 ? sqrtl(res) : 0;
}

double deltacECIEDE2000(const LAB& lab1, const LAB& lab2, double kL, double kC, double kH)
{
	double avg_l = (lab1.l + lab2.l) / 2.0;
	double c1 = sqrtl(powl(lab1.a, 2.0) + powl(lab1.b, 2.0));
	double c2 = sqrtl(powl(lab2.a, 2.0) + powl(lab2.b, 2.0));
	double avg_c = (c1 + c2) / 2.0;
	double sqrt_c_pow = sqrtl(powl(avg_c, 7) / (powl(avg_c, 7) + powl(25, 7)));
	double temp_a1 = lab1.a + lab1.a / 2.0 * (1.0 - sqrt_c_pow);
	double temp_a2 = lab2.a + lab2.a / 2.0 * (1.0 - sqrt_c_pow);
	double temp_c1 = sqrtl(powl(temp_a1, 2) + powl(lab1.b, 2));
	double temp_c2 = sqrtl(powl(temp_a2, 2) + powl(lab2.b, 2));
	double temp_avg_c = (temp_c1 + temp_c2) / 2.0;
	double h1;
	if (lab1.b == 0 && temp_a1 == 0) 
	{
		h1 = 0.0;
	}
	else 
	{
		h1 = atan2l(lab1.b, temp_a1);
		if (h1 < 0.0) h1 += toRad(360);
	}
	double h2;
	if (lab2.b == 0 && temp_a2 == 0) 
	{
		h2 = 0.0;
	}
	else 
	{
		h2 = atan2l(lab2.b, temp_a2);
		if (h2 < 0.0) h2 += toRad(360);
	}
	double H;
	if (abs(h1 - h2) <= toRad(180.0)) 
	{
		H = (h1 + h2) / 2.0;
	}
	else if (abs(h1 - h2) > toRad(180) && h1 + h2 < toRad(360)) 
	{
		H = (h1 + h2 + toRad(360)) / 2.0;
	}
	else {
		H = (h1 + h2 - toRad(360)) / 2.0;
	}
	double delta_h;
	double delta_L = lab2.l - lab1.l;
	double delta_C = c2 - c1;
	double delta_H;
	if (c1 * c2 == 0) {
		delta_h = 0;
	}
	else 
	{
		if (abs(h2 - h1) <= toRad(180))
		{
			delta_h = h2 - h1;
		}
		else if (abs(h2 - h1) > toRad(180) && h2 <= h1)
		{
			delta_h = h2 - h1 + toRad(360);
		}
		else
		{
			delta_h = h2 - h1 - toRad(360);
		}
	}
	delta_H = 2.0 * sqrtl(c1 * c2) * sinl(delta_h / 2.0);
	double sC = 1.0 + 0.045 * avg_c;
	double sH = 1.0 + 0.015 * avg_c * (1.0 - 0.17 * cosl(H - toRad(30)) + 0.24 * cosl(2.0 * H) + 0.32 * cosl(3.0 * H + toRad(6.0)) - 0.2 * cosl(4.0 * H - toRad(63.0)));
	double sL = 1.0 + ((0.015 * powl(avg_l - 50.0, 2.0)) / sqrtl(20.0 + powl(avg_l - 50.0, 2.0)));
	double rt = -2.0 * sqrt_c_pow * sinl(toRad(60.0) * expl(-powl((H - toRad(275.0)) / toRad(25.0), 2.0)));
	double res = (powl(delta_L / (kL * sL), 2.0) + powl(delta_C / (kC * sC), 2.0) + powl(delta_H / (kH * sH), 2.0) + rt * (delta_C / (kC * sC)) * (delta_H / (kH * sH)));
	return res > 0 ? sqrtl(res) : 0;
}

cv::Mat distance_s(cv::Mat Lab1, cv::Mat Lab2, string distance)
{
	cv::Mat distance_lab(Lab1.rows, Lab1.cols, CV_64FC1);
	for (int i = 0; i < Lab1.rows; i++)
	{
		for (int j = 0; j < Lab1.cols; j++)
		{
			double l1 = Lab1.at<Vec3d>(i, j)[0];
			double a1 = Lab1.at<Vec3d>(i, j)[1];
			double b1 = Lab1.at<Vec3d>(i, j)[2];
			double l2 = Lab2.at<Vec3d>(i, j)[0];
			double a2 = Lab2.at<Vec3d>(i, j)[1];
			double b2 = Lab2.at<Vec3d>(i, j)[2];			
			LAB lab1 = { l1,a1,b1 };
			LAB lab2 = { l2,a2,b2 };
			if (distance == "de94")
			{
				distance_lab.at<double>(i, j) = deltacECIEDE94(lab1, lab2);
			}
			else if (distance == "de76" || distance == "rgbl" || distance == "rgb") 
			{
				distance_lab.at<double>(i, j) = deltaECIEDE76(lab1, lab2);
			}
			else if (distance == "cmc") 
			{
				distance_lab.at<double>(i, j) = deltacECMC(lab1, lab2);
			}
			else if (distance == "de00")
			{
				distance_lab.at<double>(i, j) = deltacECIEDE2000(lab1, lab2);
			}
			else
			{
				throw  "wrong distance type" ;
			}
		}
	}
	return distance_lab;
}



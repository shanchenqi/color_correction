#ifndef Distance_H
#define Distance_H
//#include <string>
//#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
//#include <map>
//#include <tuple>
//#include <cmath>

namespace cv {
	namespace ccm {

		enum DISTANCE_TYPE {
			CIE76,
			CIE94_GRAPHIC_ARTS,
			CIE94_TEXTILES,
			CIE2000,
			CMC_1TO1,
			CMC_2TO1,
			RGB,
			RGBL
		};

		//double to_rad(double degree);
		//double to_deg(double rad);

		double delta_cie76(Vec3d lab1, Vec3d lab2) {
			return norm(lab1 - lab2);
		};

		double delta_cie94(Vec3d lab1, Vec3d lab2, double kH = 1.0, double kC = 1.0, double kL = 1.0, double k1 = 0.045, double k2 = 0.015) {
			double dl = lab1[0] - lab2[0];
			double c1 = sqrt(pow(lab1[1], 2) + pow(lab1[2], 2));
			double c2 = sqrt(pow(lab2[1], 2) + pow(lab2[2], 2));
			double dc = c1 - c2;
			double da = lab1[1] - lab2[1];
			double db = lab1[2] - lab2[2];
			double dh = pow(da, 2) + pow(db, 2) - pow(dc, 2);
			double sc = 1.0 + k1 * c1;
			double sh = 1.0 + k2 * c1;
			double sl = 1.0;
			double res = pow(dl / (kL * sl), 2) + pow(dc / (kC * sc), 2) + dh / pow(kH * sh, 2);

			return res > 0 ? sqrt(res) : 0;
		}

		double delta_cie94_graphic_arts(Vec3d lab1, Vec3d lab2) {
			return delta_cie94(lab1, lab2);
		}

		double to_rad(double degree) { return degree / 180 * CV_PI; };

		double delta_cie94_textiles(Vec3d lab1, Vec3d lab2) {
			return delta_cie94(lab1, lab2, 1.0, 1.0, 2.0, 0.048, 0.014);
		}

		double delta_ciede2000_(Vec3d lab1, Vec3d lab2, double kL = 1.0, double kC = 1.0, double kH = 1.0) {
			double delta_L_apo = lab2[0] - lab1[0];
			double l_bar_apo = (lab1[0] + lab2[0]) / 2.0;
			double C1 = sqrt(pow(lab1[1], 2) + pow(lab1[2], 2));
			double C2 = sqrt(pow(lab2[1], 2) + pow(lab2[2], 2));
			double C_bar = (C1 + C2) / 2.0;
			double G = sqrt(pow(C_bar, 7) / (pow(C_bar, 7) + pow(25, 7)));
			double a1_apo = lab1[1] + lab1[1] / 2.0 * (1.0 - G);
			double a2_apo = lab2[1] + lab2[1] / 2.0 * (1.0 - G);
			double C1_apo = sqrt(pow(a1_apo, 2) + pow(lab1[2], 2));
			double C2_apo = sqrt(pow(a2_apo, 2) + pow(lab2[2], 2));
			double C_bar_apo = (C1_apo + C2_apo) / 2.0;
			double delta_C_apo = C2_apo - C1_apo;
			//h1 and h2
			double h1_apo;
			if (C1_apo == 0) {
				h1_apo = 0.0;
			}
			else {
				h1_apo = atan2(lab1[2], a1_apo);
				if (h1_apo < 0.0) h1_apo += 2. * CV_PI;
			}
			double h2_apo;
			if (C2_apo == 0) {
				h2_apo = 0.0;
			}
			else {
				h2_apo = atan2(lab2[2], a2_apo);
				if (h2_apo < 0.0) h2_apo += 2. * CV_PI;
			}
			//delta_h_apo
			double delta_h_apo;
			if (abs(h2_apo - h1_apo) <= CV_PI)
			{
				delta_h_apo = h2_apo - h1_apo;
			}
			else if (h2_apo <= h1_apo)
			{
				delta_h_apo = h2_apo - h1_apo + 2. * CV_PI;
			}
			else
			{
				delta_h_apo = h2_apo - h1_apo - 2. * CV_PI;
			}
			//H_apo
			double H_bar_apo;
			if (C1_apo == 0 || C2_apo == 0) {
				H_bar_apo = h1_apo + h2_apo;
			}
			else if (abs(h1_apo - h2_apo) <= CV_PI) {
				H_bar_apo = (h1_apo + h2_apo) / 2.0;
			}
			else if (h1_apo + h2_apo < 2. * CV_PI) {
				H_bar_apo = (h1_apo + h2_apo + 2. * CV_PI) / 2.0;
			}
			else {
				H_bar_apo = (h1_apo + h2_apo - 2. * CV_PI) / 2.0;
			}
			//delta_H_apo
			double delta_H_apo = 2.0 * sqrt(C1_apo * C2_apo) * sin(delta_h_apo / 2.0);


			//double delta_H;

			//delta_H_apo = 2.0 * sqrt(C1 * C2) * sin(delta_h_apo / 2.0);
			double T = 1.0 - 0.17 * cos(H_bar_apo - to_rad(30.)) + 0.24 * cos(2.0 * H_bar_apo) + 0.32 * cos(3.0 * H_bar_apo + to_rad(6.0)) - 0.2 * cos(4.0 * H_bar_apo - to_rad(63.0));
			double sC = 1.0 + 0.045 * C_bar_apo;
			double sH = 1.0 + 0.015 * C_bar_apo * T;
			double sL = 1.0 + ((0.015 * pow(l_bar_apo - 50.0, 2.0)) / sqrt(20.0 + pow(l_bar_apo - 50.0, 2.0)));
			double RT = -2.0 * G * sin(to_rad(60.0) * exp(-pow((H_bar_apo - to_rad(275.0)) / to_rad(25.0), 2.0)));
			double res = (pow(delta_L_apo / (kL * sL), 2.0) + pow(delta_C_apo / (kC * sC), 2.0) + pow(delta_H_apo / (kH * sH), 2.0) + RT * (delta_C_apo / (kC * sC)) * (delta_H_apo / (kH * sH)));
			//return  sqrt(res);
			return res > 0 ? sqrt(res) : 0;
		}


		double delta_ciede2000(Vec3d lab1, Vec3d lab2) {
			return delta_ciede2000_(lab1, lab2);
		}


		double delta_cmc(Vec3d lab1, Vec3d lab2, double kL = 1, double kC = 1) {
			double dL = lab2[0] - lab1[0];
			double da = lab2[1] - lab1[1];
			double db = lab2[2] - lab1[2];
			double C1 = sqrt(pow(lab1[1], 2.0) + pow(lab1[2], 2.0));
			double C2 = sqrt(pow(lab2[1], 2.0) + pow(lab2[2], 2.0));
			//double C_bar = (C1 + C2) / 2.0;
			//double G = sqrt(pow(C_bar, 7) / (pow(C_bar, 7) + pow(25, 7)));
			//double temp_a1 = lab1[1] + lab1[1] / 2.0 * (1.0 - G);
			double dC = C2 - C1;
			double dH = sqrt(pow(da, 2) + pow(db, 2) - pow(dC, 2));

			double H1;
			if (C1 == 0.) {
				H1 = 0.0;
			}
			else {
				H1 = atan2(lab1[2], lab1[1]);
				if (H1 < 0.0) H1 += 2. * CV_PI;
			}


			double F = pow(C1, 2) / sqrt(pow(C1, 4) + 1900);
			double T = (H1 > to_rad(164) && H1 <= to_rad(345)) ? 0.56 + abs(0.2 * cos(H1 + to_rad(168))) : 0.36 + abs(0.4 * cos(H1 + to_rad(35)));;
			double sL = lab1[0] < 16. ? 0.511 : (0.040975 * lab1[0]) / (1.0 + 0.01765 * lab1[0]);;
			double sC = (0.0638 * C1) / (1.0 + 0.0131 * C1) + 0.638;
			double sH = sC * (F * T + 1.0 - F);

			return sqrt(pow(dL / (kL * sL), 2.0) + pow(dC / (kC * sC), 2.0) + pow(dH / sH, 2.0));
		}


		double delta_cmc_1to1(Vec3d lab1, Vec3d lab2) {
			return delta_cmc(lab1, lab2);
		}

		double delta_cmc_2to1(Vec3d lab1, Vec3d lab2) {
			return delta_cmc(lab1, lab2, 2, 1);
		}


		//double delta_cmc_2to1(Vec3d lab1, Vec3d lab2) {
		//	return delta_cmc(lab1, lab2, 2, 1);
		//}

		Mat distance(Mat src, Mat ref, DISTANCE_TYPE distance_type) {
			switch (distance_type)
			{
			case cv::ccm::CIE76:
				return _distancewise(src, ref, delta_cie76);
				break;
			case cv::ccm::CIE94_GRAPHIC_ARTS:
				return _distancewise(src, ref, delta_cie94_graphic_arts);
				break;
			case cv::ccm::CIE94_TEXTILES:
				return _distancewise(src, ref, delta_cie94_textiles);
				break;
			case cv::ccm::CIE2000:
				return _distancewise(src, ref, delta_ciede2000);
				break;
			case cv::ccm::CMC_1TO1:
				return _distancewise(src, ref, delta_cmc_1to1);
				break;
			case cv::ccm::CMC_2TO1:
				return _distancewise(src, ref, delta_cmc_2to1);
				break;
			case cv::ccm::RGB:
				return _distancewise(src, ref, delta_cie76);
				break;
			case cv::ccm::RGBL:
				return _distancewise(src, ref, delta_cie76);
				break;
			default:
				throw;
				break;
			}
		};


	}
}

//double deltaE_cie76(const LAB& lab1, const LAB& lab2);

//double to_rad(double degree);

//double to_deg(double rad);

//double deltacE_cmc(const LAB& lab1, const LAB& lab2, double kL, double kC);

//double deltacE_ciede94(const LAB& lab1, const LAB& lab2, double kH, double kC, double kL, double k1, double);

//double deltacE_ciede2000(const LAB& lab1, const LAB& lab2, double kL, double kC, double kH);
#endif

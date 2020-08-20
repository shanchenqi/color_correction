#include "linearize.h"

using namespace cv;
using namespace ccm;
int main() {

	 Mat s = (Mat_<Vec3d>(24, 1) <<
		Vec3d(214.11, 98.67, 37.97),
		Vec3d(231.94, 153.1, 85.27),
		Vec3d(204.08, 143.71, 78.46),
		Vec3d(190.58, 122.99, 30.84),
		Vec3d(230.93, 148.46, 100.84),
		Vec3d(228.64, 206.97, 97.5),
		Vec3d(229.09, 137.07, 55.29),
		Vec3d(189.21, 111.22, 92.66),
		Vec3d(223.5, 96.42, 75.45),
		Vec3d(201.82, 69.71, 50.9),
		Vec3d(240.52, 196.47, 59.3),
		Vec3d(235.73, 172.13, 54.),
		Vec3d(131.6, 75.04, 68.86),
		Vec3d(189.04, 170.43, 42.05),
		Vec3d(222.23, 74., 71.95),
		Vec3d(241.01, 199.1, 61.15),
		Vec3d(224.99, 101.4, 100.24),
		Vec3d(174.58, 152.63, 91.52),
		Vec3d(248.06, 227.69, 140.5),
		Vec3d(241.15, 201.38, 115.58),
		Vec3d(236.49, 175.87, 88.86),
		Vec3d(212.19, 133.49, 54.79),
		Vec3d(181.17, 102.94, 36.18),
		Vec3d(115.1, 53.77, 15.23));
	double gamma = 2.2;
	int deg = 3;
	Mat mask = saturate(s/255, 0.05, 0.93);
	
	//Linear_gray<LogPolyfit> TEST(deg, s/255, Macbeth_D50_2, mask, sRGB);
	Linear_gamma TEST(gamma);
	std::cout << TEST.linearize(s/255) << std::endl;
}
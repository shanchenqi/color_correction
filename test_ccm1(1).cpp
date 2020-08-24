#include "ccm.h"

using namespace cv;
using namespace std;
using namespace cv::ccm;

void test(int tag) {
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

	Color color = Macbeth_D65_2;
	std::vector<double> saturated_threshold = { 0, 0.98 };
	cv::Mat weight_list;
	std::string filename = "input1.png";
		

	/*ColorCorrectionModel_3x3 p1(s/255, color, sRGB, CIE2000, GAMMA, 2.2, 3,
		saturated_threshold, weight_list, 0, LEAST_SQUARE, 10000, 1e-4);*/
	/*ColorCorrectionModel p2(s / 255, color, AdobeRGB, CIE2000, GAMMA, 2.2, 3,
		saturated_threshold, weight_list, 0, LEAST_SQUARE, 5000, 1e-4, 12);*/
	ColorCorrectionModel p3(s / 255, color, WideGamutRGB, CIE2000, GRAYPOLYFIT, 2.2, 3,
		saturated_threshold, weight_list, 0, LEAST_SQUARE, 5000, 1e-4,12);
	/*ColorCorrectionModel p4(s / 255, color, ProPhotoRGB, RGBL, GRAYLOGPOLYFIT, 2.2, 3,
		saturated_threshold, weight_list, 0, LEAST_SQUARE, 5000, 1e-4,9);*/
	/*ColorCorrectionModel p5(s / 255, color, DCI_P3_RGB, RGB, IDENTITY_, 2.2, 3,
		saturated_threshold, weight_list, 0, LEAST_SQUARE, 5000, 1e-4,9);*/
	/*ColorCorrectionModel p6(s / 255, color, AppleRGB, CIE2000, COLORPOLYFIT, 2.2, 2,
		saturated_threshold, weight_list, 2, LEAST_SQUARE, 5000, 1e-4,9);*/
	/*ColorCorrectionModel p7(s / 255, color, REC_2020_RGB, CIE94_GRAPHIC_ARTS, COLORLOGPOLYFIT, 2.2, 3,
		saturated_threshold, weight_list, 0, LEAST_SQUARE, 5000, 1e-4,9);*/


	//std::cout <<"ccm1"<< p1.ccm << std::endl;
	//std::cout << "ccm2" << p2.ccm << std::endl;
     std::cout << "ccm3" << p3.ccm << std::endl;
	//std::cout << "ccm4" << p4.ccm << std::endl;
	//std::cout << "ccm5" << p5.ccm << std::endl;
    //std::cout << "ccm6" << p6.ccm << std::endl;
	//std::cout << "ccm7" << p7.ccm << std::endl;
	//std::cout << "ccm" << p.dst.colors << std::endl;


	//Mat img1 = p1.infer_image(filename);
	//Mat img2 = p2.infer_image(filename);
	//Mat img3 = p3.infer_image(filename);
	//Mat img4 = p4.infer_image(filename);
	//Mat img5 = p5.infer_image(filename, true);
	//Mat img6 = p6.infer_image(filename);
	//Mat img7 = p7.infer_image(filename);
	//imwrite("output.png", img5);
}

int main() {
	test(1);
	return 0;
}
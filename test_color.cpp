#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <map>
#include <string>
#include "color.h"
#include "utils.h"
//#include "time.h"

using namespace cv;
using namespace cv::ccm;
using namespace std;

void testsrgb(Color color) {
	Color color_rgb = color.to(sRGB);
	cout << "color_rgb.colors" << color_rgb.colors << endl;
	Color color_rgbl = color.to(sRGBL);
	cout << "color_rgbl.colors" << color_rgbl.colors << endl;
	Color color_xyz = color.to(XYZ_D65_2);
	cout << "color_xyz.colors" << color_xyz.colors << endl;
	Color color_lab = color.to(Lab_D65_2);
	cout << "color_lab.colors" << color_lab.colors << endl;
	Color color_xyz_d50 = color.to(XYZ_D50_2);
	cout << "color_xyz_d50.colors" << color_xyz_d50.colors << endl;
	Color color_lab_d50 = color.to(Lab_D50_2);
	cout << "color_lab_d50.colors" << color_lab_d50.colors << endl;
}
int main() {
	Mat test1 = (Mat_<Vec3d>(1, 1) << Vec3d(0.3, 0.2, 0.5));
	Color colortest1(test1, sRGB);
	Mat test2 = (Mat_<Vec3d>(1, 1) << Vec3d(0.3, 0.2, 0.5));
	Color colortest2(test2, XYZ_D50_2);

	/*cout << "CIE2000 " << colortest1.diff(colortest2, D65_2, CIE2000) << endl;
	cout << "CIE76 " << colortest1.diff(colortest2, D65_2, CIE76) << endl;
	cout << "CIE94_GRAPHIC_ARTS " << colortest1.diff(colortest2, D65_2, CIE94_GRAPHIC_ARTS) << endl;
	cout << "CIE94_TEXTILES " << colortest1.diff(colortest2, D65_2, CIE94_TEXTILES) << endl;
	cout << "CMC_1TO1 " << colortest1.diff(colortest2, D65_2, CMC_1TO1) << endl;
	cout << "CMC_2TO1 " << colortest1.diff(colortest2, D65_2, CMC_2TO1) << endl;
	cout << "RGB " << colortest1.diff(colortest2, D65_2, RGB) << endl;
	cout << "RGBL " << colortest1.diff(colortest2, D65_2, RGBL) << endl;*/
	
	//sRGB, sRGBL = bind(sRGB_);
	testsrgb(colortest1);
	/*cout << "sRGB.M_to: " << sRGB.M_from << endl;
	cout << "AdobeRGB.M_to: " << AdobeRGB.M_from << endl;
	cout << "WideGamutRGB.M_to: " << WideGamutRGB.M_from << endl;
	cout << "ProPhotoRGB.M_to: " << ProPhotoRGB.M_from << endl;
	cout << "XYZ::_cam(D50_2, D65_2): " << XYZ_D65_2._cam(D50_2, D65_2) << endl;
	cout << "XYZ::_cam(D55_2, D50_2, VON_KRIS): " << XYZ_D65_2._cam(D55_2, D50_2, VON_KRIS) << endl;
	cout << "XYZ::_cam(D65_2, D65_2): " << XYZ_D65_2._cam(D65_2, D65_2) << endl;
	cout << "XYZ::_cam(D65_2, D50_2, IDENTITY): " << XYZ_D65_2._cam(D65_2, D50_2, IDENTITY) << endl;

	*///cout << color_rgb.colors << endl;
	/*Mat test = (Mat_<Vec3d>(1, 1) << Vec3d(0.2, 0.3, 0.4));
	Mat rgb2graytest = rgb2gray(test);
	cout << rgb2graytest << endl;*/
	/*Mat test = (Mat_<Vec3d>(4, 1) << Vec3d(0.8, -0.5, 0.6),
		Vec3d(0.2, 0.9, -0.9),
		Vec3d(1., -0.2, 0.4),
		Vec3d(-0.4, 0.1, 0.3));
	Mat res= gamma_correction(test, 2.2);
	cout << res << endl;*/
	return 0;

}

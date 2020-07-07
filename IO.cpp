#include "IO.h"
#include <string>
#include <iostream>
#include "utils.h"

map<IO, vector<double> >  illuminants_xy;

vector<double>(*xyY)(double, double, double);

IO::IO(string illuminant,int observer){
	m_illuminant = illuminant;
	m_observer = observer;
   

}
IO::~IO() {}

bool test() {
	return 0;
}


bool IO::operator<(const IO &other )const {
	{
		return (m_illuminant < other.m_illuminant || (m_illuminant == other.m_illuminant) && (m_observer<other.m_observer));
	}
}

//IO  A_2("A", 2), A_10("A", 10), D50_2("D50", 2), D50_10("D50", 10), D55_2("D55", 2), D55_10("D55", 10), D65_2("D65", 2), D65_10("D65", 10), D75_2("D75", 2), D75_10("D75", 10), E_2("E", 2), E_10("E", 10);

map <IO, vector<double>> get_xyz_ref_white()
{
    illuminants_xy[A_2] = { 0.44757, 0.40745 };
    illuminants_xy[A_10] = { 0.45117, 0.40594 };
    illuminants_xy[D50_2] = { 0.34567, 0.35850 };
    illuminants_xy[D50_10] = { 0.34773, 0.35952 };
    illuminants_xy[D55_2] = { 0.33242, 0.34743 };
    illuminants_xy[D55_10] = { 0.33411, 0.34877 };
    illuminants_xy[D65_2] = { 0.31271, 0.32902 };
    illuminants_xy[D65_10] = { 0.31382, 0.33100 };
    illuminants_xy[D75_2] = { 0.29902, 0.31485 };
    illuminants_xy[E_2] = { 1 / 3, 1 / 3 };
    illuminants_xy[E_10] = { 1 / 3, 1 / 3 };
    map <IO, vector<double>> get_illuminant();
    map <IO, vector<double>>  illuminants;
    illuminants = get_illuminant();
    return illuminants;

    
}
vector<double> xyY2XYZ(double x, double y, double Y = 1) {
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

map <IO, vector<double>> get_illuminant() {
    map <IO, vector<double> >  illuminants;
    xyY = xyY2XYZ;
    map<IO, vector<double>>::iterator it;
    it = illuminants_xy.begin();
    for (it; it != illuminants_xy.end(); it++)
    {

        double x = it->second[0];
        double y = it->second[1];
        double Y = 1;
        vector<double> res;

        res = xyY(x, y, Y);
        illuminants[it->first] = res;

        illuminants[IO("D65", 2)] = { 0.95047, 1.0, 1.08883 };
        illuminants[IO("D65", 10)] = { 0.94811, 1., 1.07304 };
    }
    return illuminants;
}
map <tuple<IO, IO, string>, cv::Mat > CAMs;

Mat cam(IO sio, IO dio,string method = "Bradford") {
    map <IO, vector<double>> illuminants= get_xyz_ref_white();
    Mat  Von_Kries = (Mat_<double>(3, 3) << 0.40024, 0.7076, -0.08081,
        -0.2263, 1.16532, 0.0457,
        0., 0., 0.91822);
    Mat Bradford = (Mat_<double>(3, 3) << 0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296);
    //map <tuple<IO, IO, string>, cv::Mat > CAMs;
    map <String, vector< cv::Mat >> MAs;
    MAs["Identity"] = { Mat::eye(cv::Size(3,3),CV_64FC1) , Mat::eye(cv::Size(3,3),CV_64FC1) };
    MAs["Von_Krie"] = { Von_Kries ,Von_Kries.inv() };
    MAs["Bradford"] = { Bradford ,Bradford.inv() };
    
    if (CAMs.count(make_tuple(dio, sio, method)) == 1) {
        return CAMs[make_tuple(dio, sio, method)];
    }
    else {
        Mat XYZws = Mat(illuminants[dio]); 
        Mat XYZWd = Mat(illuminants[sio]);
        Mat MA = MAs[method][0];
        Mat MA_inv = MAs[method][1];
        Mat MA_res1 = MA*XYZws;
        Mat MA_res2 = MA*XYZWd;
        Mat MA_res3 = MA_res1 / MA_res2;
        cv::Mat me = cv::Mat::eye(cv::Size(3, 3), CV_64FC1);
        me.at<double>(0,0)= MA_res3.at<double>(0, 0)  ;
        me.at<double>(1, 1)= MA_res3.at<double>(1, 0) ;
        me.at<double>(2, 2) = MA_res3.at<double>(2, 0) ;
        Mat M=  MA_inv*(me);
        CAMs[make_tuple(dio, sio, method)] = M;
        return M;
    }
}




#ifndef Utils_H
#define Utils_H
#include <functional>
#include <vector>
#include <string>
#include <iostream>
#include "io.hpp"

namespace cv {
    namespace ccm {

        //typedef double (*DoubleFunc)(double);

        template<typename F>
        Mat _elementwise(Mat src, F&& lambda) {
            Mat dst = src.clone();
            const int channel = src.channels();
            
            switch (channel)
            {
            case 1:
            {
                MatIterator_<double> it, end;
                for (it = dst.begin<double>(), end = dst.end<double>(); it != end; ++it)
                    (*it) = lambda((*it));
                break;
            }
            case 3:
            {
                MatIterator_<Vec3d> it, end;
                for (it = dst.begin<Vec3d>(), end = dst.end<Vec3d>(); it != end; ++it) {
                    for (int j = 0; j < 3; j++) {
                        (*it)[j] = lambda((*it)[j]);
                    }
                }
            }
            }
           
            return dst;
        }
     /*   template<typename F>
        Mat _elementwise1(Mat src, F&& lambda) {
            Mat dst = src.clone();
            std::cout << dst.type() << std::endl;
            MatIterator_<double> it, end;
            for (it = dst.begin<double>(), end = dst.end<double>(); it != end; ++it) {
                
                    (*it)= lambda((*it));
                
            }
            return dst;
        }*/

        template<typename F>
        Mat _channelwise(Mat src, F&& lambda) {
            Mat dst = src.clone();
            MatIterator_<Vec3d> it, end;
            for (it = dst.begin<Vec3d>(), end = dst.end<Vec3d>(); it != end; ++it) {
                *it = lambda(*it);
            }
            return dst;
        }

        template<typename F>
        Mat _distancewise(Mat src, Mat ref, F&& lambda) {
            Mat dst = Mat(src.size(), CV_64FC1);
            MatIterator_<Vec3d> it_src = src.begin<Vec3d>(), end_src = src.end<Vec3d>(),
                it_ref = ref.begin<Vec3d>(), end_ref = ref.end<Vec3d>();
            MatIterator_<double> it_dst = dst.begin<double>(), end_dst = dst.end<double>();
            for (; it_src != end_src; ++it_src, ++it_ref, ++it_dst) {
                *it_dst = lambda(*it_src, *it_ref);
            }
            return dst;
        }


        double _gamma_correction(double element, double gamma) {
            return (element >= 0 ? pow(element, gamma) : -pow((-element), gamma));
        }

        Mat gamma_correction(Mat src, double gamma) {
            return _elementwise(src, [gamma](double element)->double {return _gamma_correction(element, gamma); });
        }

        Mat mask_copyto(Mat src, Mat mask) {
            Mat dst(countNonZero(mask), 1, src.type());
            const int channel = src.channels();
            auto it_mask = mask.begin<uchar>(), end_mask = mask.end<uchar>();
            switch (channel)
            {
            case 1:
            {
                auto it_src = src.begin<double>(), end_src = src.end<double>();
                auto it_dst = dst.begin<double>(), end_dst = dst.end<double>();
                for (; it_src != end_src; it_src++, it_mask++) {
                    if (*it_mask) {
                        (*it_dst) = (*it_src);
                        ++it_dst;
                    }
                }
                break;
            }
            case 3:
            {
                auto it_src = src.begin<Vec3d>(), end_src = src.end<Vec3d>();
                auto it_dst = dst.begin<Vec3d>(), end_dst = dst.end<Vec3d>();
                for (; it_src != end_src; it_src++, it_mask++) {
                    if (*it_mask) {
                        (*it_dst) = (*it_src);
                        ++it_dst;
                    }
                }
            }
            }
            return dst;
        }

        Mat multiple(Mat xyz, Mat ccm) {
            Mat tmp = xyz.reshape(1, xyz.rows * xyz.cols);
            Mat res = tmp * ccm;
            res = res.reshape(res.cols, xyz.rows);
            return res;
        }

        //Mat multiple2(Mat xyz, Mat ccm) {
        //    Mat tmp = xyz.reshape(1, xyz.rows * xyz.cols);
        //    Mat res = tmp * ccm;
        //    //res = res.reshape(3, xyz.rows);
        //    return res;
        //}
        void print(Mat m, std::string s) {
            std::cout << s << m << std::endl;
            std::cout << s << m.size() << std::endl;
        }

        Mat saturate(Mat src, double low, double up) {
            Mat dst = Mat::ones(src.size(), CV_8UC1);
            MatIterator_<Vec3d> it_src = src.begin<Vec3d>(), end_src = src.end<Vec3d>();
          //  MatIterator_<double> it_dst = dst.begin<double>(), end_dst = dst.end<double>();
            MatIterator_<uchar> it_dst = dst.begin<uchar>(), end_dst = dst.end<uchar>();
            for (; it_src != end_src; ++it_src, ++it_dst) {
                for (int i = 0; i < 3; ++i) {
                    if ((*it_src)[i] > up || (*it_src)[i] < low) {
                        *it_dst = 0.;
                        break;
                    }
                }
            }
            return dst;
        }

        Mat M_gray = (Mat_<double>(3, 1) << 0.2126, 0.7152, 0.0722);

        Mat rgb2gray(Mat rgb) {
            return multiple(rgb, M_gray);
        }

    }
}

#endif
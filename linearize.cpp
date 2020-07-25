#include "linearize.h"

Mat Linear::linearize(Mat inp)
{
    cout<<"***************linear**************"<<endl;
    return inp;
}
//Mat Linear::mask_copyto(Mat src, Mat mask) {
Mat mask_copyto(Mat src, Mat mask) {
    Mat src_(countNonZero(mask), 1, src.type());
    int countone = 0;
    
    for (int i = 0; i < mask.rows; i++) {
        if (mask.at<double>(i, 0)) {
            for (int c = 0; c < src.channels(); c++) {
                src_.at<Vec3d>(countone, 0)[c] = src.at<Vec3d>(i, 0)[c];
            }
            countone++;
        };
    }
    return src_;
}


Linear_gamma::Linear_gamma(float gamma_, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold)
{
    this->gamma = gamma_;
}

Mat Linear_gamma::linearize(Mat inp) {
    return gamma_correction(inp, gamma);
}


Linear_color_polyfit::Linear_color_polyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {
    Mat mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    Mat src_(countNonZero(mask), 1 ,src.type());
    Mat dst_(countNonZero(mask),1, cc.rgbl.type());
    this->deg = deg;
  
    this->src = mask_copyto(src, mask);
    this->dst = mask_copyto(cc.rgbl, mask);
    //src.copyTo(src_, mask);
   // src.setTo(src_, mask);
   // cc.rgbl.setTo(dst_, mask);
   // cc.rgbl.copyTo(dst_, mask);
   // this->src = src_;
   // this->dst = dst_;
  //  src.copyTo(src, mask);
 //   cc.rgbl.copyTo(dst, mask);
  //  cout << "src1234" << src << endl;
  //  src = src_.clone();
  //  dst = dst_.clone();
    calc();
    //cout <<"Linear_color_polyfit" << endl;
}


void Linear_color_polyfit::calc(void)
{
    Mat sChannels[3];
    Mat dChannels[3];
    
    split(this->src, sChannels);
    split(this->dst, dChannels);
    Mat rs = sChannels[0];
    Mat gs = sChannels[1];
    Mat bs = sChannels[2];
    Mat rd = dChannels[0];
    Mat gd = dChannels[1];
    Mat bd = dChannels[2];
    
    pr = polyfit(rs, rd, deg);
    pg = polyfit(gs, gd, deg);
    pb = polyfit(bs, bd, deg);


}


Mat Linear_color_polyfit::linearize(Mat inp)
{
    Mat inpChannels[3];

    split(inp, inpChannels);
    vector<Mat> channel;
    Mat res;

    channel.push_back(poly1d(inpChannels[0], pr, deg));
    channel.push_back(poly1d(inpChannels[1], pg, deg));
    channel.push_back(poly1d(inpChannels[2], pb, deg));
    merge(channel, res);
    return res;
 
}


Linear_color_logpolyfit::Linear_color_logpolyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {
    Mat mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    Mat src_(countNonZero(mask), 1, src.type());
    Mat dst_(countNonZero(mask), 1, cc.rgbl.type());
    this->deg = deg;
    this->src = mask_copyto(src, mask);
    this->dst = mask_copyto(cc.rgbl, mask);
    calc();
}


void Linear_color_logpolyfit::calc(void)
{
   // Mat rs = src.rowRange(0, 1).clone();
  //  Mat gs = src.rowRange(1, 2).clone();
  //  Mat bs = src.rowRange(2, 3).clone();
 //   Mat rd = dst.rowRange(0, 1).clone();
 //   Mat gd = dst.rowRange(1, 2).clone();
  //  Mat bd = dst.rowRange(2, 3).clone();
    Mat sChannels[3];
    Mat dChannels[3];
    split(src, sChannels);
    split(dst, dChannels);
    
    Mat rs = sChannels[0];
    Mat gs = sChannels[1];
    Mat bs = sChannels[2];
    Mat rd = dChannels[0];
    Mat gd = dChannels[1];
    Mat bd = dChannels[2];
    pr = _polyfit(rs, rd, deg);
    pg = _polyfit(gs, gd, deg);
    pb = _polyfit(bs, bd, deg);
}


Mat Linear_color_logpolyfit::linearize(Mat inp)
{
    cout<<"**********color_logpolyfit********** "<< endl;
    Mat channels[3];
    split(inp, channels);
    Mat r = channels[0];
    Mat g = channels[1];
    Mat b = channels[2];
    vector<Mat> channel;
    Mat res;
    channel.push_back(_lin(pr, r,deg));   
    channel.push_back(_lin(pg, g,deg));
    channel.push_back(_lin(pb, b,deg));
    merge(channel, res);
    return res;
}

Linear_gray_polyfit::Linear_gray_polyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {
    Mat mask = saturate(src, saturated_threshold[0], saturated_threshold[1])& ~cc.white_mask;
    Mat src_(countNonZero(mask), 1, src.type());
    Mat dst_(countNonZero(mask), 1, cc.grayl.type());
    this->deg = deg;
    Mat src_gray = mask_copyto(src, mask);
    this->src = rgb2gray(src_gray);
    this->dst = mask_copyto(cc.grayl, mask);
    calc();
}


void Linear_gray_polyfit::calc(void)
{
    this-> p = polyfit(src, dst, deg);
}


Mat Linear_gray_polyfit::linearize(Mat inp)
{
    Mat inpChannels[3];
    split(inp, inpChannels);
    vector<Mat> channel;
    Mat res;
    channel.push_back(poly1d(inpChannels[0], p, deg));
    channel.push_back(poly1d(inpChannels[1], p, deg));
    channel.push_back(poly1d(inpChannels[2], p, deg));
    merge(channel, res);
    return res;
}


Linear_gray_logpolyfit::Linear_gray_logpolyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {
    Mat mask = saturate(src, saturated_threshold[0], saturated_threshold[1]) & ~cc.white_mask;
    Mat src_(countNonZero(mask), 1, src.type());
    Mat dst_(countNonZero(mask), 1, cc.grayl.type());
    this->deg = deg;
    Mat src_gray = mask_copyto(src, mask);
    this->src = rgb2gray(src_gray);
    this->dst = mask_copyto(cc.grayl, mask);
    calc();
}

void Linear_gray_logpolyfit::calc(void)
{
    this->p = _polyfit(src, dst, deg);
}

Mat Linear_gray_logpolyfit::linearize(Mat inp)
{
    Mat inpChannels[3];
    split(inp, inpChannels);
    vector<Mat> channel;
    Mat res;
    channel.push_back(_lin( p, inpChannels[0], deg));
    channel.push_back(_lin( p, inpChannels[1], deg));
    channel.push_back(_lin( p, inpChannels[2], deg));
    merge(channel, res);
    return res;
}


Mat Linear::_polyfit(Mat src, Mat dst, int deg) { 
    Mat mask_ = (src > 0) & (dst > 0);
    mask_.convertTo(mask_, CV_64F);
    Mat src_, dst_;
    src_ = mask_copyto(src, mask_);
    dst_ = mask_copyto(dst, mask_);
    Mat s, d;
    log(src_, s);
    log(dst_, d);
    Mat res = polyfit(s, d, deg);
    return res;
}


Mat Linear::_lin(Mat p, Mat x, int deg) {  
    Mat mask_ = x >= 0;
    Mat y;
    log(x, y);
    y = poly1d(y, p, deg);
    Mat y_;
    exp(y, y_);
    Mat res;
    y_.copyTo(res, mask_);
    return res;
}


Mat Linear::polyfit(Mat src_x, Mat src_y, int order) {
    int npoints = src_x.checkVector(1);
    int nypoints = src_y.checkVector(1);
    Mat_<double> srcX(src_x), srcY(src_y);
    Mat_<double> A = Mat_<double>::ones(npoints, order + 1);
    for (int y = 0; y < npoints; ++y)
    {
        for (int x = 1; x < A.cols; ++x)
        {         
            A.at<double>(y, x) = srcX.at<double>(y) * A.at<double>(y, x - 1);           
        }
    }
    Mat w;
    cv::solve(A, srcY, w, DECOMP_SVD);
    return w;
}

Mat Linear::poly1d(Mat src, Mat w, int deg) {
    Mat res_polyfit(src.size(),src.type());
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            double res = 0;
            for (int d = 0; d <= deg; d++) {
                res += pow(src.at<double>(i, j), d) * w.at<double>(d, 0);
                res_polyfit.at<double>(i, j) = res;
            }      
        }
    }  
    return res_polyfit;
}


Linear* get_linear(string linear, float gamma_, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {
    Linear* p = new Linear(gamma_, deg, src, cc, saturated_threshold);
    if (linear == "Linear") {
        p = new Linear(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "Linear_identity") {
        p = new Linear_identity(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "Linear_gamma") {
        p = new Linear_gamma(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "Linear_color_polyfit") {
        p = new Linear_color_polyfit(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "Linear_color_logpolyfit") {
        p = new Linear_color_logpolyfit(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "Linear_gray_polyfit") {
        p = new Linear_gray_polyfit(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "Linear_gray_logpolyfit") {
        p = new Linear_gray_logpolyfit(gamma_, deg, src, cc, saturated_threshold);
    }
    return p;
}
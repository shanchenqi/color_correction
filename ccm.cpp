#include "ccm.h"
#include "linearize.h"


CCM_3x3::CCM_3x3(Mat src_, Mat dst, string dst_colorspace, string dst_illuminant, int dst_observer, Mat dst_whites, string colorchecker, vector<double> saturated_threshold, string colorspace, string linear_, float gamma, int deg, string distance_, string dist_illuminant, int dist_observer, Mat weights_list, double weights_coeff, bool weights_color, string initial_method, double xtol_, double ftol_)
{
    this->src = src_;
    IO dist_io = IO(dist_illuminant, dist_observer);
    this->cs = get_colorspace(colorspace);
    cs->set_default(dist_io);
    ColorChecker cc_;
    cout << "dst.empty()" << dst.empty() << endl;
    if (!dst.empty()) {
        cc_ = ColorChecker(dst, dst_colorspace, IO(dst_illuminant, dst_observer), dst_whites);
    }
    else if (colorchecker == "Macbeth_D65_2") {
      
        cc_ = ColorChecker(ColorChecker2005_LAB_D65_2, "LAB", IO("D65", 2), Arange_18_24);
    }
    else if (colorchecker == "Macbeth_D50_2") {
        cc_ = ColorChecker(ColorChecker2005_LAB_D50_2, "LAB", IO("D50", 2), Arange_18_24);
    }
    //else //
    this->cc = ColorCheckerMetric(cc_, colorspace, dist_io);

    //linear = get_linear(linear_)(gamma, deg, src, cc, saturated_threshold);
    this->linear = get_linear(linear_);
    Linear linear(gamma, deg, this->src, this->cc, saturated_threshold);
    // Mat weights;
    if (!weights_list.empty()) {
        this->weights = weights_list;
    }
    else if (weights_coeff != 0) {
        Mat dChannels[3];
        split(this->cc.lab, dChannels);
        Mat cc_lab_0 = dChannels[0];
        Mat weights_;
        pow(cc_lab_0, weights_coeff, weights_);
        this->weights = weights_;
    }
    /*if (!weights_list.data) {
        this->weights = weights_list;
    }
    else if (weights_coeff != 0) {
        Mat cc_lab_0 =this-> cc.lab.rowRange(0, 1);
        Mat weights_;
        pow(cc_lab_0, weights_coeff, weights_);
        this->weights = weights_;
    }
    */

    Mat weight_mask = Mat::ones( src.rows, 1, CV_64FC1);
    if (weights_color) {
        weight_mask = this->cc.color_mask;
    }

    Mat saturate_mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
   
    this->mask= (weight_mask) & (saturate_mask);
    this->mask.convertTo(this->mask, CV_64F);
    this->src_rgbl = linear.linearize(this->src);
    this->src_rgb_masked = mask_copyto(this->src, mask);
    src_rgbl_masked = mask_copyto(this->src_rgbl, mask);
    this->dst_rgb_masked = mask_copyto(this->cc.rgb, mask);
    
    this->dst_rgbl_masked = mask_copyto(this->cc.rgbl, mask);
    this->dst_lab_masked = mask_copyto(this->cc.lab, mask);
    if (this->weights.data) {
        this->weights.copyTo(this->weights_masked, this->mask);
        cout << this->weights_masked <<endl;
        this->weights_masked_norm = this->weights_masked / mean(this->weights_masked);
    }
    this->masked_len = this->src_rgb_masked.rows;

    if (initial_method == "white_balance") {
        this->ccm0 = this->initial_white_balance(this->src_rgbl_masked, this->dst_rgbl_masked);
    }
    else if (initial_method == "least_square") {
        cout << "this->src_rgbl_masked" <<this->src_rgbl_masked << "this->dst_rgbl_masked" << this->dst_rgbl_masked << endl;
        this->ccm0 = this->initial_least_square(this->src_rgbl_masked, this->dst_rgbl_masked);
    }

    this->distance = distance_;
    if (this->distance == "rgb") {
        this->calculate_rgb();
    }
    else if (this->distance == "rgbl") {
        this->calculate_rgbl();
    }
    else {
        this->calculate();
    }
    this->prepare();

}

Mat CCM_3x3::initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {
    Mat sChannels[3];
    split(src_rgbl, sChannels);

    Mat dChannels[3];
    split(dst_rgbl, dChannels);

    Scalar rs = sum(sChannels[0]);
    Scalar gs = sum(sChannels[1]);
    Scalar bs = sum(sChannels[2]);
    Scalar rd = sum(dChannels[0]);
    Scalar gd = sum(dChannels[1]);
    Scalar bd = sum(dChannels[2]);
   
    Mat initial_white_balance_ = (Mat_<double>(3, 3) << rd[0] / rs[0], 0, 0, 0, gd[0] / gs[0], 0, 0, 0, bd[0] / bs[0]);
    return initial_white_balance_;
}

Mat CCM_3x3::initial_least_square(Mat src_rgbl, Mat dst_rgbl) {
    Mat res;
    Mat srcc = src_rgbl.reshape(1, 0);
    Mat dstt = dst_rgbl.reshape(1, 0);
    cv::solve(srcc, dstt, res, DECOMP_EIG);
    return res;
}

class loss_rgb_F : public cv::MinProblemSolver::Function{//, public CCM_3x3 {
public:
    CCM_3x3 ccm_loss;
    loss_rgb_F(CCM_3x3 ccm3x3) {
        ccm_loss = ccm3x3;
    }
    int getDims() const { return 2; }
    double calc(const double* x) const {
        Mat ccm(3, 3, CV_64F, &x);
        Mat lab_est = ccm_loss.cs->rgbl2rgb(ccm_loss.src_rgbl_masked * ccm);
        Mat dist = distance_s(lab_est, ccm_loss.dst_rgb_masked, ccm_loss.distance);
        Mat dist_;
        pow(dist, 2.0, dist_);
        if (ccm_loss.weights.data) {
            dist_ = ccm_loss.weights_masked_norm * dist_;
        }
        Scalar ss = sum(dist_);
        return ss[0];
    }
};

void CCM_3x3::calculate_rgb(void) {
    cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
    cv::Ptr<MinProblemSolver::Function> ptr_F(new loss_rgb_F(*this));
    solver->setFunction(ptr_F);
    double res = solver->minimize(this->ccm0);
    double error = pow((res / this->masked_len), 0.5);
    cout << "error:" << error << endl;
}

double CCM_3x3::loss_rgbl(Mat ccm) {
    Mat dist_;
    cv::pow((this->dst_rgbl_masked - this->src_rgbl_masked * this->ccm), 2, dist_);
    if (this->weights.data) {
        dist_ = this->weights_masked_norm * dist_;
    }
    Scalar ss = sum(dist_);
    return ss[0];
}

void CCM_3x3::calculate_rgbl(void) {
    if (this->weights.data) {
        this->ccm = initial_least_square(this->src_rgbl_masked, this->dst_rgbl_masked);
    }
    else {
        Mat w_, w;
        pow(this->weights_masked_norm, 0.5, w_);
        w = Mat::diag(w_);
        this->ccm = initial_least_square(this->src_rgbl_masked * w, this->dst_rgbl_masked * w);
    }
    cout << "error" << (loss_rgbl(this->ccm) / this->masked_len) << endl;
    double error = pow((loss_rgbl(this->ccm) / this->masked_len), 0.5);
   // double error =  ((loss_rgbl(this->ccm) / this->masked_len), 0.5);
    cout <<"error"<< error<< endl;
}

extern int loss_F_count = 0;
class loss_F : public cv::MinProblemSolver::Function{//,  public CCM_3x3 {
public:
    CCM_3x3 ccm_loss;
    loss_F(CCM_3x3 ccm3x3) {
        ccm_loss=ccm3x3;
    }
    int getDims() const { return 9; }
    double calc(const double* x) const {
        loss_F_count++;
        for (int i = 0; i < 9; i++) {
             cout <<" " <<x[i];
        }
        cout << endl;
        Mat ccm(3, 3, CV_64F);
        for (int i = 0; i < ccm.rows; i++) {
            for (int j = 0; j < ccm.cols; j++) {
                ccm.at<double>(i, j) = x[ccm.rows * i + j];
            }
        }  
        IO io_;
        Mat xyz = ccm_loss.src_rgbl_masked;
        Mat res_loss=xyz;
        for (int i = 0; i < xyz.rows; i++) {
            for (int j = 0; j < xyz.cols; j++) {
                for (int m = 0; m < 3; m++) {//¾ØÕó³Ë·¨todoº¯Êý

                    double res1 = xyz.at<Vec3d>(i, j)[0] * ccm.at<double>(m, 0);
                    double res2 = xyz.at<Vec3d>(i, j)[1] * ccm.at<double>(m, 1);
                    double res3 = xyz.at<Vec3d>(i, j)[2] * ccm.at<double>(m, 2);
                    res_loss.at<Vec3d>(i, j)[m] = res1 + res2 + res3;
                }
            }
        }
        Mat lab_est = ccm_loss.cs->rgbl2lab(res_loss, io_);
       // cout << "lab_est"<<lab_est << endl;
        Mat dist = distance_s(lab_est, ccm_loss.dst_rgb_masked, ccm_loss.distance);
        Mat dist_;
        pow(dist, 2, dist_);
        if (ccm_loss.weights.data) {
            dist_ = ccm_loss.weights_masked_norm * dist_;
        }
        //Scalar ss = sum(dist_);
        Scalar ss = sum(dist_);
        //cout <<"ss[0] " <<ss[0] << endl;

        return ss[0];
    }
};


void CCM_3x3::calculate(void) {
    Mat step = (Mat_<double>(9, 1) << 0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1,0.1);  
    cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();

    cv::Ptr<MinProblemSolver::Function> ptr_F(new loss_F(*this)); 
    solver->setInitStep(step);
    solver->setFunction(ptr_F);
    //this->ccm0 = (Mat_<double>(9, 1) << 2.04,-0.206, 0.13,-1.515, 1.38, -0.4498, 0.2728,-0.164017,1.29);
    
    //this->ccm0 =( Scalar_<double>(9,1)<< 2.297, 0.119, 0.24612, -0.9426, 0.7241, -0.4196, 1.17243, 0.19689, 3.941103 );
    //cout <<"this->src_rgbl_masked calc  "<< this->src_rgbl_masked << endl;
   // extern int loss_F_count = 0;
    cout << "this->ccm0" << this->ccm0 << endl;
    double res = solver->minimize(this->ccm0);
    cout << "this->ccm0"<< this->ccm0 << endl;
    cout << "loss_F_count  " << loss_F_count<<endl;
    //cout<<"int loss_F_count"<< loss_F_count <<endl;
    cout<<"res"<<res<<endl;
    cout << "error" << res / this->masked_len << endl;
    double error = pow((res / this->masked_len), 0.5);
    cout << "error:" << error << endl;
}


void CCM_3x3::value(int number) {
    RNG rng;
    Mat_<double>rand(number, 3);
    rng.fill(rand, RNG::UNIFORM, 0., 1.);
    Mat mask_ = saturate(infer(rand, false), 0, 1);
    Scalar ss = sum(mask);
    double sat = ss[0] / number;
    cout << "sat:" << sat << endl;
    Mat rgbl = this->cs->rgb2rgbl(rand);
    mask_ = saturate(rgbl * this->ccm.inv(), 0, 1);
    Scalar sss = sum(mask_);
    double dist_ = sss[0] / number;
    cout << "dist:" << dist_ << endl;
}


Mat CCM_3x3::infer(Mat img, bool L) {
    if (!this->ccm.data)
    {
        throw "No CCM values!";
    }
    Mat img_lin = this->linear->linearize(img);
    Mat img_ccm = img_lin * this->ccm;
    if (L == true) {
        return img_ccm;
    }
    return this->cs->rgbl2rgb(img_ccm);
}


Mat CCM_3x3::infer_image(string imgfile, bool L , int inp_size , int out_size ) {
    Mat img = imread(imgfile);
    Mat img_;
    cvtColor(img, img_, COLOR_BGR2RGB);
    img_ = img_ / inp_size;
    Mat out = infer(img_, L);
    Mat out_ = out * out_size;
    out_.convertTo(out_, CV_8UC1, 100, 0.5);
    Mat img_out = min(max(out_, 0), out_size);
    Mat out_img;
    cvtColor(img_out, out_img, COLOR_RGB2BGR);
    return out_img;
}


void CCM_4x3::prepare(void) {
    this->src_rgbl_masked = add_column(this->src_rgbl_masked);
}


Mat CCM_4x3::add_column(Mat arr) {
    Mat arr1 = Mat::ones(arr.rows, 1, CV_8U);
    Mat arr_out;
    vconcat(arr, arr1, arr_out);
    return arr_out;
}


Mat CCM_4x3::initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {
    Scalar rs = sum(src_rgbl.rowRange(0, 1));
    Scalar gs = sum(src_rgbl.rowRange(1, 2));
    Scalar bs = sum(src_rgbl.rowRange(2, 3));
    Scalar rd = sum(src_rgbl.rowRange(0, 1));
    Scalar gd = sum(src_rgbl.rowRange(1, 2));
    Scalar bd = sum(src_rgbl.rowRange(2, 3));
    Mat initial_white_balance_ = (Mat_<double>(3, 3) << rd[0] / rs[0], 0, 0, 0, gd[0] / gs[0], 0, 0, 0, bd[0] / bs[0]);//
    return initial_white_balance_;
}


Mat CCM_4x3::infer(Mat img, bool L ) {
    if (!this->ccm.data) {
        throw "No CCM values!";
    }
    Mat img_lin = this->linear->linearize(img);
    Mat img_ccm = add_column(img_lin) * this->ccm;
    if (L) {
        return img_ccm;
    }
    return this->cs->rgbl2rgb(img_ccm);
}


void CCM_4x3::value(int number) {
    RNG rng;
    Mat_<double>rand(number, 3);
    rng.fill(rand, RNG::UNIFORM, 0, 1);
    Mat mask_ = saturate(infer(rand, false), 0, 1);
    Scalar ss = sum(mask_);
    double sat = ss[0] / number;
    cout << "sat:" << sat << endl;
    Mat rgbl = this->cs->rgb2rgbl(rand);
    Mat up = this->ccm.rowRange(1, 3);
    Mat down = this->ccm.rowRange(3, this->ccm.rows);
    mask_ = saturate((rgbl - Mat::ones(number, 1, CV_8U) * down) * up.inv(), 0, 1);
    Scalar sss = sum(mask_);
    double dist_ = sss[0] / number;
    cout << "dist:" << dist_ << endl;
}
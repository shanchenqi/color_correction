#include "ccm.h"
#include "linearize.h"

extern int loss_F_count = 0;

CCM_3x3::CCM_3x3(Mat src_, Mat dst, string dst_colorspace, string dst_illuminant, int dst_observer, Mat dst_whites, string colorchecker, vector<double> saturated_threshold, string colorspace, string linear_, float gamma, int deg,  string dist_illuminant, int dist_observer, Mat weights_list, double weights_coeff, bool weights_color,  string ccm_shape)
{
    this->shape = (ccm_shape == "3x3") ? 3 : 4;
    cout <<"this->shape   "<< this->shape << endl;
    this->src = src_;
    IO dist_io = IO(dist_illuminant, dist_observer);
    this->cs = get_colorspace(colorspace);
    cs->set_default(dist_io);
    ColorChecker cc_;
    if (!dst.empty()) {
        cc_ = ColorChecker(dst, dst_colorspace, IO(dst_illuminant, dst_observer), dst_whites);
    }
    else if (colorchecker == "Macbeth_D65_2") {    
        cc_ = ColorChecker(ColorChecker2005_LAB_D65_2, "LAB", IO("D65", 2), Arange_18_24);
    }
    else if (colorchecker == "Macbeth_D50_2") {
        cc_ = ColorChecker(ColorChecker2005_LAB_D50_2, "LAB", IO("D50", 2), Arange_18_24);
    }
    this->cc = ColorCheckerMetric(cc_, colorspace, dist_io);
    this->linear = get_linear(linear_, gamma, deg, this->src, this->cc, saturated_threshold);
    
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

    Mat weight_mask = Mat::ones( src.rows, 1, CV_64FC1);
    if (weights_color) {
        weight_mask = this->cc.color_mask;
    }
    Mat saturate_mask = saturate(src, saturated_threshold[0], saturated_threshold[1]); 
    this->mask= (weight_mask) & (saturate_mask);
    this->mask.convertTo(this->mask, CV_64F);
    this->src_rgbl = this->linear->linearize(this->src);
    this->src_rgb_masked = mask_copyto(this->src, mask);
    this->src_rgbl_masked = mask_copyto(this->src_rgbl, mask);
    cout << "*************this->src_rgbl_masked***********" << this->src_rgbl_masked << endl;
    this->dst_rgb_masked = mask_copyto(this->cc.rgb, mask);
    this->dst_rgbl_masked = mask_copyto(this->cc.rgbl, mask);
    this->dst_lab_masked = mask_copyto(this->cc.lab, mask);
    if (this->weights.data) {
        this->weights.copyTo(this->weights_masked, this->mask);
        this->weights_masked_norm = this->weights_masked / mean(this->weights_masked);
    }
    this->masked_len = this->src_rgb_masked.rows;
   
    //prepare();
    //if (initial_method == "white_balance") {
    //    this->ccm0 = initial_white_balance(this->src_rgbl_masked, this->dst_rgbl_masked);
    //}
    //else if (initial_method == "least_square") {
    //    //cout << "this->src_rgbl_masked" <<this->src_rgbl_masked << "this->dst_rgbl_masked" << this->dst_rgbl_masked << endl;
    //    this->ccm0 = initial_least_square(this->src_rgbl_masked, this->dst_rgbl_masked);
    //} 
    //this->distance = distance_;
    //if (this->distance == "rgb") {
    //    calculate_rgb();
    //}
    //else if (this->distance == "rgbl") {
    //    calculate_rgbl();
    //}
    //else {
    //    this->calculate();
    //}
}
void CCM_3x3::calc(string initial_method,string distance_) {
    prepare();
  if (initial_method == "white_balance") {
      this->ccm0 = this->initial_white_balance(this->src_rgbl_masked, this->dst_rgbl_masked);
  }
  else if (initial_method == "least_square") {
      //cout << "this->src_rgbl_masked" <<this->src_rgbl_masked << "this->dst_rgbl_masked" << this->dst_rgbl_masked << endl;
      this->ccm0 = initial_least_square(this->src_rgbl_masked, this->dst_rgbl_masked);
  } 
  this->distance = distance_;
  if (this->distance == "rgb") {
      calculate_rgb();
  }
  else if (this->distance == "rgbl") {
      calculate_rgbl();
  }
  else {
      this->calculate();
  }
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
    cv::solve(srcc, dstt, res, DECOMP_NORMAL);
    cout << "res" << res << endl;
    return res;
}

class loss_rgb_F : public cv::MinProblemSolver::Function{
public:
    //if()
    CCM_3x3 ccm_loss;
    int loss_shape;
    
    loss_rgb_F(CCM_3x3 ccm3x3,int shape) {
        ccm_loss = ccm3x3;
        loss_shape = shape;
    }
    int getDims() const { return 3*loss_shape; }
    double calc(const double* x) const {
        Mat ccm(loss_shape, 3, CV_64F);
        for (int i = 0; i < ccm.rows; i++) {
            for (int j = 0; j < ccm.cols; j++) {
                ccm.at<double>(i, j) = x[ccm.rows * i + j];
            }
        }
        loss_F_count++;
        Mat res_loss(ccm_loss.src_rgbl_masked.size(), ccm_loss.src_rgbl_masked.type());
        res_loss = mult(ccm_loss.src_rgbl_masked, ccm);
        Mat lab_est = ccm_loss.cs->rgbl2rgb(res_loss);
        cout << ccm_loss.cs->rgbl2rgb(res_loss) << endl;
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
    cv::Ptr<MinProblemSolver::Function> ptr_F(new loss_rgb_F(*this,this->shape));
    Mat reshapeccm = this->ccm0.reshape(0, 1);  
    solver->setFunction(ptr_F);
    RNG step_rng;
    Mat_<double>step(reshapeccm.size());
    solver->setInitStep(step);
    step_rng.fill(step, RNG::UNIFORM, 0., 1.);
    cout << "reshapeccm" << reshapeccm << endl;
    double res = solver->minimize(reshapeccm);
    cout << "reshapeccm" << reshapeccm << endl;
    this->ccm = reshapeccm.reshape(0, this->shape);
    cout << "loss_F_count  " << loss_F_count << endl;
    //cout<<"int loss_F_count"<< loss_F_count <<endl;
    cout << "res" << res << endl;
    cout << "error" << res / this->masked_len << endl;
    double error = pow((res / this->masked_len), 0.5);
    cout << "error:" << error << endl;
}
//void CCM_3x3::prepare(void) {};
double CCM_3x3::loss_rgbl(Mat ccm) {
    Mat dist_;
    Mat dist_res = mult(this->src_rgbl_masked, this->ccm);
    cv::pow((this->dst_rgbl_masked - dist_res), 2, dist_);
    Mat res_dist = dist_.reshape(1, 0);
    if (this->weights.data) {
        dist_ = this->weights_masked_norm * dist_;
    }
    Scalar ss = sum(res_dist);
    return ss[0];
}

void CCM_3x3::calculate_rgbl(void) {
    if (!this->weights.data) {
        this->ccm = initial_least_square(this->src_rgbl_masked, this->dst_rgbl_masked);
    }
    else {
        Mat w_, w;
        //cout <<"this->weights_masked_norm"<< this->weights_masked_norm <<endl;
        pow(this->weights_masked_norm, 0.5, w_);
        w = Mat::diag(w_);
        this->ccm = initial_least_square(mult(this->src_rgbl_masked, w), mult(this->dst_rgbl_masked, w));
    }
    double error = pow((loss_rgbl(this->ccm) / this->masked_len), 0.5);
    cout << "this->ccm" << this->ccm << endl;
    cout <<"error"<< error<< endl;
}

//extern int loss_F_count = 0;
class loss_F : public cv::MinProblemSolver::Function{//,  public CCM_3x3 {
public:
   // CCM_3x3 ccm_loss;
    CCM_3x3 ccm_loss;
    int loss_shape;
    //loss_F(CCM_3x3 ccm3x3, int shape) {
    loss_F(CCM_3x3 ccm3x3, int shape) {
        ccm_loss=ccm3x3;
        loss_shape = shape;
    }
    int getDims() const { return loss_shape*3; }
    double calc(const double* x) const {
        loss_F_count++;
        Mat ccm(loss_shape, 3, CV_64F);
        for (int i = 0; i < ccm.rows; i++) {
            for (int j = 0; j < ccm.cols; j++) {
                ccm.at<double>(i, j) = x[ccm.rows * i + j];
            }
        }  
        IO io_;
        //Mat res_loss(ccm_loss.src_rgbl_masked.size(), ccm_loss.src_rgbl_masked.type());
        Mat res_loss(ccm_loss.src_rgbl_masked.size(),CV_64FC3);
      //  cout << " ****************ccm_loss.src_rgbl_masked********" << ccm_loss.src_rgbl_masked << endl;
        res_loss = mult4D(ccm_loss.src_rgbl_masked, ccm);
        Mat lab_est = ccm_loss.cs->rgbl2lab(res_loss, io_);
        Mat dist = distance_s(lab_est, ccm_loss.dst_lab_masked, ccm_loss.distance);
        Mat dist_;   
        pow(dist, 2, dist_);
        if (ccm_loss.weights.data) {
            dist_ = ccm_loss.weights_masked_norm * dist_;
        }
        Scalar ss = sum(dist_);
        return ss[0];
    }
};


void CCM_3x3::calculate(void) { 
    cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
    cv::Ptr<MinProblemSolver::Function> ptr_F(new loss_F(*this,this->shape));   
    solver->setFunction(ptr_F);
    Mat reshapeccm = this->ccm0.reshape(0, 1);
    RNG step_rng;
    Mat_<double>step(reshapeccm.size());
    step_rng.fill(step, RNG::UNIFORM, 0., 1.);
    solver->setInitStep(step);
    cout << "reshapeccm" << reshapeccm << endl;
    double res = solver->minimize(reshapeccm);
    cout << "reshapeccm"<< reshapeccm << endl;
    this->ccm = reshapeccm.reshape(0, 3);
    cout << "loss_F_count  " << loss_F_count<<endl;
    //cout<<"int loss_F_count"<< loss_F_count <<endl;
    cout<<"res"<<res<<endl;
    double error = pow((res / this->masked_len), 0.5);
    cout << "error:" << error << endl;
}


void CCM_3x3::value(int number) {
    RNG rng;
    Mat_<Vec3d>rand(number, 1);
    rng.fill(rand, RNG::UNIFORM, 0., 1.);
    Mat mask_ = saturate(infer(rand, false), 0, 1);
    Scalar ss = sum(mask_);
    double sat = ss[0] / number;
    cout <<"sat"<< sat << endl;
    Mat rgbl = this->cs->rgb2rgbl(rand);
    mask_ = saturate(mult(rgbl , this->ccm.inv()), 0, 1);
    Mat mask_pre = mask_.reshape(1, 0);
    Scalar sss = sum(mask_pre);
    double dist_ = sss[0] / number;
    cout << "dist_" <<dist_<< endl;
}

Mat CCM_3x3::infer(Mat img, bool L) {
    if (!this->ccm.data)
    {
        throw "No CCM values!";
    }
    L = false;
    Mat img_lin = this->linear->linearize(img);
    Mat img_ccm(img_lin.size(), img_lin.type());
    img_ccm = mult(img_lin, this->ccm);
    if (L == true) {
        return img_ccm;
    }
    return this->cs->rgbl2rgb(img_ccm);
}

Mat CCM_3x3::infer_image(string imgfile, bool L , int inp_size , int out_size ) {
    Mat img = imread(imgfile);
    Mat img_;
    cvtColor(img, img_, COLOR_BGR2RGB);
    img_.convertTo(img_, CV_64F);
    img_ = img_ / inp_size;
    Mat out = infer(img_, L);
    Mat out_ = out * out_size;
    out_.convertTo(out_, CV_8UC3);
    Mat img_out = min(max(out_, 0), out_size);
    Mat out_img;
    cvtColor(img_out, out_img, COLOR_RGB2BGR);
    return out_img;
}

void CCM_4x3::prepare(void) {
    cout << "&&&&&&&&&&&&&&&&prepare()&&&&&&&&&&&&&&&&&&&&&&" << endl;
    cout << "this->src_rgbl_masked" << this->src_rgbl_masked << endl;
    this->src_rgbl_masked = add_column(this->src_rgbl_masked);
}

Mat CCM_4x3::add_column(Mat arr) {
    
    Mat arr1 = Mat::ones(arr.size(), CV_64F);
    cout << arr.size() << endl;
    cout << arr1.size() << endl;
    Mat arr_out(arr.size(),CV_64FC4);
    Mat arr_channels[3];
   // Mat arr_out_channels[4];
    split(arr, arr_channels);
    //split(arr_out, arr_out_channels);
    vector<Mat> arrout_channel;
    arrout_channel.push_back(arr_channels[0]);
    arrout_channel.push_back(arr_channels[1]);
    arrout_channel.push_back(arr_channels[2]);
    arrout_channel.push_back(arr1);
    merge(arrout_channel, arr_out);
    //hconcat(arr, arr1, arr_out);
    cout << "arr_out" << arr_out.size() << endl;
    return arr_out;
}

Mat CCM_4x3::initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {
    cout << "(((((((((((while_balance))))))))))))" << endl;
    Mat schannels[3];
    Mat dchannels[3];
    split(src_rgbl, schannels);
    split(dst_rgbl, dchannels);
    Scalar rs = sum(schannels[0]);
    Scalar gs = sum(schannels[1]);
    Scalar bs = sum(schannels[2]);
    Scalar rd = sum(dchannels[0]);
    Scalar gd = sum(dchannels[1]);
    Scalar bd = sum(dchannels[2]);
    Mat initial_white_balance_ = (Mat_<double>(4, 3) << rd[0] / rs[0], 0, 0, 0, gd[0] / gs[0], 0, 0, 0, bd[0] / bs[0], 0, 0, 0);
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
#include "colorchecker.h"

//vector<double> white_m(10,1);

ColorChecker::ColorChecker(Mat color, string colorspace, IO io_, Mat whites) {
	//if (colorspace == "lab")
	if (colorspace == "LAB")
	{
		this->lab = color;	
		this->io = io_;
		
	}
	else
	{
		this->rgb = color;
		this->cs = get_colorspace(colorspace);
		//this->cs =new sRGB£»
	}

	//vector<bool> white_m(color.rows, false);
	//vector<double> white_m(color.rows, 0);
	vector<double> white_m(color.rows, 1);


	if (!whites.empty())
	{
		for (int i = 0; i < whites.cols; i++)
		{
			
			white_m[whites.at<double>(0, i)] = 0;
			
		}
		this->white_mask =Mat(white_m, true);
	}
	//Mat white_mask;

//	white_mask = Mat(white_m,true);
	
	//color_mask = ~Mat(white_m);
	//color_mask = white_mask;
	color_mask = Mat(white_m, true);
}

ColorCheckerMetric::ColorCheckerMetric(ColorChecker colorchecker, string colorspace, IO io_)
{
	this->cc = colorchecker;
	this->cs = get_colorspace(colorspace);
	this->io = io_;
	
	if (!this->cc.lab.empty())
	{
		cout<<"******************"<<endl;
		this->lab = lab2lab(this->cc.lab, cc.io, io_);
		this->xyz = lab2xyz(lab, io_);
		//cout << "xyz" << this->xyz << endl;
		this->rgbl = this->cs->xyz2rgbl(this->xyz, io_);
		//cout << "this->rgbl" << this->rgbl;
		this->rgb = cs->rgbl2rgb(this->rgbl);
		//cout << "this->rgb" << this->rgb << endl;;
		//cout << "this->rgbl" << this->rgbl << endl;
	}
	else
	{
		cout << "############" << endl;
		this->rgb = cs->xyz2rgb(cc.cs->rgb2xyz(cc.rgb, IO("D65", 2)), IO("D65", 2));
		this->rgbl = cs->rgb2rgbl(rgb);
		this->xyz = cs->rgbl2xyz(rgbl, io);
		this->lab = xyz2lab(xyz, io);
	}
	this->grayl = xyz2grayl(xyz);
	this->white_mask = cc.white_mask;
	this->color_mask = cc.color_mask;
}


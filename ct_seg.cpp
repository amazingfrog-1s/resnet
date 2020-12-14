#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

double cal_mean_stddev(Mat src) {
	Mat gray, mat_mean, mat_stddev;
	//cvtColor(src, gray, COLOR_RGB2GRAY); // 转换为灰度图
	meanStdDev(src, mat_mean, mat_stddev);
	return mat_mean.at<double>(0, 0);
}

Mat ct_bin_mask(Mat img) {
	vector<Mat> channels;
	split(img, channels);
	double r_mean = cal_mean_stddev(channels[2]);
	cout << r_mean << endl;
	Mat bin_mask = Mat::zeros(Size(img.cols, img.rows), CV_8UC3);
	for (int row = 0; row < img.rows; row++){
		for (int col = 0; col < img.cols; col++){
			if (img.at<Vec3b>(row, col)[0] + img.at<Vec3b>(row, col)[1] + img.at<Vec3b>(row, col)[2] > 500)
					bin_mask.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
		}
	}
	return bin_mask;
}


Mat fusion_img(Mat img, Mat bin_mask) {
	Mat fusion = Mat::zeros(Size(img.cols, img.rows), CV_8UC3);
	img.copyTo(fusion, bin_mask);
	return fusion;
}

Mat circle_crop(Mat img, int radius) {
    Mat circle_mask = Mat::zeros(Size(img.cols, img.rows), CV_8UC3);
    Point center(img.cols / 2, img.rows / 2);
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            int temp = ((col - center.x) * (col - center.x) + (row - center.y) * (row - center.y));
            if (temp < (radius * radius)) {
                circle_mask.at<Vec3b>(row, col)[0] = 255;
                circle_mask.at<Vec3b>(row, col)[1] = 255;
                circle_mask.at<Vec3b>(row, col)[2] = 255;
            }
        }
    }
    return fusion_img(img, circle_mask);
}


Mat ct_filter(int mode, Mat src, int ksize) {
	Mat out;
	Mat gse = getStructuringElement(MORPH_RECT, Size(ksize, ksize));
    if (mode == 1) {
        dilate(src, out, gse);
    }
    if (mode == 2) {
        erode(src, out, gse);
    }
	return out;
}

#define devView(i) imshow(#i,i)
void demoView(vector<Mat> mats) {
    int w = mats[0].cols, h = mats[0].rows;
	Mat canvas = Mat::zeros(Size(4 * w, 2 * h), CV_8UC3);
	mats[0].copyTo(canvas(Rect(0, 0, w, h)));
    mats[1].copyTo(canvas(Rect(w, 0, w, h)));
    mats[2].copyTo(canvas(Rect(2*w, 0, w, h)));
    mats[3].copyTo(canvas(Rect(3*w, 0, w, h)));
    mats[4].copyTo(canvas(Rect(0, h, w, h)));
    mats[5].copyTo(canvas(Rect(w, h, w, h)));
    mats[6].copyTo(canvas(Rect(2*w, h, w, h)));
    mats[7].copyTo(canvas(Rect(3*w, h, w, h)));
	imshow("", canvas);
}


Mat LCC(Mat src, int thres1, int area) {
    Mat gray, bin;
     cvtColor(src, gray, COLOR_BGR2GRAY);
     threshold(gray, bin, thres1, 255, THRESH_BINARY);
     //生成随机颜色，用于区分不同连通域
     RNG rng(10086);
     Mat out, stats, centroids;
     int number = connectedComponentsWithStats(bin, out, stats, centroids, 8, CV_16U);
     Mat result = Mat::zeros(gray.size(), src.type());
     int w = result.cols;
     int h = result.rows;
     for (int row = 0; row < h; row++) {
         for (int col = 0; col < w; col++) {
             int label = out.at<uint16_t>(row, col);
             //背景的黑色不改变
             if (label == 0) {
                 continue;
             }
             if (stats.at<int>(label, CC_STAT_AREA) < area) {
                 continue;
             }
             result.at<Vec3b>(row, col)[0] = 255;
             result.at<Vec3b>(row, col)[1] = 255;
             result.at<Vec3b>(row, col)[2] = 255;
         }
     }
     return result;
}

int erode_filter1 = 15;
int erode_filter2 = 5;
int dilate_filter1 = 5;
int lcc_1 = 50;
int lcc_2 = 150;
int area = 100 * 100;
int re_ksize = 300;

vector<Mat> ct_seg(Mat src) {
    vector<Mat> results;
    Mat crop, filter1, lcc1, lcc2, sub_mask, filter2, filter3, dst;
    resize(src, src, Size(re_ksize, re_ksize));
    crop = circle_crop(src, 0.5 * re_ksize);
    filter1 = ct_filter(2, crop, erode_filter1);
    lcc1 = LCC(filter1, lcc_1, area);
    lcc2 = LCC(filter1, lcc_2, area);
    sub_mask = lcc1 - lcc2;
    sub_mask = LCC(sub_mask, 150, 0.5 * area);// using LCC to delete small noise area
    filter2 = ct_filter(2, sub_mask, erode_filter2);
    filter3 = ct_filter(1, filter2, dilate_filter1);
    dst = Mat::zeros(Size(sub_mask.cols, sub_mask.rows), CV_8UC3);
    src.copyTo(dst, filter3);
    results.push_back(src); 
    results.push_back(filter1);
    results.push_back(lcc1);
    results.push_back(lcc2);
    results.push_back(sub_mask);
    results.push_back(filter2);
    results.push_back(filter3);
    results.push_back(dst);


    return results;
}

void seg_write(cv::String pattern, cv::String save_path)
{
    vector<cv::String> fn;
    glob(pattern, fn, false);
    vector<Mat> images;
    size_t count = fn.size(); //number of png files in images folder
    for (size_t i = 0; i < count; i++){
        images.emplace_back(cv::imread(fn[i]));

        imwrite(save_path + to_string(i) + ".jpg", ct_seg(cv::imread(fn[i])));
    }
}
int main(){
    // generate datasets
    // seg_write("./negetive/*.*","./pre_negetive/");
    // Mat src = imread("covid7.png");
    
    
    // display the processing
    Mat src = imread("covid2.png");
    demoView(ct_seg(src));
    

    waitKey(0);
	return 0;
}


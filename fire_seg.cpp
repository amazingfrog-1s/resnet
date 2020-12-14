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

Mat fire_bin_mask(Mat img) {
	vector<Mat> channels;
	split(img, channels);
	double r_mean = cal_mean_stddev(channels[2]);
	cout << r_mean << endl;
	Mat bin_mask = Mat::zeros(Size(img.cols, img.rows), CV_8UC3);
	for (int row = 0; row < img.rows; row++)
	{
		for (int col = 0; col < img.cols; col++)
		{
			/* 注意 Mat::at 函数是个模板函数, 需要指明参数类型, 因为这张图是具有红蓝绿三通道的图,
			   所以它的参数类型可以传递一个 Vec3b, 这是一个存放 3 个 uchar 数据的 Vec(向量). 这里
			   提供了索引重载, [2]表示的是返回第三个通道, 在这里是 Red 通道, 第一个通道(Blue)用[0]返回 */

			if (img.at<Vec3b>(row, col)[0] + img.at<Vec3b>(row, col)[1] + img.at<Vec3b>(row, col)[2] > 500)
				if(img.at<Vec3b>(row, col)[2] > 200)
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


int main() {

	Mat fire1 = imread("fire1.jpg");
	Mat fire2 = imread("fire2.png");
	Mat fire3 = imread("fire3.jpg");
	Mat fire4 = imread("fire4.png");
	Mat fire5 = imread("fire5.png");

	Mat image = fire5;


	imshow("imshow", image);
	imshow("binary_mask", fire_bin_mask(image));
	imshow("fusion_img", fusion_img(image, fire_bin_mask(image)));
	waitKey(0);


	return 0;
}


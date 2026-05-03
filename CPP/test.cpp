#include <opencv2/opencv.hpp>
#include <iostream>
//#include "quickdemo.h"

using namespace cv;
using namespace std;


// 改进版椒盐噪声函数：支持单通道/多通道图像
Mat addSaltPepperNoise(Mat& img, float noise_density) {
    Mat noisy_img = img.clone();
    // 初始化随机种子（确保每次噪声位置不同）
    srand((unsigned int)time(NULL));

    int total_pixels = img.rows * img.cols;
    int noise_count = total_pixels * noise_density;
    int channels = img.channels(); // 获取图像实际通道数

    // 生成盐噪声（白点）
    for (int i = 0; i < noise_count / 2; i++) {
        int x = rand() % img.cols;
        int y = rand() % img.rows;
        
        // 按通道数分支访问像素
        if (channels == 1) { // 单通道（灰度图）
            noisy_img.at<uchar>(y, x) = 255;
        } else if (channels == 3) { // 3通道（彩色图）
            noisy_img.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
        }
    }

    // 生成椒噪声（黑点）
    for (int i = 0; i < noise_count / 2; i++) {
        int x = rand() % img.cols;
        int y = rand() % img.rows;
        
        if (channels == 1) {
            noisy_img.at<uchar>(y, x) = 0;
        } else if (channels == 3) {
            noisy_img.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
        }
    }
    return noisy_img;
}

Mat addGaussianNoise(Mat& img, double mean, double std) {
    Mat noisy_img = img.clone();
    int channels = img.channels(); // 获取图像实际通道数

    // 根据通道数生成对应维度的噪声
    int noise_type = (channels == 1) ? CV_32FC1 : CV_32FC3;
    Mat noise(img.size(), noise_type);
    
    // 生成对应通道数的高斯噪声
    if (channels == 1) {
        randn(noise, Scalar(mean), Scalar(std)); // 单通道噪声
    } else {
        randn(noise, Scalar(mean, mean, mean), Scalar(std, std, std)); // 3通道噪声
    }

    // 转换图像类型（保持通道数一致）
    noisy_img.convertTo(noisy_img, noise_type);
    noisy_img += noise; // 同通道数相加，无匹配问题
    // 转回8位图像（保持通道数一致）
    noisy_img.convertTo(noisy_img, (channels == 1) ? CV_8UC1 : CV_8UC3);

    return noisy_img;
}

int main(int argc, char** argv) {
    Mat sample = imread("F:/Test/CPP/picture/home.jpg"); 

    if (sample.empty()) {
        printf("could not load image....\n");
        return -1;
    }
    // 1. 显示原始彩色图
    namedWindow("1. Original Color", WINDOW_AUTOSIZE); 
    imshow("1. Original Color", sample);

    // 2. 转灰度图并显示
    Mat gray;
    cvtColor(sample, gray, COLOR_BGR2GRAY);
    namedWindow("2. Original Gray", WINDOW_AUTOSIZE); 
    imshow("2. Original Gray", gray);

    // 3. 生成噪声并显示
    Mat gray_gauss_noise = addGaussianNoise(gray, 0, 15);
    namedWindow("Noise", WINDOW_AUTOSIZE); 
    imshow("Noise", gray_gauss_noise);

    Mat gauss_blur_3, gauss_blur_5;
    GaussianBlur(gray_gauss_noise, gauss_blur_3, Size(3,3), 1.5);
    namedWindow("gauss_blur_3", WINDOW_AUTOSIZE); 
    imshow("gauss_blur_3", gauss_blur_3);
   

    Mat gauss_kernel_1d = getGaussianKernel(3, 1.5); // 一维核
    Mat gauss_kernel_2d = gauss_kernel_1d * gauss_kernel_1d.t(); // 二维核
    Mat manual_gauss_blur;
    filter2D(gray_gauss_noise, manual_gauss_blur, -1, gauss_kernel_2d);
    namedWindow("manual_gauss_blur", WINDOW_AUTOSIZE); 
    imshow("manual_gauss_blur", manual_gauss_blur);

    // 循环等待，ESC退出
    while (true) {
        int c = waitKey(1);
        if (c == 27) {
            break;
        }
    }
    destroyAllWindows();
    return 0;
}

//& "C:/Program Files/mingw64/bin/g++.exe" -g test.cpp quickdemo.cpp -o test.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual

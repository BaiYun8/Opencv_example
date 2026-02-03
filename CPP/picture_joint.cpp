//图像拼接
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "quickdemo.h"

using namespace cv;
using namespace std;


void generate_mask(Mat &img, Mat &mask) {
    int w = img.cols;
    int h = img.rows;
    // 双重循环遍历图像的每个像素
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            // 取img的当前像素（Vec3b：3通道uchar类型，对应BGR）
            Vec3b p = img.at<Vec3b>(row, col);
            int b = p[0]; int g = p[1]; int r = p[2];
            // 核心判断：如果像素是纯黑色（B=G=R=0）→ 标记为掩码的255（白色），否则为0（黑色）
            if (b == g && g == r && r == 0) {
                mask.at<uchar>(row, col) = 255;
            }
        }
    }
}

void linspace(Mat& image, float begin, float finish, int w1, Mat &mask) {
    int offsetx = 0; // 记录每行有效区域的起始列（mask=0的第一个列）
    float interval = 0; // 渐变的步长
    float delta = 0; // 渐变的列数（有效区域的宽度）
    // 遍历每一行（逐行生成渐变，保证每行的渐变对齐）
    for (int i = 0; i < image.rows; i++) {
        offsetx = 0; interval = 0; delta = 0; // 每行初始化
        // 遍历每一列
        for (int j = 0; j < image.cols; j++) {
            // 取掩码的当前像素值（0=有效区域，255=无效区域）
            int pv = mask.at<uchar>(i, j);
            // 情况1：首次遇到有效区域（pv=0，offsetx=0）→ 初始化渐变参数
            if (pv == 0 && offsetx == 0) {
                offsetx = j; // 记录有效区域的起始列
                delta = w1 - offsetx; // 渐变的总列数（左图宽度 - 有效区域起始列）
                interval = (finish - begin) / (delta - 1); // 渐变步长：(结束值-开始值)/(列数-1)
                // 计算当前列的权重值
                image.at<float>(i, j) = begin + (j - offsetx)*interval;
            }
            // 情况2：在有效区域内（pv=0），且未超出渐变列数 → 继续计算渐变权重
            else if (pv == 0 && offsetx > 0 && (j - offsetx) < delta) {
                image.at<float>(i, j) = begin + (j - offsetx)*interval;
            }
            // 情况3：无效区域（pv=255）→ 保持初始权重1（无需处理，因为mask1/mask2初始为1）
        }
    }
}


int main()
{
    Mat left = imread("F:/Test/CPP/picture/q11.jpg");
    Mat right = imread("F:/Test/CPP/picture/q22.jpg");
    if(left.empty() || right.empty())
    {
        cout << " read picture failed.." << endl;
        return -1;
    }

    // 提取特征点与描述子
	vector<KeyPoint> keypoints_right, keypoints_left;
	Mat descriptors_right, descriptors_left;
	auto detector = AKAZE::create();
	detector->detectAndCompute(left, Mat(), keypoints_left, descriptors_left);
	detector->detectAndCompute(right, Mat(), keypoints_right, descriptors_right);

    //暴力匹配
    vector<vector<DMatch>> knn_matches;
    auto matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    //auto matcher = BFMatcher::create(NORM_L2);
    matcher->knnMatch(descriptors_left, descriptors_right, knn_matches, 2);

    const float radio_thresh = 0.7f;    // 比例阈值
    vector<DMatch> good_matches;        // 存储筛选后的优质匹配对
    // 遍历所有KNN匹配结果，执行Ratio Test筛选
    for(size_t t=0; t<knn_matches.size(); t++)
    {
        //第1个匹配的距离 < 阈值 * 第2个匹配的距离 → 视为优质匹配
        if(knn_matches[t][0].distance < radio_thresh * knn_matches[t][1].distance)
        {
            good_matches.push_back(knn_matches[t][0]);
        }
    }
    printf("total good match points : %d\n", good_matches.size());
    //匹配结果可视化
    //Mat dst;
    //drawMatches(left, keypoints_left, right, keypoints_right, good_matches, dst);
    //namedWindow("output", WINDOW_FREERATIO);
    //imshow("output", dst);

    vector<Point2f> left_pts;
    vector<Point2f> right_pts;
    for(size_t t=0; t<good_matches.size(); t++)
    {
        left_pts.push_back(keypoints_left[good_matches[t].queryIdx].pt);
        right_pts.push_back(keypoints_right[good_matches[t].trainIdx].pt);
    }
    // 核心：求解右图到左图的单应性矩阵H
    Mat H = findHomography(right_pts, left_pts, RANSAC);//源图->目标图


    //初始化全景图画布 + 左图拷贝（拼接的基础）
    int h = max(left.rows, right.rows);
    int w = left.cols + right.cols;
    Mat panorama_01 = Mat::zeros(Size(w, h), CV_8UC3);
    // 定义左图的感兴趣区域（ROI）：左上角(0,0)，尺寸和左图一致
    Rect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = left.cols;
    roi.height = left.rows;
    left.copyTo(panorama_01(roi));
    imwrite("F:/Test/CPP/picture/panorama_01.png", panorama_01);

    // 右图透视变换
    // 核心：用单应性矩阵H，把右图透视变换到左图的坐标系
    Mat panorama_02;
    warpPerspective(right, panorama_02, H, Size(w, h));
    imwrite("F:/Test/CPP/picture/panorama_02.png", panorama_02);

    // 创建掩码画布：黑色背景，尺寸和全景图一致，单通道灰度图
    Mat mask = Mat::zeros(Size(w, h), CV_8UC1);
    // 核心：根据配准后的右图（panorama_02），生成有效区域掩码
    generate_mask(panorama_02, mask);
    imwrite("F:/Test/CPP/picture/mask.png", mask);

    // 创建两个权重掩码：初始值全为1（float类型，用于加权计算）
    Mat mask1 = Mat::ones(Size(w, h), CV_32FC1); // 左图的权重掩码
    Mat mask2 = Mat::ones(Size(w, h), CV_32FC1); // 右图的权重掩码

    // 为mask1生成渐变：从1→0（左图在重叠区域的权重逐渐降低）
    linspace(mask1, 1, 0, left.cols, mask);
    // 为mask2生成渐变：从0→1（右图在重叠区域的权重逐渐升高）
    linspace(mask2, 0, 1, left.cols, mask);

    namedWindow("mask1", WINDOW_FREERATIO);
    imshow("mask1", mask1);
    namedWindow("mask2", WINDOW_FREERATIO);
    imshow("mask2", mask2);


    // 处理左图权重掩码mask1：单通道→3通道
    Mat m1;
    vector<Mat> mv;
    mv.push_back(mask1);
    mv.push_back(mask1);
    mv.push_back(mask1);
    merge(mv, m1); // merge：把3个单通道矩阵合并为1个3通道矩阵
    // 左图画布转换为float类型（避免加权计算时的精度丢失）
    panorama_01.convertTo(panorama_01, CV_32F);
    // 左图加权：逐像素 × 权重掩码m1
    multiply(panorama_01, m1, panorama_01);

    // 处理右图权重掩码mask2：和mask1同理，单通道→3通道
    mv.clear();
    mv.push_back(mask2);
    mv.push_back(mask2);
    mv.push_back(mask2);
    Mat m2;
    merge(mv, m2);
    // 配准后的右图转换为float类型，逐像素 × 权重掩码m2
    panorama_02.convertTo(panorama_02, CV_32F);
    multiply(panorama_02, m2, panorama_02);

    // 核心：加权后的左图 + 加权后的右图 = 无缝全景图
    Mat panorama;
    add(panorama_01, panorama_02, panorama);
    // 浮点型转换回8位uchar型（图像保存的标准类型）
    panorama.convertTo(panorama, CV_8U);
    // 保存并显示全景图
    namedWindow("out", WINDOW_FREERATIO);
    imshow("out", panorama);

    while (true) {
        int c = waitKey(1);
        if (c == 27) {
            break;
        }
    }
    destroyAllWindows();
    return 0;

}



//& "C:/Program Files/mingw64/bin/g++.exe" -g picture_joint.cpp -o picture_joint.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual

//picture_joint
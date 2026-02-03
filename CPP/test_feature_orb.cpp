#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "quickdemo.h"

using namespace cv;
using namespace std;


int main()
{
    Mat book = imread("F:/Test/CPP/picture/book.jpg");
    Mat book_on_desk = imread("F:/Test/CPP/picture/book_on_desk.jpg");
    if(book.empty() || book_on_desk.empty())
    {
        printf("could not load image....\n");
        return -1;
    }

    // 初始化ORB检测器（核心参数可根据需求调整）
    Ptr<ORB> orb = ORB::create(
        1000,    
        1.2f,  
        8,      
        31,     
        0,      
        2,      
        ORB::HARRIS_SCORE, 
        31,     
        20      
    );

    //检测ORB关键点并计算描述子
    vector<KeyPoint> kp_book, kp_desk;
    orb->detect(book, kp_book);                // 检测book的关键点
    orb->detect(book_on_desk, kp_desk);        // 检测book_on_desk的关键点

    Mat des_book, des_desk;
    orb->compute(book, kp_book, des_book);     // 为book的关键点计算描述子
    orb->compute(book_on_desk, kp_desk, des_desk); // 为book_on_desk的关键点计算描述子

    Mat result_book, result_desk;
    drawKeypoints(book, kp_book, result_book, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(book_on_desk, kp_desk, result_desk, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    waitKey(1); 
    imshow("1", result_book);
    imshow("2", result_desk);

    vector<DMatch> goodMatches;
    QuickDemo qd;
    qd.ORBBFMatcher(book, book_on_desk, kp_book, kp_desk, des_book, des_desk, goodMatches);
    
    Mat matchResult;
    drawMatches(
        book, kp_book,                // 第一张图+关键点
        book_on_desk, kp_desk,        // 第二张图+关键点
        goodMatches,                  // 筛选后的优质匹配
        matchResult,                  // 输出匹配图像
        Scalar(0, 255, 0),            // 匹配线颜色（绿色）
        Scalar(255, 0, 0),            // 未匹配关键点颜色（红色）
        vector<char>(),               // 匹配掩码（默认空）
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS // 不绘制未匹配的单点
    );

    // 
    imshow("ORB暴力特征匹配结果", matchResult);
  
    //// 循环等待，ESC退出
    while (true) {
        int c = waitKey(1);
        if (c == 27) {
            break;
        }
    }
    destroyAllWindows();
    return 0;
}

//& "C:/Program Files/mingw64/bin/g++.exe" -g test_feature_orb.cpp quickdemo.cpp -o test_feature_orb.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual

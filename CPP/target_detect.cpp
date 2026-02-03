#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "quickdemo.h"

using namespace cv;
using namespace std;

//文档对齐，特征匹配

void orb_feature_detect(Mat &image, vector<KeyPoint> &keypoint, Mat &description)
{
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

    orb->detect(image, keypoint);
    orb->compute(image, keypoint, description);
}



int main()
{
    Mat book = imread("F:/Test/CPP/picture/book.jpg");
    Mat book_on_desk = imread("F:/Test/CPP/picture/book_on_desk.jpg");
    if(book.empty() || book_on_desk.empty())
    {
        printf("could not load image....\n");
        return -1;
    }

    vector<KeyPoint> kp_book, kp_desk;
    Mat des_book, des_desk;
    orb_feature_detect(book, kp_book, des_book);
    orb_feature_detect(book_on_desk, kp_desk, des_desk);

    //使用暴力匹配特征点
    vector<DMatch> Matche_BF;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(des_book, des_desk, Matche_BF);

    float good_rate = 0.15f;
    int num_good_matches = Matche_BF.size() * good_rate;
    sort(Matche_BF.begin(), Matche_BF.end());
    Matche_BF.erase(Matche_BF.begin() + num_good_matches, Matche_BF.end());

    Mat outimg1;
    drawMatches(book, kp_book, book_on_desk, kp_desk, Matche_BF, outimg1);
    imshow("Matche_BF", outimg1);

    vector<Point2f> obj_pts;
    vector<Point2f> scene_pts;
    for(size_t t=0; t<Matche_BF.size(); t++)
    {
        obj_pts.push_back(kp_book[Matche_BF[t].queryIdx].pt);
        scene_pts.push_back(kp_desk[Matche_BF[t].trainIdx].pt);
    }
    Mat h = findHomography(obj_pts, scene_pts, RANSAC);
    if (h.empty()) 
    {
        cout << "Error: 单应性矩阵求解失败！" << endl;
        return -1;
    }

    vector<Point2f> srcPts;
    srcPts.push_back(Point2f(0, 0)); // 左上角
    srcPts.push_back(Point2f(book.cols, 0)); // 右上角
    srcPts.push_back(Point2f(book.cols, book.rows)); // 右下角
    srcPts.push_back(Point2f(0, book.rows)); // 左下角
    vector<Point2f> dstPts(4); 
    perspectiveTransform(srcPts, dstPts, h);
    for (int i = 0; i < 4; i++) 
    {
        // (i+1)%4：实现四角点循环连接（3→0，闭合矩形）
        line(book_on_desk, dstPts[i], dstPts[(i + 1) % 4], Scalar(0, 0, 255), 2, 8, 0);
    }
    namedWindow("Object Detection Result", WINDOW_FREERATIO);
    imshow("Object Detection Result", book_on_desk);

    while (true) {
        int c = waitKey(1);
        if (c == 27) {
            break;
        }
    }
    destroyAllWindows();
    return 0;


}

//& "C:/Program Files/mingw64/bin/g++.exe" -g target_detect.cpp -o target_detect.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual

//target_detect
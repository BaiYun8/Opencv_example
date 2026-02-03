#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "quickdemo.h"

using namespace cv;
using namespace std;

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

void match_min(vector<DMatch> matches, vector<DMatch> &good_matches)
{
    double min_dist = 10000, max_dist = 0;
    for(int i=0; i<matches.size(); i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    cout << "min_dist = " << min_dist << endl;
    cout << "max_dist = " << max_dist << endl;

    for(int i=0; i<matches.size(); i++)
    {
        if(matches[i].distance <= max(2*min_dist, 20.0))
        {
            good_matches.push_back(matches[i]);
        }
    }
}

void ransacMatchFilter(const vector<DMatch>& matches,
                       const vector<KeyPoint>& querykeypoint,
                       const vector<KeyPoint>& reainkeypoint,
                       vector<DMatch>& matches_ransac)
{
    vector<Point2f> srcpoint(matches.size()), dstpoint(matches.size());
    for(int i=0; i<matches.size(); i++)
    {
        srcpoint[i] = querykeypoint[matches[i].queryIdx].pt;
        dstpoint[i] = reainkeypoint[matches[i].queryIdx].pt;
    }
    vector<int> inliermask(srcpoint.size());
    findHomography(srcpoint, dstpoint, RANSAC, 5, inliermask);
    for(int i=0; i<inliermask.size(); i++)
    {
        if(inliermask[i])
        {
            matches_ransac.push_back(matches[i]);
        }
    }
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

    //提取orb特征点
    vector<KeyPoint> kp_book, kp_desk;
    Mat des_book, des_desk;
    orb_feature_detect(book, kp_book, des_book);
    orb_feature_detect(book_on_desk, kp_desk, des_desk);

    //使用暴力匹配特征点
    vector<DMatch> Matche_BF;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(des_book, des_desk, Matche_BF);
    cout << "match = " << Matche_BF.size() << endl;
    
    //计算最小汉明距离
    double min_dist = 10000, max_dist = 0;
    for(int i=0; i<Matche_BF.size(); i++)
    {
        double dist = Matche_BF[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    cout << "min_dist = " << min_dist << endl;
    cout << "max_dist = " << max_dist << endl;
    vector<DMatch> good_matches_bf;
    for (int i = 0; i < Matche_BF.size(); i++) {
        if (Matche_BF[i].distance <= max(2 * min_dist, 30.0)) { // 避免min_dist为0
            good_matches_bf.push_back(Matche_BF[i]);
        }
    }

    Mat outimg1;
    drawMatches(book, kp_book, book_on_desk, kp_desk, good_matches_bf, outimg1);
    imshow("good_matches_bf", outimg1);

    //===========================================//

    vector<DMatch> good_ransac;
    ransacMatchFilter(good_matches_bf, kp_book, kp_desk, good_ransac);
    cout << "ransac.size = " << good_ransac.size() << endl;



    //===========================================//

    if((des_book.type() != CV_32F) && (des_desk.type() != CV_32F))
    {
        des_book.convertTo(des_book, CV_32F);
        des_desk.convertTo(des_desk, CV_32F);
    }

    vector<DMatch> Matche_FLANN;
    FlannBasedMatcher matcher_flann;
    matcher_flann.match(des_book, des_desk, Matche_FLANN);
    cout << "Matche_flann match = " << Matche_FLANN.size() << endl;

    double max_dist2 = 0; 
    double min_dist2 = 100;
    for(int i=0; i < des_book.rows; i++)
    {
        double dist2 = Matche_FLANN[i].distance;
        if(dist2 < min_dist2) min_dist2 = dist2;
        if(dist2 > max_dist2) max_dist2 = dist2;
    }
    cout << "min_dist = " << min_dist2 << endl;
    cout << "max_dist = " << max_dist2 << endl;
    vector<DMatch> good_matches_flann;
    for(int i=0; i < des_book.rows; i++)
    {
        if(Matche_FLANN[i].distance < 0.4 * max_dist2)
        {
            good_matches_flann.push_back(Matche_FLANN[i]);
        }
    }
    Mat outimg2;
    drawMatches(book, kp_book, book_on_desk, kp_desk, good_matches_flann, outimg2);
    imshow("good_matches_flann", outimg2);


    while (true) {
        int c = waitKey(1);
        if (c == 27) {
            break;
        }
    }
    destroyAllWindows();
    return 0;
}

//& "C:/Program Files/mingw64/bin/g++.exe" -g test_feature_detect.cpp -o test_feature_detect.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual

//test_feature_detect
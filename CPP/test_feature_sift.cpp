#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>  
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;  

// 自定义SIFT暴力匹配+Ratio Test筛选函数（替代原QuickDemo的ORBBFMatcher）
void SIFTBFMatcher(const Mat& img1, const Mat& img2,
                   const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2,
                   const Mat& des1, const Mat& des2,
                   vector<DMatch>& goodMatches) {
    // SIFT是浮点型描述子，使用NORM_L2距离
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn_matches;
    // KNN匹配，取Top2
    matcher.knnMatch(des1, des2, knn_matches, 2);

    // Lowe's Ratio Test筛选优质匹配（经验阈值0.75）
    const float ratio_thresh = 0.75f;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            goodMatches.push_back(knn_matches[i][0]);
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

    // 初始化SIFT检测器
    Ptr<SIFT> sift = SIFT::create(
        0,    // 最大关键点数量
        3,      // 尺度空间层数
        0.04,   // 对比度阈值（值越小关键点越多）
        10,     // 边缘阈值（值越大过滤越少边缘点）
        1.6     // 初始高斯核sigma值
    );

    // 检测SIFT关键点并计算描述子
    vector<KeyPoint> kp_book, kp_desk;
    sift->detect(book, kp_book);                
    sift->detect(book_on_desk, kp_desk);        

    Mat des_book, des_desk;
    sift->compute(book, kp_book, des_book);     
    sift->compute(book_on_desk, kp_desk, des_desk); 

    // 绘制SIFT关键点
    Mat result_book, result_desk;
    drawKeypoints(book, kp_book, result_book, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(book_on_desk, kp_desk, result_desk, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SIFT keypoint-book", result_book);
    imshow("SIFT keypoint-deskbook", result_desk);
    waitKey(1); 

    vector<DMatch> goodMatches;
    SIFTBFMatcher(book, book_on_desk, kp_book, kp_desk, des_book, des_desk, goodMatches);
    
    // 绘制匹配结果
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

    // 显示匹配结果
    imshow("SIFT暴力特征匹配结果", matchResult);
    cout << "筛选后的优质匹配对数：" << goodMatches.size() << endl;
  
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

//& "C:/Program Files/mingw64/bin/g++.exe" -g test_feature_sift.cpp -o test_feature_sift.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/MinGW/lib -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_xfeatures2d4120 -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual -DOPENCV_ENABLE_NONFREE=1
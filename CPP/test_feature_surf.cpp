#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>  
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;  

// 自定义SURF暴力匹配+Ratio Test筛选函数
void SURFBFMatcher(const Mat& img1, const Mat& img2,
                   const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2,
                   const Mat& des1, const Mat& des2,
                   vector<DMatch>& goodMatches) {
    // SURF也是浮点型描述子，使用NORM_L2距离
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(des1, des2, knn_matches, 2);

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

    Ptr<SURF> surf = SURF::create(
        500,    
        4,      
        3,      
        false,  
        false   
    );

    // 检测SURF关键点并计算描述子
    vector<KeyPoint> kp_book, kp_desk;
    surf->detect(book, kp_book);                
    surf->detect(book_on_desk, kp_desk);        
    Mat des_book, des_desk;
    surf->compute(book, kp_book, des_book);     
    surf->compute(book_on_desk, kp_desk, des_desk); 
    cout << "book图SURF关键点数量：" << kp_book.size() << endl;
    cout << "desk图SURF关键点数量：" << kp_desk.size() << endl;

    // 绘制SURF关键点
    Mat result_book, result_desk;
    drawKeypoints(book, kp_book, result_book, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(book_on_desk, kp_desk, result_desk, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SURF keypoint-book", result_book);
    imshow("SURF keypoint-deskbook", result_desk);

    // 特征匹配
    vector<DMatch> goodMatches;
    SURFBFMatcher(book, book_on_desk, kp_book, kp_desk, des_book, des_desk, goodMatches);
    
    // 绘制匹配结果
    Mat matchResult;
    drawMatches(
        book, kp_book,                
        book_on_desk, kp_desk,        
        goodMatches,                  
        matchResult,                  
        Scalar(0, 255, 0),            
        Scalar(255, 0, 0),           
        vector<char>(),               
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );
    imshow("SURF暴力特征匹配结果", matchResult);
    cout << "筛选后的优质匹配对数：" << goodMatches.size() << endl;
  
    waitKey(0); 
    destroyAllWindows();
    return 0;
}



//& "C:/Program Files/mingw64/bin/g++.exe" -g test_feature_surf.cpp -o test_feature_surf.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/MinGW/lib -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_xfeatures2d4120 -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual -DOPENCV_ENABLE_NONFREE=1
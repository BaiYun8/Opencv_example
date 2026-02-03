#include <opencv2/opencv.hpp>
#include <iostream>
#include "quickdemo.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    Mat sample = imread("F:/Test/CPP/picture/sample2.png"); 
    Mat target = imread("F:/Test/CPP/picture/target.png");

    if (sample.empty() || target.empty()) {
        printf("could not load image....\n");
        return -1;
    }

    // 显示缩放后的图片
    namedWindow("input window", WINDOW_AUTOSIZE); 
    imshow("sample", sample);
    imshow("target", target);


    //// 对缩放后的图片进行亮度调整
    QuickDemo qd;
    qd.histogram_demo2(sample, target);
    
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

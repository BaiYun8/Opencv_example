#include <opencv2/opencv.hpp>
#include <iostream>
#include "quickdemo.h"

using namespace cv;
using namespace std;


int main()
{
    VideoCapture capture("F:/Test/CPP/panda.mp4");
    if(!capture.isOpened())
    {
        printf("could not open the camera...\n");
        return -1;
    }
    namedWindow("frame", WINDOW_AUTOSIZE);

    ////特征捕获
    //int fps = capture.get(CAP_PROP_FPS);
    //int width = capture.get(CAP_PROP_FRAME_WIDTH);
    //int height = capture.get(CAP_PROP_FRAME_HEIGHT);
    //int num_of_frames = capture.get(CAP_PROP_FRAME_COUNT);
    //int type = capture.get(CAP_PROP_FOURCC);
    //printf("frame size(w=%d, h=%d), FPS:%d, frames: %d \n", width,height,fps,num_of_frames);
    
    //
    Mat frame, hsv, mask, result;
    while(true)
    {
        bool ret = capture.read(frame);
        if(!ret) break;
        imshow("frame", frame);
        cvtColor(frame, hsv, COLOR_RGB2HLS);
        imshow("hsv", hsv);
        inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
        imshow("mask", mask);


        char c = waitKey(50);
        if(c == 27){
            break;
        }
    }
}

//& "C:/Program Files/mingw64/bin/g++.exe" -g test_video.cpp -o test_video.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual

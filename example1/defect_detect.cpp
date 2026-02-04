#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

Mat tpl;
void sort_box(vector<Rect> &boxes);
void detect_defect(Mat &binary, vector<Rect> rects, vector<Rect> &defect);

int main(int argc, char** argv)
{
    Mat src = imread("F:/Test/example1/ce_01.jpg");
    if(src.empty())
    {
        cout << "imread failed ..." << endl;
        return -1;
    }
    namedWindow("imput", WINDOW_AUTOSIZE);
    imshow("imput", src);

    // 图像二值化
    Mat gray, binary;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    imshow("binary", binary);
    // 定义结构元素并开运算
    Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
    morphologyEx(binary, binary, MORPH_OPEN, se);
    imshow("binary2", binary);

    // 轮廓发现
    vector<vector<Point>> contours; //轮廓
	vector<Vec4i> hierarchy;        //存放轮廓的结构变量
    vector<Rect> rects;
    findContours(binary, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
    int height = src.rows;
    for(size_t t=0; t<contours.size(); t++)
    {
       //给定一个轮廓点集，返回能“刚好包住它”的最小轴对齐矩形。把每一个不规则齿条轮廓，转成一个规则的ROI矩形
       Rect rect = boundingRect(contours[t]); 
       double area = contourArea(contours[t]);
       if(area < 150)
       {
        continue;
       }
       if(rect.height > (height/2))
       {
        continue;
       }
       rects.push_back(rect);
       //rectangle(src, rect, Scalar(0, 0, 255), 2, 8, 0);
       //drawContours(src, contours, t, Scalar(0, 0, 255), 2, 8);
    }

    sort_box(rects);
    tpl = binary(rects[1]);

    vector<Rect> defects;
	detect_defect(binary, rects, defects);

    for (int i = 0; i < defects.size(); i++) {
		rectangle(src, defects[i], Scalar(0, 0, 255), 2, 8, 0);
		putText(src, "bad", defects[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 1, 8);
	}
	imshow("detect result", src);
	imwrite("F:/Test/example1/detection_result.png", src);

    waitKey(0);
	return 0;
    
    //while (true) {
    //    int c = waitKey(1);
    //    if (c == 27) {
    //        break;
    //    }
    //}
    //destroyAllWindows();
    //return 0;
}


/*
    binary：二值化后的图像（0 或 255），表示可能的目标或背景
    rects： 候选检测区域（矩形列表）
    defect：检测到的缺陷区域（矩形列表）
    tpl：   模板图像
*/
void detect_defect(Mat &binary, vector<Rect> rects, vector<Rect> &defect)
{
    int h = tpl.rows;
	int w = tpl.cols;
	int size = rects.size();
    for(int i = 0; i < size; i++)
    {
        //构建差异（diff）
        Mat roi = binary(rects[i]);
        resize(roi, roi, tpl.size());
        Mat mask; //差异图
        subtract(tpl, roi, mask);  //模板与当前 ROI 做差, mask 中像素值越大，表示差异越明显。
        
        Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
		morphologyEx(mask, mask, MORPH_OPEN, se);
		threshold(mask, mask, 0, 255, THRESH_BINARY);
		//imshow("mask", mask);
		//waitKey(0);

        //统计差异像素数量
        int count = 0;
        for(int row = 0; row < h; row++)
        {
            for(int col = 0; col < w; col++)
            {
                int pv = mask.at<uchar>(row, col);
                if(pv == 255)
                {
                    count++;
                }
            }
        }

        int mh = mask.rows + 2;
        int mw = mask.cols + 2;
        Mat m1 = Mat::zeros(Size(mw, mh), mask.type());
        Rect mroi;
        mroi.x = 1;
        mroi.y = 1;
        mroi.height = mask.rows;
        mroi.width = mask.cols;
        mask.copyTo(m1(mroi));

        // 轮廓分析
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(m1, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
		
        bool find = false;
		for (size_t t = 0; t < contours.size(); t++) {
			Rect rect = boundingRect(contours[t]);
			float ratio = (float)rect.width / ((float)rect.height);
            // 过滤：宽高比>4且贴紧上下边缘的轮廓（判定为无效噪声/干扰）
			if (ratio > 4.0 && (rect.y < 5 || (m1.rows - (rect.height + rect.y)) < 10)) {
				continue;
			}
            // 计算当前轮廓的实际面积
			double area = contourArea(contours[t]);
			if (area > 10) {
				printf("ratio : %.2f, area : %.2f \n", ratio, area);
				find = true;
			}
		}

		if (count > 50 && find) {
			printf("count : %d \n", count);
			defect.push_back(rects[i]);
		}
    }
}

void sort_box(vector<Rect> &boxes)
{
    int size = boxes.size();
	for (int i = 0; i < size - 1; i++) 
    {
		for (int j = i; j < size; j++) 
        {
			int x = boxes[j].x;
			int y = boxes[j].y;
			if (y < boxes[i].y) 
            {
				Rect temp = boxes[i];
				boxes[i] = boxes[j];
				boxes[j] = temp;
			}
		}
	}
}


//& "C:/Program Files/mingw64/bin/g++.exe" -g defect_detect.cpp -o defect_detect.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual

//defect_detect
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;


Mat transformCorner(Mat &image, RotatedRect &rect);
bool isXCorner(Mat &image);
void scanAndDetectQRCode(Mat & image);
int main()
{
    Mat src = imread("F:/Test/example3/wxqrcode.jpg");
    if(src.empty())
    {
        printf("could not load image file...");
		return -1;
    }
    namedWindow("input", WINDOW_AUTOSIZE);
    imshow("input", src);

    //找到并框住角
    scanAndDetectQRCode(src);

    while (true) {
        int c = waitKey(1);
        if (c == 27) {
            break;
        }
    }
    destroyAllWindows();
    return 0;
}


void scanAndDetectQRCode(Mat & image) 
{
    Mat gray, binary; 
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("binary", binary);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    //1. 寻找轮廓
    findContours(binary.clone(), contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point()); 
    Mat result = Mat::zeros(image.size(), CV_8UC1);
    for(size_t i=0; i<contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if(area < 100)
        {
            continue;
        }

        RotatedRect rect = minAreaRect(contours[i]);
        float width = rect.size.width;
        float height = rect.size.height;
        float rate = min(width, height) / max(width, height);
        if(rate > 0.85 && width < image.cols / 4.0f && height < image.rows / 4.0f)
        {
            Mat qr_roi = transformCorner(image, rect);

            if(isXCorner(qr_roi))
            {
                //// 1. 显示矫正后的角点图像（窗口名可自定义）
                //imshow("result", qr_roi);
                //// 2. 等待用户操作：0表示无限等待，直到按任意键/关闭窗口
                //waitKey(0);
                //// 3. 释放窗口资源（可选，但推荐，避免内存占用）
                //destroyWindow("result");

                drawContours(image, contours, static_cast<int>(i), Scalar(255, 0, 0), 2, 8);
                drawContours(result, contours, static_cast<int>(i), Scalar(255), 2, 8);
            }
        }
    }

    // 1. 从 result 中提取所有白色像素点
    vector<Point> whitePoints;
    findNonZero(result, whitePoints);
    if (whitePoints.empty())
    {
        cout << "test here no picture" << endl;
        return;
    }
    
    // 2. 求能包围所有白色像素点的最小旋转矩形
    RotatedRect qrRect = minAreaRect(whitePoints);
    Point2f pts[4];
    qrRect.points(pts); 

    // 3. 用旋转矩形生成掩码
    Mat mask = Mat::zeros(image.size(), CV_8UC1);
    vector<Point> qrCorners;
    for (int i = 0; i < 4; i++)
        qrCorners.push_back(Point(pts[i])); 
        //把图像区域填充白色，形成黑色背景的掩码
    fillConvexPoly(mask, qrCorners, Scalar(255));

    // 4. 用掩码从原始图像中精准抠出二维码 ROI
    Mat qr_roi;
    image.copyTo(qr_roi, mask);

    // 5. 裁剪到最小外接矩形的边界框（防止越界）
    Rect boundRect = qrRect.boundingRect();
    boundRect &= Rect(0, 0, image.cols, image.rows);
    Mat qr_crop = qr_roi(boundRect);

    imshow("QR ROI", qr_crop);
    imshow("output", image);
}


bool isXCorner(Mat &image)
{
    Mat gray, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    //验证中心像素是否为黑色
    int width = binary.cols;
    int height = binary.rows;
    int cy = height/2;
    int cx = width / 2;
    int pv = binary.at<uchar>(cy, cx);
    if(pv==255) return false;

    //查找中心黑块的左右边界
    int xb = 0;
    int offset = 0;
    int start = 0, end = 0;
    bool findleft = false, findright = false;
    while(true)
    {
        offset++;

        if ((cx - offset) <= width / 8 || (cx + offset) >= width - 1) 
        {
            start = -1;
            end = -1;
            break;
        }

        pv = binary.at<uchar>(cy, cx - offset);
        if(pv == 255){
            start = cx - offset;
            findleft = true;
        }

        pv = binary.at<uchar>(cy, cx + offset);
        if(pv == 255){
            end = cx + offset;
            findright = true;
        }

        if(findleft && findright)
        {
            break;
        }
    }
    if(start <= 0 || end <= 0)
    {
        return false;
    }
    xb = end - start; // xb = 中心黑块的宽度
    

    //查找中心黑块两侧的白环宽度
    int w1x = 0, w2x = 0;
    for(int col=start; col>0; col--)
    {
        pv = binary.at<uchar>(cy, col);
        if(pv == 0)
        {
            w1x = start - col;
            break;
        }
    }
    for (int col = end; col < width - 1; col++) {
		pv = binary.at<uchar>(cy, col);
		if (pv == 0) {
			w2x = col - end;
			break;
		}
	}


    // 向左找：左侧白环左边界 → 最左侧边缘 之间的黑块宽度（b1x）
    int b1x = 0, b2x = 0;
    for (int col = (start - w1x); col >0; col--) {
        pv = binary.at<uchar>(cy, col);
        if (pv == 255) {  // 找到最左侧黑块的左边界（白色像素）
            b1x = start - col - w1x;
            break;
        }
        else {
            b1x++;
        }
    }
    // 向右找：右侧白环右边界 → 最右侧边缘 之间的黑块宽度（b2x）
    for (int col = (end + w2x); col < width; col++) {
        pv = binary.at<uchar>(cy, col);
        if (pv == 255) {  // 找到最右侧黑块的右边界（白色像素）
            b2x = col - end - w2x;
            break;
        }
        else {
            b2x++;
        }
    }

    float sum = xb + b1x + b2x + w1x + w2x;  // 总长度（对应7个模块）
    // 归一化到7个模块（×7），四舍五入（+0.5）转为整数
    xb = static_cast<int>((xb / sum)*7.0 + 0.5);
    b1x = static_cast<int>((b1x / sum)*7.0 + 0.5);
    b2x = static_cast<int>((b2x / sum)*7.0 + 0.5);
    w1x = static_cast<int>((w1x / sum)*7.0 + 0.5);
    w2x = static_cast<int>((w2x / sum)*7.0 + 0.5);

    if ((xb == 3 || xb == 4) && b1x == b2x && w1x == w2x && w1x == b1x && b1x == 1) 
    {
        return true;
    }
    else {
        return false;
    }

}







Mat transformCorner(Mat &image, RotatedRect &rect)
{
    //获取4个源点坐标
    Point2f src_pts[4];
    rect.points(src_pts);
    vector<Point> src_corners;
    for(int i=0; i<4; i++)
    {
        src_corners.push_back(src_pts[i]);
    }

    //定义目标矩阵的四个点
    vector<Point> dst_corners;
    int width = static_cast<int>(rect.size.width);
    int height = static_cast<int>(rect.size.height);
    dst_corners.push_back(Point(0,0));
    dst_corners.push_back(Point(width,0));
    dst_corners.push_back(Point(width,height));
    dst_corners.push_back(Point(0,height));

    //计算单应性矩阵（核心！实现点集的映射）
    Mat h = findHomography(src_corners, dst_corners);

    Mat result = Mat::zeros(height, width, image.type());
    warpPerspective(image, result, h, result.size());

    
    return result;
} 



//& "C:/Program Files/mingw64/bin/g++.exe" -g qr_code.cpp -o qr_code.exe -I F:/opencv/opencv/build/x64/install/include -L F:/opencv/opencv/build/x64/install/x64/mingw/lib -lopencv_world4120 -std=c++17 -Wall -Wno-overloaded-virtual

//qr_code


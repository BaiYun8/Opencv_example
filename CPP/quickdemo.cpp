#include <opencv2/dnn.hpp>
#include "quickdemo.h"
#include <random>

using namespace cv;
using namespace std;


//图像色彩空间转化
void QuickDemo::colorSpace_Demo(Mat &image) {
	Mat gray, hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	// H 0 ~ 180, S, V 
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imwrite("F:/Test/CPP/test_picture_hsv.jpg", hsv);
	imwrite("F:/Test/CPP/test_picture_gray.jpg", gray);
    cout << "HSV and Gray images saved!" << endl;
}

//图像创建和赋值
void QuickDemo :: mat_creation_demo()
{
	// Mat m1, m2;
	// m1 = image.clone();
	// image.copyTo(m2);

	// 创建空白图像
	Mat m3 = Mat::zeros(Size(8, 8), CV_8UC3);
	m3 = Scalar(0, 0, 255);
	std::cout << "width: " << m3.cols << " height: " << m3.rows << " channels: "<<m3.channels()<< std::endl;
	// std::cout << m3 << std::endl;

	Mat m4;
	m3.copyTo(m4);
	m4 = Scalar(0, 255, 255);
	imshow("图像", m3);
	imshow("图像4", m4);
}

void QuickDemo::pixel_visit_demo(Mat &image)
{
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();

	/*
	for(int row =0; row < h; row++)
	{
		for(int col =0; col < w; col++)
		{
			if(dims == 1)
			{
				image.at<uchar>(row, col) = 255 - image.at<uchar>(row, col);

			}
			if(dims == 3)
			{
				Vec3b bgr = image.at<Vec3b>(row, col);
				image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
				image.at<Vec3b>(row, col)[2] = 255 - bgr[2];
			}
		}
	}
	*/

	for (int row = 0; row < h; row++) 
    {
        uchar* current_row = image.ptr<uchar>(row);
        for (int col = 0; col < w; col++) 
        {
            if (dims == 1) 
            { // 灰度图像（这部分没问题，无需修改）
                int pv = *current_row;
                *current_row++ = 255 - pv;
            }
            if (dims == 3) 
            { // 彩色图像：拆分成“先读值，后移动指针”
                // 第一步：先读取当前指针指向的B/G/R值（此时指针未移动）
                uchar b_val = *current_row;    // 读取B通道值
                uchar g_val = *(current_row+1); // 读取G通道值
                uchar r_val = *(current_row+2); // 读取R通道值
                
                // 第二步：赋值并移动指针（顺序明确，无歧义）
                *current_row++ = 255 - b_val; // B通道取反，指针+1
                *current_row++ = 255 - g_val; // G通道取反，指针+1
                *current_row++ = 255 - r_val; // R通道取反，指针+1
            }
        }
    }
}

static void on_lightness(int b, void* userdata) {
	Mat image = *((Mat*)userdata);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	addWeighted(image, 1.0, m, 0, b, dst);
	imshow("亮度与对比度调整", dst);
}

static void on_contrast(int b, void* userdata) {
	Mat image = *((Mat*)userdata);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	double contrast = b / 100.0;
	addWeighted(image, contrast, m, 0.0, 0, dst);
	imshow("亮度与对比度调整", dst);
}


void QuickDemo::tracking_bar_demo(Mat &image) {
	namedWindow("亮度与对比度调整", WINDOW_FREERATIO);
	int lightness = 50;
	int max_value = 100;
	int contrast_value = 100;
	createTrackbar("Value Bar:", "亮度与对比度调整", &lightness, max_value, on_lightness, (void*) (&image));
	createTrackbar("Contrast Bar:", "亮度与对比度调整", &contrast_value, 200, on_contrast, (void*)(&image));
	on_lightness(50, &image);
}


void QuickDemo::color_style_demo(Mat &image) {
	int colormap[] = {
		COLORMAP_AUTUMN,
		COLORMAP_BONE,
		COLORMAP_JET,
		COLORMAP_WINTER,
		COLORMAP_RAINBOW,
		COLORMAP_OCEAN,
		COLORMAP_SUMMER,
		COLORMAP_SPRING,
		COLORMAP_COOL,
		COLORMAP_PINK,
		COLORMAP_HOT,
		COLORMAP_PARULA,
		COLORMAP_MAGMA,
		COLORMAP_INFERNO,
		COLORMAP_PLASMA,
		COLORMAP_VIRIDIS,
		COLORMAP_CIVIDIS,
		COLORMAP_TWILIGHT,
		COLORMAP_TWILIGHT_SHIFTED
	};
	
	Mat dst;
	int index = 0;
	while (true) {
		int c = waitKey(500);
		if (c == 27) { // 退出
			break;
		}
		applyColorMap(image, dst, colormap[index%19]);
		index++;
		imshow("颜色风格", dst);
	}
}

//图像像素的逻辑操作
void QuickDemo::bitwise_demo(Mat &image) {
	Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
	rectangle(m1, Rect(100, 100, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(m2, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);
	imshow("m1", m1);
	imshow("m2", m2);
	Mat dst;
	bitwise_xor(m1, m2, dst);
	imshow("像素位操作", dst);
}

//单个通道分离
void QuickDemo::channels_demo(Mat &image) {
	std::vector<Mat> mv;
	split(image, mv);
	imshow("蓝色", mv[0]);
	imshow("绿色", mv[1]);
	imshow("红色", mv[2]);

	Mat dst;
	mv[0] = 0;
	// mv[1] = 0;
	merge(mv, dst);
	imshow("红色", dst);

	int from_to[] = { 0, 2, 1,1, 2, 0 };
	mixChannels(&image, 1, &dst, 1, from_to, 3);
	imshow("通道混合", dst);
}

//色彩空间转化
void QuickDemo::inrange_demo(Mat &image) {
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);

	Mat redback = Mat::zeros(image.size(), image.type());
	redback = Scalar(40, 40, 200);//红色
	bitwise_not(mask, mask);
	imshow("mask", mask);
	image.copyTo(redback, mask);
	imshow("roi区域提取", redback);
}

void QuickDemo::pixel_statistic_demo(Mat &image)
{
	double minv,maxv;
	Point minLoc, maxLoc;
	std::vector<Mat> mv;
	split(image, mv);
	for(size_t i=0; i<mv.size(); i++)
	{
		minMaxLoc(mv[i], &minv, &maxv, &minLoc, &maxLoc, Mat());
		std::cout <<"No. channels:"<< i << " min value:" << minv << " max value:" << maxv << std::endl;
	}
	Mat mean, stddev;
	Mat redback = Mat::zeros(image.size(),image.type());
	redback = Scalar(40, 40, 200);
	meanStdDev(redback, mean, stddev);
	imshow("redback", redback);
	std::cout << "means:" << mean << std::endl;
	std::cout<< " stddev:" << stddev << std::endl;

}

//在空白画布上绘制矩形、圆形、直线、椭圆（旋转矩形），再与原始图像融合
void QuickDemo::drawing_demo(Mat &image) 
{
	Rect rect;//Rect是 OpenCV 的矩形类
	rect.x = 100;
	rect.y = 100;
	rect.width = 250;
	rect.height = 300;

	Mat bg = Mat::zeros(image.size(), image.type());
	rectangle(bg, rect, Scalar(0,0,255), -1, 8 ,0);
	circle(bg, Point(350, 400), 15, Scalar(255, 0, 0), -1, 8, 0);
	line(bg, Point(100, 100), Point(350, 400), Scalar(0, 255, 0), 4, LINE_AA, 0);
	
	RotatedRect rrt;
	rrt.center = Point(200, 200);  // 旋转矩形的中心坐标
	rrt.size = Size(100, 200);     // 旋转矩形的尺寸（宽100，高200）
	rrt.angle = 90.0;              // 旋转角度（90度，顺时针）
	ellipse(bg, rrt, Scalar(0, 255, 255), 2, 8);

	Mat dst;
	addWeighted(image, 0.7, bg, 0.3, 0, dst);
	imshow("绘制演示", bg);

}

//512x512 的黑色画布上随机绘制彩色直线
void QuickDemo::random_drawing()
{
	Mat canvas = Mat::zeros(Size(512,512), CV_8UC3);
	int w = canvas.cols;
	int h = canvas.rows;

	RNG rng(12345);
	while(true)
	{
		int c = waitKey(10);
		if(c == 27)
		{
			break;
		}
		int x1 = rng.uniform(0, w);
		int y1 = rng.uniform(0, h);
		int x2 = rng.uniform(0, w);
		int y2 = rng.uniform(0, h);
		int b = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int r = rng.uniform(0, 255);
		line(canvas, Point(x1, y1), Point(x2, y2), Scalar(b, g, r), 1, LINE_AA, 0);
		imshow("随机绘制演示", canvas);
	}
}

void QuickDemo::polyline_drawing_demo() {
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	//定义多边形的顶点坐标
	Point p1(100, 100);
	Point p2(300, 150);
	Point p3(300, 350);
	Point p4(250, 450);
	Point p5(50, 450);
	//构建顶点向量（存储多边形顶点）
	std::vector<Point> pts;
	pts.push_back(p1);
	pts.push_back(p2);
	pts.push_back(p3);
	pts.push_back(p3);
	pts.push_back(p4);
	pts.push_back(p5);
	// polylines(canvas, pts, true, Scalar(0, 255, 0), -1, 8, 0);
	std::vector<std::vector<Point>> contours;
	contours.push_back(pts);
	//drawContours的核心要求：输入是二维向量，因为它支持同时绘制多个轮廓；
	drawContours(canvas, contours, 0, Scalar(0, 0, 255), -1, 8);
	imshow("绘制多边形", canvas);
}


Point sp(-1, -1);//矩形起点（初始值-1表示未激活）
Point ep(-1, -1);//矩形终点
Mat temp;
//参数：鼠标事件类型，鼠标事件发生时的像素坐标，鼠标按键 / 键盘辅助标志，用户自定义传入的数据
static void on_draw(int event, int x, int y, int flags, void *userdata) {
	Mat image = *((Mat*)userdata);//解析传入的用户数据（原始图像）
	//鼠标左键按下
	if (event == EVENT_LBUTTONDOWN)
	{
		sp.x = x;
		sp.y = y;
		std::cout <<"start point:" << sp << std::endl;
	}
	//鼠标左键松开
	else if (event == EVENT_LBUTTONUP) 
	{
		ep.x = x;
		ep.y = y;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0) {
				Rect box(sp.x, sp.y, dx, dy);//构建矩形区域
				temp.copyTo(image);
				imshow("ROI区域", image(box));
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
				imshow("鼠标绘制", image);
				// ready for next drawing
				sp.x = -1;
				sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE) {
		if (sp.x > 0 && sp.y > 0) {
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0) {
				Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
				imshow("鼠标绘制", image);
			}
		}
	}
}


void QuickDemo::mouse_drawing_demo(Mat &image) {
	namedWindow("鼠标绘制", WINDOW_AUTOSIZE);
	setMouseCallback("鼠标绘制", on_draw, (void*)(&image));
	imshow("鼠标绘制", image);
	temp = image.clone();
}

//图像数据归一化
void QuickDemo::norm_demo(Mat &image) {
	Mat dst;
	std::cout << image.type() << std::endl;
	image.convertTo(image, CV_32F);
	std::cout << image.type() << std::endl;
	normalize(image, dst, 1.0, 0, NORM_MINMAX);
	std::cout << dst.type() << std::endl;
	imshow("normalize", dst);
	Vec3f pixel = dst.at<Vec3f>(0,0);
	std::cout << "归一化后像素值:B=" << pixel[0] << ", G=" << pixel[1] << ", R=" << pixel[2] << std::endl;
	// CV_8UC3, CV_32FC3
}

//图像尺寸缩放
void QuickDemo::resize_demo(Mat &image) {
	Mat zoomin, zoomout;
	int h = image.rows;
	int w = image.cols;
	//缩小图像：尺寸变为原始的1/2
	resize(image, zoomin, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
	imshow("zoomin", zoomin);
	//放大图像：尺寸变为原始的1.5倍
	resize(image, zoomout, Size(w*1.5, h*1.5), 0, 0, INTER_LINEAR);
	imshow("zoomout", zoomout);
}

//图像翻转
void QuickDemo::flip_demo(Mat &image) 
{
	Mat dst;
	// flip(image, dst, 0); // 上下翻转
	// flip(image, dst, 1); // 左右翻转
	flip(image, dst, -1); // 180°旋转
	imshow("图像翻转", dst);
}

void QuickDemo::rotate_demo(Mat &image) 
{
	Mat dst, M;
	int w = image.cols;
	int h = image.rows;
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), 45, 1.0);
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));
	int nw = cos*w + sin*h;
	int nh = sin*w + cos*h;
	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1,2) += (nh / 2 - h / 2);
	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(255, 255, 0));
	imshow("旋转演示", dst);
}

void QuickDemo::video_demo(Mat &image) {
	VideoCapture capture("F:/Test/CPP/panda.mp4");
	//获取视频的基础属性（宽、高、总帧数、帧率）
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int count = capture.get(CAP_PROP_FRAME_COUNT);//总帧数
	double fps = capture.get(CAP_PROP_FPS);//帧率（每秒帧数）
	std::cout << "frame width:" << frame_width << std::endl;
	std::cout << "frame height:" << frame_height << std::endl;
	std::cout << "FPS:" << fps << std::endl;
	std::cout << "Number of Frames:" << count << std::endl;
	//VideoWriter writer("D:/test_video.mp4", capture.get(CAP_PROP_FOURCC), fps, Size(frame_width, frame_height), true);
	int fourcc = VideoWriter::fourcc('X','V','I','D'); // XVID编码（Windows默认支持）
	VideoWriter writer("F:/Test/CPP/test.avi", fourcc, fps, Size(frame_width, frame_height), true);

	Mat frame;
	while (true) {
		capture.read(frame);
		flip(frame, frame, 1);
		if (frame.empty()) {
			break;
		}
		imshow("frame", frame);
		colorSpace_Demo(frame);
		writer.write(frame);
		// TODO: do something....
		int c = waitKey(1);
		if (c == 27) { // 退出
			break;
		}
	}

	// release
	capture.release();
	writer.release();
}

void QuickDemo::histogram_demo(Mat &image) {
	// 三通道分离
	std::vector<Mat> bgr_plane;
	split(image, bgr_plane);
	// 定义参数变量
	const int channels[1] = { 0 };//计算直方图的通道索引（单通道固定为0）
	const int bins[1] = { 256 };//直方图的“区间数”（bin数）：0~255分256个区间
	float hranges[2] = { 0,255 };//像素值范围：0（黑）~255（白）
	const float* ranges[1] = { hranges };// 像素值范围的指针（calcHist要求格式）
	Mat b_hist;
	Mat g_hist;
	Mat r_hist;
	// 计算Blue, Green, Red通道的直方图
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);

	// 显示直方图
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / bins[0]);
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
	// 归一化直方图数据
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	// 绘制直方图曲线
	for (int i = 1; i < bins[0]; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}
	// 显示直方图
	namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
	imshow("Histogram Demo", histImage);
}

void QuickDemo::histogram_2d_demo(Mat &image) {
	// 2D 直方图
	Mat hsv, hs_hist;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	int hbins = 30, sbins = 32;
	int hist_bins[] = { hbins, sbins };
	float h_range[] = { 0, 180 };
	float s_range[] = { 0, 256 };
	const float* hs_ranges[] = { h_range, s_range };
	int hs_channels[] = { 0, 1 };
	calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);
	double maxVal = 0;
	minMaxLoc(hs_hist, 0, &maxVal, 0, 0);
	int scale = 10;
	Mat hist2d_image = Mat::zeros(sbins*scale, hbins * scale, CV_8UC3);
	for (int h = 0; h < hbins; h++) {
		for (int s = 0; s < sbins; s++)
		{
			float binVal = hs_hist.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			rectangle(hist2d_image, Point(h*scale, s*scale),
				Point((h + 1)*scale - 1, (s + 1)*scale - 1),
				Scalar::all(intensity),
				-1);
		}
	}
	applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET);
	imshow("H-S Histogram", hist2d_image);
	imwrite("F:/Test/CPP/hist_2d.png", hist2d_image);
}

void QuickDemo::histogram_eq_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("gray picture", gray);
	Mat dst;
	equalizeHist(gray, dst);
	imshow("直方图均衡化演示", dst);
}

void QuickDemo::blur_demo(Mat &image) {
	Mat dst;
	blur(image, dst, Size(15, 15), Point(-1, -1));
	imshow("图像模糊", dst);
}

void QuickDemo::gaussian_blur_demo(Mat &image) {
	Mat dst;
	GaussianBlur(image, dst, Size(0, 0), 15);
	imshow("高斯模糊", dst);
}

void QuickDemo::bifilter_demo(Mat &image) {
	Mat dst;
	bilateralFilter(image, dst, 0, 100, 10);
	imshow("双边模糊", dst);
}

void QuickDemo::face_detection_demo() {
	std::string root_dir = "F:/opencv/opencv_tutorial_data-master/";
	dnn::Net net = dnn::readNetFromTensorflow(root_dir+ "opencv_face_detector_uint8.pb", root_dir+"opencv_face_detector.pbtxt");
	VideoCapture capture("F:/Test/CPP/panda.mp4");
	Mat frame;
	while (true) 
	{
		capture.read(frame);//循环读取每一帧图片
		if (frame.empty()) {
			break;
		}
		Mat blob = dnn::blobFromImage(
            frame,                  // 输入帧
            1.0,                    // 像素值缩放因子（不缩放，保持1.0）
            Size(300, 300),         // 模型输入尺寸（固定300x300）
            Scalar(104, 177, 123),  // 像素均值（BGR顺序，模型训练时的归一化均值）
            false,                  // 是否交换RB通道（模型训练用BGR，OpenCV也是BGR，故false）
            false                   // 是否裁剪图像（保持原比例，仅缩放）
        );
		net.setInput(blob);// NCHW，将预处理后的Blob输入模型
		Mat probs = net.forward(); // 模型前向推理：得到检测结果（人脸位置+置信度）
		Mat detectionMat(
            probs.size[2], probs.size[3],  // 行数=N，列数=7
            CV_32F,                        // 数据类型：32位浮点
            probs.ptr<float>()             // 指向输出张量的指针
        );
		// 解析结果，遍历所有检测框，筛选高置信度人脸并绘制
		for (int i = 0; i < detectionMat.rows; i++) 
		{
			//提取当前检测框的置信度（第3列，索引2）：0~1，越大越可信
			float confidence = detectionMat.at<float>(i, 2);
			//过滤低置信度（只保留>0.5的框，避免误检）
			if (confidence > 0.5) 
			{
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3)*frame.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4)*frame.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5)*frame.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6)*frame.rows);
				Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
		imshow("人脸检测演示", frame);
		int c = waitKey(1);
		if (c == 27) { // 退出
			break;
		}
	}
}

//普通二值化
void QuickDemo::image_binaryzation(Mat &src)
{
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	imshow("gray input", gray);
	threshold(gray, binary, 200, 255, THRESH_BINARY);
	imshow("threshold binary", binary);

	threshold(gray, binary, 127, 255, THRESH_BINARY_INV);
	imshow("threshold binary invert", binary);

	threshold(gray, binary, 127, 255, THRESH_TRUNC);
	imshow("threshold TRUNC", binary);

	threshold(gray, binary, 127, 255, THRESH_TOZERO);
	imshow("threshold to zero", binary);

	threshold(gray, binary, 127, 255, THRESH_TOZERO_INV);
	imshow("threshold to zero invert", binary);
}

//全局二值化--自动阈值
void QuickDemo::image_binaryzation2(Mat &src)
{
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	imshow("gray",gray);
	Scalar m = mean(gray);
	printf("means : %.2f\n", m[0]);

	//算法自动计算出的最优全局阈值
	double t1 = threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("otsu binary", binary);

	double t2 = threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_TRIANGLE);
	imshow("triangle binary", binary);

	printf("otsu theshold : %.2f, triangle threshold : %.2f \n", t1, t2);
}

void QuickDemo::image_connect_detect(Mat &src)
{
	GaussianBlur(src, src, Size(3, 3), 0);
	Mat gray, binary;
	cvtColor(src, gray, COLOR_RGB2GRAY);
	imshow("grey ori_picture", gray);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	imshow("binary", binary);

}


void QuickDemo::ccl_stats_demo(Mat &image)
{
	RNG rng(12345);
	GaussianBlur(image, image, Size(3, 3), 0);//高斯去噪
	Mat gray, binary;
	cvtColor(image, gray, COLOR_RGB2GRAY);
	imshow("grey ori_picture", gray);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);//阈值转化
	imshow("binary", binary);

	Mat labels = Mat::zeros(image.size(), CV_32S);
	Mat stats, centroids;
	int num_lables = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S, CCL_DEFAULT);
	vector<Vec3b> colorTable(num_lables);
	// backgound color，根据标签的数量生成不同颜色的标签
	colorTable[0] = Vec3b(0, 0, 0);
	for(int i=1; i<num_lables; i++)
	{
		colorTable[i] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
	}

	//创建彩色图，根据每个像素点的颜色标签，对其进行赋值上色
	Mat result = Mat::zeros(image.size(), CV_8UC3);
	int w = result.cols;
	int h = result.rows;
	for(int row=0; row<h; row++)
	{
		for(int col=0; col<w; col++)
		{
			int label = labels.at<int>(row, col);//获取当前像素的连通域标签
			result.at<Vec3b>(row, col) = colorTable[label];//根据标签取对应颜色，赋值给结果图的像素
		}
	}

	for (int i = 1; i < num_lables; i++) 
	{
		int cx = centroids.at<double>(i,0);
		int cy = centroids.at<double>(i,1);
		int x = stats.at<int>(i, CC_STAT_LEFT);    // 外接矩形左上角x坐标
		int y = stats.at<int>(i, CC_STAT_TOP);     // 外接矩形左上角y坐标
		int width = stats.at<int>(i, CC_STAT_WIDTH);   // 外接矩形宽度
		int height = stats.at<int>(i, CC_STAT_HEIGHT); // 外接矩形高度
		int area = stats.at<int>(i, CC_STAT_AREA);     // 连通域的像素面积

		circle(result, Point(cx, cy), 3, Scalar(0, 0, 255), 2, 8, 0);
		Rect box(x, y, width, height);
		rectangle(result, box, Scalar(0, 255, 0), 2, 8);
		putText(result, format("%d", area), Point(x, y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 1);
	}
	putText(result, format("number: %d", num_lables - 1), Point(10, 10), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 1);
	printf("total labels : %d \n", (num_lables - 1));
	imshow("CCL demo", result);

}


//轮廓
void QuickDemo::Profile_demo(Mat &image)
{
	GaussianBlur(image, image, Size(3, 3), 0);//高斯去噪
	Mat gray, binary;
	cvtColor(image, gray, COLOR_RGB2GRAY);
	imshow("grey ori_picture", gray);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);//阈值转化
	imshow("binary", binary);


	vector<vector<Point>> contours;// 轮廓集合：嵌套向量
	vector<Vec4i> hirearchy;       // 层级信息
	findContours(binary.clone(), contours, hirearchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	cout << "test to " << contours.size() << " profile" << endl;
	if (!contours.empty()) {
        cout << "the number of first Profile point is :" << contours[0].size() << endl;
	}
	for(size_t t = 0; t < contours.size(); t++)
	{
		drawContours(image, contours, t, Scalar(0, 0, 255), 2, LINE_AA);
	}
	imshow("find contours demo", image);

}

void QuickDemo::nebula_demo(Mat &image)
{
	GaussianBlur(image, image, Size(3, 3), 0);
	Mat gray, binary;
	cvtColor(image, gray, COLOR_RGB2GRAY);
	imshow("gray picture", gray);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("binary picture", binary);

	int height = binary.rows;
	int width = binary.cols;
	vector<vector<Point>> contours;
	vector<Vec4i> hirearchy;
	findContours(binary, contours, hirearchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	
	double max_area = -1;
	double cindex = -1;
	for(size_t t=0; t<contours.size(); t++)
	{
		Rect rect = boundingRect(contours[t]);
		if(rect.height >= height || rect.width >= width)
		{
			continue;
		}
		double area = contourArea(contours[t]);
		double len = arcLength(contours[t], true);
		if (area > max_area) {
			max_area = area;
			cindex = t;
		}
	}
}

Mat drawHSV2DHist_Simple(const Mat& hist, int h_bins, int s_bins) {
    // 1. 固定画布尺寸：宽500，高600
    const int canvas_w = 500;
    const int canvas_h = 600;

    // 2. 获取直方图最大像素数（归一化用）
    double max_val;
    minMaxLoc(hist, 0, &max_val);

    // 3. 创建临时小画布（h_bins × s_bins），先填充像素数对应的亮度
    Mat hist_small(h_bins, s_bins, CV_8UC1, Scalar(0));
    for (int h = 0; h < h_bins; h++) {
        for (int s = 0; s < s_bins; s++) {
            float bin_val = hist.at<float>(h, s);
            uchar brightness = static_cast<uchar>(bin_val / max_val * 255);
            hist_small.at<uchar>(h, s) = brightness;
        }
    }

    // 4. 把小画布等比例拉伸到500×600（INTER_LINEAR保证拉伸后平滑）
    Mat hist_img;
    resize(hist_small, hist_img, Size(canvas_w, canvas_h), 0, 0, INTER_LINEAR);

    return hist_img;
}

void QuickDemo::histogram_demo2(Mat &image1, Mat &image2)
{
	Mat sample_hsv, target_hsv;
	cvtColor(image1, sample_hsv, COLOR_BGR2HSV);
	cvtColor(image2, target_hsv, COLOR_BGR2HSV);
	imshow("sample_hsv", sample_hsv);
	imshow("target_hsv", target_hsv);

	int h_bins = 48, s_bins = 48;
	int histSize[] = { h_bins, s_bins };
	int channels[] = { 0, 1 };
	Mat roiHist;
	float h_range[] = { 0, 180 };
	float s_range[] = { 0, 255 };
	const float* ranges[] = { h_range, s_range };
	calcHist(&target_hsv, 1, channels, Mat(), roiHist, 2, histSize, ranges, true, false);
	normalize(roiHist, roiHist, 0, 255, NORM_MINMAX, -1, Mat());
	MatND backproj; 
	calcBackProject(&sample_hsv, 1, channels, roiHist, backproj, ranges, 1.0);
	imshow("back projection demo", backproj);

}

//采用Harris 角点检测
void harris_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	Mat dst;
	double k = 0.04;
	int blocksize = 2;
	int ksize = 3;
	cornerHarris(gray, dst, blocksize, ksize, k);
	Mat dst_norm = Mat::zeros(dst.size(), dst.type());
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, -1, Mat());
	convertScaleAbs(dst_norm, dst_norm);

	// draw corners
	RNG rng(12345);
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			int rsp = dst_norm.at<uchar>(row, col);
			if (rsp > 150) {
				int b = rng.uniform(0, 255);
				int g = rng.uniform(0, 255);
				int r = rng.uniform(0, 255);
				circle(image, Point(col, row), 5, Scalar(b, g, r), 2, 8, 0);
			}
		}
	}
}

//Shi-Tomasi 角点检测
void shitomas_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	vector<Point2f> corners;
	double quality_level = 0.01;
	RNG rng(12345);
	goodFeaturesToTrack(gray, corners, 200, quality_level, 3, Mat(), 3, false);
	for (int i = 0; i<corners.size(); i++) {
		int b = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int r = rng.uniform(0, 255);
		circle(image, corners[i], 5, Scalar(b, g, r), 2, 8, 0);
	}
}


// 生成椒盐噪声
void addGaussianNoise(Mat& src, Mat& dst, double mean = 0.0, double stddev = 20.0) 
{
    dst = src.clone();
    // 正态分布随机数生成器
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> gaussDist(mean, stddev);

    // 遍历所有像素
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if (dst.channels() == 1) { // 灰度图
                // 叠加噪声并限制范围 [0,255]
                int newValue = saturate_cast<uchar>(dst.at<uchar>(i, j) + gaussDist(gen));
                dst.at<uchar>(i, j) = newValue;
            } else if (dst.channels() == 3) { // 彩色图
                Vec3b& pixel = dst.at<Vec3b>(i, j);
                pixel[0] = saturate_cast<uchar>(pixel[0] + gaussDist(gen)); // B通道
                pixel[1] = saturate_cast<uchar>(pixel[1] + gaussDist(gen)); // G通道
                pixel[2] = saturate_cast<uchar>(pixel[2] + gaussDist(gen)); // R通道
            }
        }
    }
}

void QuickDemo::ORBBFMatcher(const Mat& img1, const Mat& img2, 
                             vector<KeyPoint>& kp1, vector<KeyPoint>& kp2,
                             Mat& des1, Mat& des2, 
                             vector<DMatch>& goodMatches) {
    // ORB描述子是二进制向量，用NORM_HAMMING距离匹配
    BFMatcher matcher(NORM_HAMMING, true); // crossCheck=true：双向匹配更精准
    
    // 执行描述子匹配
    vector<DMatch> allMatches;
    matcher.match(des1, des2, allMatches);
    
    // 筛选优质匹配
    double minDist = 1000;
    for (const auto& match : allMatches) {
        if (match.distance < minDist) {
            minDist = match.distance;
        }
    }
    
    for (const auto& match : allMatches) {
        if (match.distance < max(2 * minDist, 30.0)) { // 避免minDist为0
            goodMatches.push_back(match);
        }
    }
}
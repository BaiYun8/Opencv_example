#include "yolo_ort.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

int main()
{
    vector<string> classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    
    std::shared_ptr<YOLOv5DNNDetector> detector(new YOLOv5DNNDetector());
	detector->initConfig("F:/Test/yolo/yolov5s.onnx", 640, 640, 0.25f, 0.45f);
	
	std::vector<DetectResult> results;

	cv::VideoCapture capture("F:/Test/yolo/test1.mp4");
	double fps = capture.get(cv::CAP_PROP_FPS);//获取原视频帧率
	int frame_delay = 1000 / fps;//计算每帧需要的时间
	cout << "the time using : " << frame_delay << endl;

	cv::Mat frame;
	while (true) {
		auto start_time = high_resolution_clock::now();
		bool ret = capture.read(frame);
		if (frame.empty()) {
			break;
		}

		
		detector->detect(frame, results);

		for (DetectResult dr : results) {
        cv::Rect box = dr.box;
        cv::putText(frame, classNames[dr.classId], 
            cv::Point(box.tl().x, box.tl().y - 10), 
            cv::FONT_HERSHEY_SIMPLEX, .5, 
            cv::Scalar(0, 0, 0));
    	}


		cv::imshow("YOLOv5-6.1 + OpenCV DNN - by gloomyfish", frame);
		auto end_time = high_resolution_clock::now();
    	auto duration = duration_cast<milliseconds>(end_time - start_time);
		cout << "the time process using : " << duration.count() << endl;


		char c = cv::waitKey(frame_delay);
		if (c == 27) { // ESC ÍË³ö
			break;
		}
		// reset for next frame
		results.clear();
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

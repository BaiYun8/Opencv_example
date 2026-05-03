#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <sstream>


struct DetectResult {
	int classId;
	float score;
	cv::Rect box;
};


class YOLOv5DNNDetector {
public:
	void initConfig(const std::string& onnxpath, int iw, int ih, float threshold, float nms_threshold = 0.45f);
	void detect(cv::Mat& frame, std::vector<DetectResult>& result);
private:
	int input_w = 640;
	int input_h = 640;
	cv::dnn::Net net;
	float threshold_score;  // 修复：原来声明为 int，0.25 被截断为 0
	float threshold_nms;    // NMS IOU 阈值，从 detect 中的硬编码提取出来


	// ONNX Runtime CUDA 核心成员
	Ort::Env env{ nullptr }; //运行环境
	Ort::Session session{ nullptr };//会话参数
	Ort::AllocatorWithDefaultOptions allocator; //获取输入和输出的名称
	std::vector<const char*> input_names = { "images" };
	std::vector<const char*> output_names = { "output0" };

};
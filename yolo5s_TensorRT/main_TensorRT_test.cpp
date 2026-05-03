#include "yolov5_TensorRT.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

// ====== Profiling 配置 ======
// 设为 true 开启 IProfiler 逐层分析（会拖慢推理，仅调试时使用）
static const bool ENABLE_LAYER_PROFILING = true;
// 跳过前 N 帧的预热，预热期间不统计 profiling（GPU 首次推理会有编译/缓存开销）
static const int  WARMUP_FRAMES = 10;
// 采集 N 帧后自动关闭 IProfiler 并打印结果（0 = 一直采到视频结束）
static const int  PROFILE_FRAMES = 50;

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
	detector->initConfig("F:/Test/yolo5s_TensorRT/yolov5s_int8.engine", 640, 640, 0.25f, 0.45f);
	

	std::vector<DetectResult> results;

	cv::VideoCapture capture("F:/Test/yolo/test1.mp4");
	if(!capture.isOpened())
	{
		cerr << "failed to open video" <<endl;
		return -1;
	}
	double fps = capture.get(cv::CAP_PROP_FPS);
	int frame_delay = static_cast<int>(1000.0 / fps);
	cout << "video fps: " << fps << ", frame_delay: " << frame_delay << " ms" << endl;

	cv::Mat frame;
	int frame_count = 0;
	float gpu_time_sum = 0.0f;
	int gpu_time_count = 0;

	while (true) {
		auto start_time = high_resolution_clock::now();
		if (!capture.read(frame) || frame.empty()) {
    		break;
		}
		frame_count++;

		// ====== Profiling: 预热结束后开启 IProfiler ======
		if (ENABLE_LAYER_PROFILING && frame_count == WARMUP_FRAMES + 1) {
			cout << "\n[Profiling] Warmup done (" << WARMUP_FRAMES
			     << " frames), starting layer profiling..." << endl;
			detector->enableLayerProfiling(true);
		}

		detector->detect(frame, results);

		// ====== 方案2: 收集 CUDA Events 测量的纯 GPU 推理时间 ======
		if (frame_count > WARMUP_FRAMES) {
			gpu_time_sum += detector->getLastGpuInferTimeMs();
			gpu_time_count++;
		}

		// ====== Profiling: 采集够指定帧数后关闭 IProfiler 并打印 ======
		if (ENABLE_LAYER_PROFILING && PROFILE_FRAMES > 0
		    && frame_count == WARMUP_FRAMES + PROFILE_FRAMES) {
			detector->enableLayerProfiling(false);
			detector->printLayerProfilingResults();
			cout << "[CUDA Events] Avg GPU inference time over "
			     << gpu_time_count << " frames: "
			     << gpu_time_sum / gpu_time_count << " ms" << endl;
		}

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
		// 每帧打印: 总耗时(CPU) + 纯GPU推理耗时(CUDA Events)
		cout << "frame #" << frame_count
		     << "  total: " << duration.count() << " ms"
		     << "  gpu_infer: " << detector->getLastGpuInferTimeMs() << " ms" << endl;

		char c = cv::waitKey(frame_delay);
		if (c == 27) {
			break;
		}
		results.clear();
	}

	// ====== 视频结束时输出最终统计 ======
	if (ENABLE_LAYER_PROFILING && (PROFILE_FRAMES == 0 || frame_count < WARMUP_FRAMES + PROFILE_FRAMES)) {
		// 如果 PROFILE_FRAMES=0 或者视频提前结束，在这里打印
		detector->enableLayerProfiling(false);
		detector->printLayerProfilingResults();
	}
	if (gpu_time_count > 0) {
		cout << "\n[CUDA Events] Final avg GPU inference time: "
		     << gpu_time_sum / gpu_time_count << " ms over "
		     << gpu_time_count << " frames" << endl;
	}

	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

// cl /EHsc main_TensorRT_test.cpp yolov5_TensorRT.cpp /I "F:/opencv/opencv/build/include" /I "F:\TensorRT\TensorRT-8.5.2.2.Windows10.x86_64.cuda-11.8.cudnn8.6\TensorRT-8.5.2.2\include" /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include" /link /LIBPATH:"F:/opencv/opencv/build/x64/vc16/lib" /LIBPATH:"F:\TensorRT\TensorRT-8.5.2.2.Windows10.x86_64.cuda-11.8.cudnn8.6\TensorRT-8.5.2.2\lib" /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64" opencv_world4120.lib nvinfer.lib cudart.lib

//main_TensorRT_test
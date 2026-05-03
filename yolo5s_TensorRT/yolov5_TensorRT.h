#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <map>
#include <string>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

struct DetectResult {
	int classId;
	float score;
	cv::Rect box;
};

class TRTLogger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			printf("[TRT] %s\n", msg);
		}
	}
};

// ==================== 方案1: IProfiler ====================
// TRT 每执行完一层就回调 reportLayerTime，收集每层耗时
class SimpleProfiler : public nvinfer1::IProfiler {
public:
	struct LayerRecord {
		float total_ms = 0.0f;
		int count = 0;
	};

	void reportLayerTime(const char* layerName, float ms) noexcept override;
	void printSummary() const;
	void reset();

	std::map<std::string, LayerRecord> records;
	// 保留层插入顺序，打印时按网络顺序输出
	std::vector<std::string> layer_order;
};

class YOLOv5DNNDetector {
public:
	YOLOv5DNNDetector() = default;
	~YOLOv5DNNDetector();
	void initConfig(const std::string& enginePath, int iw, int ih, float threshold, float nms_threshold = 0.45f);
	void detect(cv::Mat& frame, std::vector<DetectResult>& results);

	// ====== Profiling 控制 ======
	// 开启/关闭 IProfiler 逐层分析（有性能开销，会强制层级同步）
	void enableLayerProfiling(bool enable = true);
	// 打印 IProfiler 逐层统计结果
	void printLayerProfilingResults() const;
	// 获取上一帧纯 GPU 推理耗时（CUDA Events 测量，单位 ms）
	float getLastGpuInferTimeMs() const { return last_gpu_infer_ms; }

private:
	int input_w = 640;
	int input_h = 640;
	float threshold_score;
	float threshold_nms;

	nvinfer1::IRuntime* runtime = nullptr;
	nvinfer1::ICudaEngine* engine = nullptr;
	nvinfer1::IExecutionContext* context = nullptr;
	TRTLogger gLogger;

	void* device_buffers[2] = { nullptr, nullptr };
	float* host_output = nullptr;
	int output_size = 0;

	// ====== 方案1: IProfiler ======
	SimpleProfiler profiler;
	bool layer_profiling_enabled = false;

	// ====== 方案2: CUDA Events（纯 GPU 推理计时）======
	cudaEvent_t evt_infer_start = nullptr;
	cudaEvent_t evt_infer_end = nullptr;
	float last_gpu_infer_ms = 0.0f;
};
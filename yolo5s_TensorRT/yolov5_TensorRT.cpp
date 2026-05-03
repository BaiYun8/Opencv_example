#include "yolov5_TensorRT.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>
#include <opencv2/dnn.hpp>
#include <nvtx3/nvToolsExt.h>

using namespace cv;
using namespace std;

// ==================== SimpleProfiler 实现 ====================

void SimpleProfiler::reportLayerTime(const char* layerName, float ms) noexcept {
    std::string name(layerName);
    if (records.find(name) == records.end()) {
        layer_order.push_back(name);
    }
    records[name].total_ms += ms;
    records[name].count++;
}

void SimpleProfiler::printSummary() const {
    if (records.empty()) {
        cout << "[Profiler] No layer data collected." << endl;
        return;
    }

    // 计算总耗时
    float total_ms = 0.0f;
    int num_inferences = 0;
    for (auto& pair : records) {
        total_ms += pair.second.total_ms;
        if (num_inferences == 0) num_inferences = pair.second.count;
    }

    cout << "\n======== TensorRT Layer Profiling (" << num_inferences << " inferences) ========" << endl;
    cout << left << setw(50) << "Layer"
         << right << setw(12) << "Avg (ms)"
         << setw(12) << "Total (ms)"
         << setw(10) << "Pct (%)" << endl;
    cout << string(84, '-') << endl;

    // 按网络执行顺序输出
    for (auto& name : layer_order) {
        auto& rec = records.at(name);
        float avg_ms = rec.total_ms / rec.count;
        float pct = (rec.total_ms / total_ms) * 100.0f;

        // 层名太长时截断显示
        string display_name = name;
        if (display_name.size() > 48) {
            display_name = display_name.substr(0, 45) + "...";
        }

        cout << left << setw(50) << display_name
             << right << setw(12) << fixed << setprecision(4) << avg_ms
             << setw(12) << fixed << setprecision(4) << rec.total_ms
             << setw(9) << fixed << setprecision(2) << pct << "%" << endl;
    }

    cout << string(84, '-') << endl;
    float avg_total = total_ms / num_inferences;
    cout << left << setw(50) << "TOTAL"
         << right << setw(12) << fixed << setprecision(4) << avg_total
         << setw(12) << fixed << setprecision(4) << total_ms << endl;
    cout << "================================================================\n" << endl;
}

void SimpleProfiler::reset() {
    records.clear();
    layer_order.clear();
}

// ==================== YOLOv5DNNDetector 实现 ====================

YOLOv5DNNDetector::~YOLOv5DNNDetector() {
    if (context) context->destroy();
    if (engine) engine->destroy();
    if (runtime) runtime->destroy();
    if (device_buffers[0]) cudaFree(device_buffers[0]);
    if (device_buffers[1]) cudaFree(device_buffers[1]);
    if (host_output) delete[] host_output;
    // 方案2: 销毁 CUDA Events
    if (evt_infer_start) cudaEventDestroy(evt_infer_start);
    if (evt_infer_end) cudaEventDestroy(evt_infer_end);
}

void YOLOv5DNNDetector::initConfig(const string& enginePath, int iw, int ih, float threshold, float nms_threshold) {
    this->input_w = iw;
    this->input_h = ih;
    this->threshold_score = threshold;
    this->threshold_nms = nms_threshold;

    ifstream file(enginePath, ios::binary);
    if (!file.good()) {
        cerr << "Error reading engine: " << enginePath << endl;
        return;
    }
    file.seekg(0, ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    char* data = new char[size];
    file.read(data, size);
    file.close();

    runtime = nvinfer1::createInferRuntime(gLogger);
    //反序列化引擎，把 .engine 文件变成可运行的模型。
    engine = runtime->deserializeCudaEngine(data, size, nullptr);
    if(!engine)
    {
        cerr << "Failed to deserialize engine!" << endl;
        return;
    }
    //创建执行上下文
    context = engine->createExecutionContext();
    delete[] data;

    // 分配 GPU 内存（输入）
    cudaMalloc(&device_buffers[0], 1 * 3 * input_h * input_w * sizeof(float));

    // 输出 buffer 分配
    nvinfer1::Dims out_dims = engine->getBindingDimensions(1);
    output_size = 1;
    for (int i = 0; i < out_dims.nbDims; i++) {
        output_size *= out_dims.d[i];
    }
    cudaMalloc(&device_buffers[1], output_size * sizeof(float));
    host_output = new float[output_size];

    // 方案2: 创建 CUDA Events 用于纯 GPU 推理计时
    cudaEventCreate(&evt_infer_start);
    cudaEventCreate(&evt_infer_end);

    cout << "TensorRT Engine loaded: " << enginePath << endl;
}

// ====== Profiling 控制方法 ======

// 空 Profiler：关闭逐层分析时挂这个，回调什么也不做（TRT 8.6 不允许传 nullptr）
static class NoOpProfiler : public nvinfer1::IProfiler {
    void reportLayerTime(const char*, float) noexcept override {}
} s_noopProfiler;

void YOLOv5DNNDetector::enableLayerProfiling(bool enable) {
    layer_profiling_enabled = enable;
    if (context) {
        context->setProfiler(enable ? static_cast<nvinfer1::IProfiler*>(&profiler)
                                    : static_cast<nvinfer1::IProfiler*>(&s_noopProfiler));
    }
    if (enable) {
        profiler.reset();
    }
    cout << "[Profiling] Layer profiling " << (enable ? "ENABLED" : "DISABLED") << endl;
}

void YOLOv5DNNDetector::printLayerProfilingResults() const {
    profiler.printSummary();
}

// ====== 检测主函数 ======

void YOLOv5DNNDetector::detect(Mat& frame, vector<DetectResult>& results) {
    // 方案3: NVTX 整体标记 — nsys 时间轴上显示为一个完整的 "Detect" 区间
    nvtxRangePushA("Detect");

    results.clear();
    int w = frame.cols;
    int h = frame.rows;
    int _max = max(h, w);

    // 方案3: NVTX 预处理标记
    nvtxRangePushA("Preprocess");
    Mat image = Mat::zeros(Size(_max, _max), CV_8UC3);
    Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    float x_factor = image.cols / (float)input_w;
    float y_factor = image.rows / (float)input_h;

    Mat blob = dnn::blobFromImage(image, 1 / 255.0, Size(input_w, input_h), Scalar(0, 0, 0), true, false);
    nvtxRangePop();  // Preprocess

    // 方案3: NVTX H2D 拷贝标记
    nvtxRangePushA("H2D Copy");
    cudaMemcpy(device_buffers[0], blob.ptr<float>(),
        1 * 3 * input_h * input_w * sizeof(float), cudaMemcpyHostToDevice);
    nvtxRangePop();  // H2D Copy

    // 方案2: CUDA Events 记录推理开始时刻
    cudaEventRecord(evt_infer_start);

    // 方案3: NVTX 推理标记
    nvtxRangePushA("Inference (executeV2)");
    context->executeV2(device_buffers);
    nvtxRangePop();  // Inference

    // 方案2: CUDA Events 记录推理结束时刻
    cudaEventRecord(evt_infer_end);
    cudaEventSynchronize(evt_infer_end);
    cudaEventElapsedTime(&last_gpu_infer_ms, evt_infer_start, evt_infer_end);

    // 方案3: NVTX D2H 拷贝标记
    nvtxRangePushA("D2H Copy");
    cudaMemcpy(host_output, device_buffers[1],
        output_size * sizeof(float), cudaMemcpyDeviceToHost);
    nvtxRangePop();  // D2H Copy

    // 方案3: NVTX 后处理标记
    nvtxRangePushA("Postprocess (NMS + Draw)");

    int num_boxes = engine->getBindingDimensions(1).d[1];
    int num_elements = engine->getBindingDimensions(1).d[2];
    int num_classes = num_elements - 5;

    vector<Rect> boxes;
    vector<int> classIds;
    vector<float> confidences;

    for (int i = 0; i < num_boxes; i++) {
        float* ptr = host_output + i * num_elements;
        float confidence = ptr[4];
        if (confidence < threshold_score) continue;

        Mat class_scores(1, num_classes, CV_32F, ptr + 5);
        Point classIdPoint;
        double score;
        minMaxLoc(class_scores, 0, &score, 0, &classIdPoint);

        if (score > threshold_score) {
            float cx = ptr[0];
            float cy = ptr[1];
            float ow = ptr[2];
            float oh = ptr[3];

            int x = (int)((cx - 0.5f * ow) * x_factor);
            int y = (int)((cy - 0.5f * oh) * y_factor);
            int width = (int)(ow * x_factor);
            int height = (int)(oh * y_factor);

            Rect box;
            box.x = max(0, x);
            box.y = max(0, y);
            box.width = min(width, frame.cols - box.x);
            box.height = min(height, frame.rows - box.y);

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back((float)score);
        }
    }

    vector<int> indexes;
    dnn::NMSBoxes(boxes, confidences, threshold_score, threshold_nms, indexes);

    for (int index : indexes) {
        DetectResult dr;
        dr.box = boxes[index];
        dr.classId = classIds[index];
        dr.score = confidences[index];
        results.push_back(dr);
        rectangle(frame, dr.box, Scalar(0, 0, 255), 2);
    }

    putText(frame, "TensorRT GPU", Point(20, 40), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0), 2);

    nvtxRangePop();  // Postprocess
    nvtxRangePop();  // Detect
}
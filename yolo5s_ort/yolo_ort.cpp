#include "yolo_ort.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;
using namespace std::chrono;

void YOLOv5DNNDetector::initConfig(const string& onnxpath, int iw, int ih, float threshold, float nms_threshold)
{
    this->input_w = iw;
    this->input_h = ih;
    this->threshold_score = threshold;
    this->threshold_nms = nms_threshold;

    this->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5_CUDA");

    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
    session_options.EnableProfiling(L"ort_profile");

    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    std::wstring wpath(onnxpath.begin(), onnxpath.end());
    this->session = Ort::Session(this->env, wpath.c_str(), session_options);
}

void YOLOv5DNNDetector::detect(Mat& frame, vector<DetectResult>& results)
{
    const auto detect_start = high_resolution_clock::now();

    results.clear();
    const int w = frame.cols;
    const int h = frame.rows;
    const int max_side = max(h, w);

    Mat image = Mat::zeros(Size(max_side, max_side), CV_8UC3);
    Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    const float x_factor = max_side / 640.0f;
    const float y_factor = max_side / 640.0f;

    Mat blob;
    resize(image, blob, Size(input_w, input_h));
    blob.convertTo(blob, CV_32F, 1.0 / 255.0);

    vector<int64_t> input_shape = { 1, 3, input_h, input_w };
    const size_t tensor_size = static_cast<size_t>(3) * input_h * input_w;
    vector<float> input_data(tensor_size);

    vector<Mat> channels(3);
    split(blob, channels);
    for (int c = 0; c < 3; ++c) {
        memcpy(input_data.data() + c * input_h * input_w,
            channels[c].data,
            input_h * input_w * sizeof(float));
    }

    const auto preprocess_end = high_resolution_clock::now();

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        tensor_size,
        input_shape.data(),
        input_shape.size());

    const auto inference_start = high_resolution_clock::now();
    auto output_tensors = session.Run(
        Ort::RunOptions{ nullptr },
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1);
    const auto inference_end = high_resolution_clock::now();

    float* preds = output_tensors[0].GetTensorMutableData<float>();
    auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_boxes = static_cast<int>(out_shape[1]);
    const int num_classes = static_cast<int>(out_shape[2]) - 5;

    vector<Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;

    for (int i = 0; i < num_boxes; ++i) {
        const float confidence = preds[i * (num_classes + 5) + 4];
        if (confidence < threshold_score) {
            continue;
        }

        Mat scores(1, num_classes, CV_32F, preds + i * (num_classes + 5) + 5);
        Point class_id_point;
        double max_score = 0.0;
        minMaxLoc(scores, nullptr, &max_score, nullptr, &class_id_point);

        if (max_score < threshold_score) {
            continue;
        }

        const float cx = preds[i * (num_classes + 5) + 0];
        const float cy = preds[i * (num_classes + 5) + 1];
        const float box_w = preds[i * (num_classes + 5) + 2];
        const float box_h = preds[i * (num_classes + 5) + 3];

        const int x = static_cast<int>((cx - 0.5f * box_w) * x_factor);
        const int y = static_cast<int>((cy - 0.5f * box_h) * y_factor);
        const int width = static_cast<int>(box_w * x_factor);
        const int height = static_cast<int>(box_h * y_factor);

        Rect box;
        box.x = max(0, x);
        box.y = max(0, y);
        box.width = min(width, frame.cols - box.x);
        box.height = min(height, frame.rows - box.y);

        boxes.push_back(box);
        class_ids.push_back(class_id_point.x);
        confidences.push_back(static_cast<float>(max_score));
    }

    vector<int> indices;
    dnn::NMSBoxes(boxes, confidences, threshold_score, threshold_nms, indices);

    for (size_t i = 0; i < indices.size(); ++i) {
        const int idx = indices[i];
        DetectResult res;
        res.classId = class_ids[idx];
        res.score = confidences[idx];
        res.box = boxes[idx];
        results.push_back(res);

        rectangle(frame, res.box, Scalar(0, 0, 255), 2);
        rectangle(frame,
            Point(res.box.x, res.box.y - 20),
            Point(res.box.x + res.box.width, res.box.y),
            Scalar(0, 255, 255), -1);
    }

    putText(frame, "YOLOv5 ONNX Runtime GPU", Point(20, 40),
        FONT_HERSHEY_PLAIN, 2, Scalar(255, 0, 0), 2);

    const auto postprocess_end = high_resolution_clock::now();

    const auto preprocess_ms = duration_cast<milliseconds>(preprocess_end - detect_start).count();
    const auto inference_ms = duration_cast<milliseconds>(inference_end - inference_start).count();
    const auto postprocess_ms = duration_cast<milliseconds>(postprocess_end - inference_end).count();
    const auto total_ms = duration_cast<milliseconds>(postprocess_end - detect_start).count();

    cout << "[detect profiling] preprocess=" << preprocess_ms
        << " ms, inference=" << inference_ms
        << " ms, postprocess=" << postprocess_ms
        << " ms, total=" << total_ms << " ms" << endl;
}

//cl /EHsc main_ort_test.cpp yolo_ort.cpp /I "F:/opencv/opencv/build/include" /I "F:\onnxruntime\onnxruntime-win-x64-gpu-1.16.3\onnxruntime-win-x64-gpu-1.16.3\include" /link /LIBPATH:"F:/opencv/opencv/build/x64/vc16/lib" /LIBPATH:"F:\onnxruntime\onnxruntime-win-x64-gpu-1.16.3\onnxruntime-win-x64-gpu-1.16.3\lib" opencv_world4120.lib onnxruntime.lib

//main_ort_test

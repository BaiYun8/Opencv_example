本目录主要围绕 YOLOv5s 目标检测模型的 C++ 部署展开，使用 OpenCV 进行视频读取、图像预处理和检测结果绘制，使用 ONNX Runtime 加载并运行 `yolov5s.onnx` 模型，同时启用了 CUDA 执行提供器，用于验证 GPU 推理能力。

## 一、整体功能
该项目实现了一个基于 YOLOv5s 的视频目标检测 demo。程序读取本地视频文件，对每一帧图像进行预处理，然后送入 ONNX Runtime 执行模型推理，解析 YOLOv5s 输出结果，经过置信度过滤和 NMS 非极大值抑制后，在视频画面上绘制检测框和类别名称。
整体流程如下：
1. 加载 YOLOv5s ONNX 模型；
2. 初始化 ONNX Runtime 推理环境；
3. 启用 CUDA Provider 进行 GPU 推理；
4. 使用 OpenCV 读取视频帧；
5. 对每帧图像进行 resize、归一化、通道转换等预处理；
6. 构造 ONNX Runtime 输入 Tensor；
7. 执行模型推理；
8. 解析模型输出；
9. 根据置信度筛选检测结果；
10. 使用 NMS 去除重复框；
11. 在原始图像上绘制目标框、类别名称和检测信息；
12. 输出每帧处理耗时，用于性能分析。

## 二、主要文件说明
## 1. `main_ort_test.cpp`
该文件是程序入口，主要负责视频读取、检测器初始化、逐帧推理和结果显示。
## 2. `yolo_ort.cpp`
该文件完成了 ONNX Runtime 初始化、图像预处理、模型推理、结果解析和后处理。

## 三、模型与依赖文件
目录中包含以下关键依赖：
1. `yolov5s.onnx`：用于 ONNX Runtime 推理的 YOLOv5s 模型；
2. `yolov5s.pt`：PyTorch 格式的 YOLOv5s 模型文件；
3. `onnxruntime.dll`：ONNX Runtime 运行库；
4. `onnxruntime_providers_cuda.dll`：CUDA 推理后端；
5. `onnxruntime_providers_tensorrt.dll`：TensorRT 后端相关库；
6. `opencv_world4120.dll`：OpenCV 动态库；
7. `ort_profile_*.json`：ONNX Runtime profiling 生成的性能分析文件。





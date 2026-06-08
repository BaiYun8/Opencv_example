本目录主要围绕 YOLOv5s 目标检测模型的 TensorRT C++ 部署展开。相比 ONNX Runtime 版本，本部分代码重点验证 TensorRT Engine 的加载、GPU 推理、显存管理、后处理解析以及性能 profiling。整体目标是将 YOLOv5s 模型转换为 TensorRT engine，并在 C++ 程序中实现视频目标检测。

# 一、整体功能
该项目实现了一个基于 TensorRT 的 YOLOv5s 视频目标检测 demo。程序读取本地视频文件，对每一帧图像进行预处理，然后将输入数据拷贝到 GPU，调用 TensorRT 执行推理，再把输出结果拷贝回 CPU 进行后处理，最终在视频画面上绘制检测框和类别名称。
整体流程如下：
1. 加载 TensorRT `.engine` 模型文件；
2. 反序列化 TensorRT Engine；
3. 创建 TensorRT Runtime 和 ExecutionContext；
4. 为输入和输出分配 GPU 显存；
5. 使用 OpenCV 读取视频帧；
6. 对每帧图像进行 resize、归一化和 NCHW 格式转换；
7. 将输入 Tensor 从 CPU 拷贝到 GPU；
8. 调用 `executeV2` 执行 TensorRT 推理；
9. 将输出结果从 GPU 拷贝回 CPU；
10. 解析 YOLOv5 输出结果；
11. 根据置信度过滤候选框；
12. 使用 NMS 去除重复检测框；
13. 在原图上绘制检测框和类别信息；
14. 统计 CPU 总耗时、GPU 推理耗时和逐层耗时。

# 二、主要文件说明
## 1. `main_TensorRT_test.cpp`
该文件是程序入口，主要负责检测器初始化、视频读取、逐帧检测、结果显示和性能统计。
该文件相当于整个 TensorRT 推理 demo 的调度层，负责把检测器应用到视频流上，并观察运行性能。
## 2. `yolov5_TensorRT.h`
该文件定义了检测结果结构体、TensorRT 日志类、逐层 profiler，以及 YOLOv5 TensorRT 检测器类。
该头文件把 TensorRT 推理相关的运行时对象、显存 buffer、profiling 工具和检测接口进行了封装。

## 3. `yolov5_TensorRT.cpp`
该文件是 TensorRT 推理的核心实现部分，完成了 engine 加载、显存分配、预处理、GPU 推理、后处理和性能分析。
这部分体现了 TensorRT C++ 部署中的核心流程：不是直接加载 ONNX 模型推理，而是加载已经构建好的 engine 文件运行。

# 三、性能分析设计
这个目录相比 ONNX Runtime 版本，多了比较完整的 TensorRT 性能分析逻辑。
## 1. CUDA Event 计时
代码中使用 `cudaEventRecord` 和 `cudaEventElapsedTime` 测量纯 GPU 推理耗时，也就是 `executeV2` 的执行时间。
## 2. TensorRT IProfiler 逐层分析
代码实现了 `SimpleProfiler`，可以统计 TensorRT 每一层的执行时间
这个设计可以帮助分析模型中哪些层耗时最多，为后续优化提供依据。
## 3. NVTX 标记
代码中使用了 NVTX 标记：
这些标记可以配合 NVIDIA Nsight Systems 等工具，在时间线上观察各阶段耗时分布，对性能分析非常有帮助。

# 四、模型与依赖文件
目录中包含多个模型和运行依赖：
1. `yolov5s.onnx`：YOLOv5s ONNX 模型；
2. `yolov5s.pt`：PyTorch 格式模型；
3. `yolov5s_fp32.engine`：FP32 精度 TensorRT engine；
4. `yolov5s_fp16.engine`：FP16 精度 TensorRT engine；
5. `yolov5s_int8.engine`：INT8 精度 TensorRT engine；
6. `yolov5s_trtexec.engine`：通过 trtexec 工具生成的 engine；
7. `yolov5s_calibration.cache`：INT8 量化校准缓存；
8. `opencv_world4120.dll`：OpenCV 动态库；
9. `zlibwapi.dll`：TensorRT 相关依赖；
10. `onnxruntime*.dll`：目录中也保留了 ONNX Runtime 相关库，但当前 C++ TensorRT 主流程主要使用 TensorRT 和 CUDA。
该目录不仅验证了 TensorRT 推理，还包含了 FP32、FP16、INT8 多种精度模型的部署尝试。



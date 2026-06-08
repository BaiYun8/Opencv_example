本链接主要关注一下三个文件中的内容：

CPP            ：文件中主要是涉及到OpenCV 在 C++ 环境下的基础图像处理。

yolo5s_ort     ：文件中围绕 YOLOv5s 目标检测模型的 C++ 部署展开，使用 OpenCV 进行视频读取、图像预处理和检测结果绘制，使用 ONNX Runtime 加载并运行 `yolov5s.onnx` 模型。

yolo5s_TensorRT：主要围绕 YOLOv5s 目标检测模型的 TensorRT C++ 部署展开。
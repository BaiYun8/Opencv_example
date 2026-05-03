"""
YOLOv5 ONNX -> TensorRT INT8 Engine 构建脚本(PTQ 量化)
================================================
核心组件:
  1. preprocess_exact        - 图像预处理函数
  2. Yolov5EntropyCalibrator  - INT8 校准器,继承 trt.IInt8EntropyCalibrator2
  3. build_int8_engine        - 构建 INT8 engine 的主函数

流程(和 FP32/FP16 类似,只是 config 多了 INT8 flag 和 Calibrator):
  1. 创建 Logger
  2. 创建 Builder 和 Network
  3. 创建 ONNX Parser,解析 onnx 文件
  4. 创建 BuilderConfig(INT8 flag + Calibrator)
  5. 构建 engine 并序列化保存

注意:
  - 第一次构建会慢(3-10 分钟),因为要跑完整校准
  - 校准完成后生成 calibration.cache,下次构建秒级完成
  - 如果改了校准集或模型,要删除 cache 文件让它重新校准
"""

import os
import time
import glob
import numpy as np
import cv2
import tensorrt as trt
from cuda import cudart

# ===================== 路径配置 =====================
ONNX_PATH        = r"F:\Test\yolo\yolov5s.onnx"                           # 输入 onnx
OUTPUT_DIR       = r"F:\Test\yolo5s_TensorRT"                             # engine 输出目录
CALIB_DATA_DIR   = r"F:\Test\yolo5s_TensorRT\calib_data"                  # 校准集目录(你 500 张图所在地)
ENGINE_PATH      = os.path.join(OUTPUT_DIR, "yolov5s_int8.engine")        # 最终 engine 文件
CALIB_CACHE_PATH = os.path.join(OUTPUT_DIR, "yolov5s_calibration.cache")  # 校准缓存文件

# 网络输入参数(和 yolov5s.onnx 的输入一致)
INPUT_NAME  = "images"
INPUT_SHAPE = (1, 3, 640, 640)   # (batch, channel, height, width)

# Workspace 大小(字节),4GB
WORKSPACE_SIZE = 4 * (1 << 30)


# ========================================================================
# 预处理函数(和你之前 ORT 代码里的 preprocess_exact 一致)
# 校准和推理必须用同一个预处理,不然 INT8 精度会很差
# ========================================================================
def preprocess_exact(img_path: str) -> np.ndarray:
    """
    读图并做预处理,返回 (1, 3, 640, 640) 的 float32 数组
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    # 用正方形黑色画布 letterbox(和 ORT 代码保持一致)
    _max = max(h, w)
    image = np.zeros((_max, _max, 3), dtype=np.uint8)
    image[0:h, 0:w] = img

    # Resize 到 640x640
    blob = cv2.resize(image, (640, 640))

    # 归一化 [0, 255] -> [0, 1]
    blob = blob.astype(np.float32) / 255.0

    # HWC -> CHW,添加 batch 维度
    channels = cv2.split(blob)
    input_data = np.zeros((1, 3, 640, 640), dtype=np.float32)
    for c in range(3):
        input_data[0, c] = channels[c]

    # 确保内存连续(TRT 要求)
    return np.ascontiguousarray(input_data)


# ========================================================================
# Calibrator 类
# 继承 IInt8EntropyCalibrator2,实现 4 个必须的方法
# ========================================================================
class Yolov5EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir: str, cache_path: str, input_shape=(1, 3, 640, 640)):
        # 必须调用父类构造函数
        super().__init__()

        self.cache_path = cache_path
        self.input_shape = input_shape
        self.batch_size = input_shape[0]

        # 扫描校准集目录下的所有 jpg 文件
        self.image_paths = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"校准集目录为空: {calib_dir}")
        print(f"[Calibrator] 加载 {len(self.image_paths)} 张校准图")

        # 当前读取的索引
        self.current_index = 0

        # 在 GPU 上预分配一块显存,大小 = 一个 batch 的字节数
        # float32 是 4 字节,所以总字节数 = 1*3*640*640*4 = 4915200 字节 ≈ 4.7 MB
        self.nbytes = int(np.prod(input_shape) * np.float32().itemsize)
        err, self.device_input = cudart.cudaMalloc(self.nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc 失败: {err}")
        print(f"[Calibrator] 已分配 GPU 显存 {self.nbytes / 1024:.1f} KB")

    def get_batch_size(self):
        """
        返回 batch 大小,TRT 会调用这个方法
        """
        return self.batch_size

    def get_batch(self, names):
        """
        核心方法:TRT 反复调用这个,每次返回一批预处理好的数据
        当所有校准图都用完时返回 None,告诉 TRT 校准结束

        参数:
            names: list[str],网络输入张量的名字,对我们来说就是 ['images']

        返回:
            list[int] 或 None:包含 GPU 显存地址的列表
        """
        # 如果所有图都用完了,返回 None 结束校准
        if self.current_index >= len(self.image_paths):
            return None

        # 读当前这张图并预处理
        img_path = self.image_paths[self.current_index]
        batch_data = preprocess_exact(img_path)

        if batch_data is None:
            # 图读取失败,跳过这张,继续下一张
            print(f"[Calibrator] 警告:图片读取失败,跳过 {img_path}")
            self.current_index += 1
            return self.get_batch(names)

        # 把 numpy 数组从 CPU 拷贝到 GPU 显存
        err, = cudart.cudaMemcpy(
            self.device_input,
            batch_data.ctypes.data,
            self.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpy 失败: {err}")

        # 打印进度(每 50 张打印一次,避免刷屏)
        self.current_index += 1
        if self.current_index % 50 == 0 or self.current_index == len(self.image_paths):
            print(f"[Calibrator] 进度 {self.current_index}/{len(self.image_paths)}")

        # 返回 GPU 指针列表,TRT 会从这里读数据
        return [int(self.device_input)]

    def read_calibration_cache(self):
        """
        如果缓存文件存在就读取返回,不存在返回 None
        返回缓存后 TRT 就不会再跑校准了,直接用缓存里的量化参数
        """
        if os.path.exists(self.cache_path):
            print(f"[Calibrator] 读取缓存: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                return f.read()
        print("[Calibrator] 未找到缓存,将执行完整校准")
        return None

    def write_calibration_cache(self, cache):
        """
        校准结束后 TRT 会把结果传给这个方法,我们写到文件里
        下次构建同一个模型直接读这个缓存,秒级完成
        """
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            f.write(cache)
        print(f"[Calibrator] 缓存已保存: {self.cache_path}")

    def __del__(self):
        """
        析构时释放 GPU 显存,避免显存泄漏
        """
        if hasattr(self, "device_input") and self.device_input is not None:
            cudart.cudaFree(self.device_input)


# ========================================================================
# 构建 INT8 engine 的主函数
# ========================================================================
def build_int8_engine(onnx_path: str, engine_path: str,
                     calib_dir: str, cache_path: str) -> bool:
    """
    构建 INT8 精度的 TensorRT Engine
    """

    # ---------- 第一步:创建 Logger ----------
    logger = trt.Logger(trt.Logger.WARNING)
    print("[1/5] Logger 创建完成")

    # ---------- 第二步:创建 Builder 和 Network ----------
    builder = trt.Builder(logger)
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch_flag)
    print("[2/5] Builder 和 Network 创建完成")

    # ---------- 第三步:创建 Parser,解析 onnx 文件 ----------
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnx_path):
        print(f"[错误] 找不到 ONNX 文件: {onnx_path}")
        return False

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[错误] ONNX 解析失败:")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return False

    print(f"[3/5] ONNX 解析完成: {onnx_path}")

    # 打印网络输入输出,确认和预期一致
    print("    === 网络输入 ===")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"      {tensor.name} : shape={tensor.shape}, dtype={tensor.dtype}")
    print("    === 网络输出 ===")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"      {tensor.name} : shape={tensor.shape}, dtype={tensor.dtype}")

    # ---------- 第四步:创建 BuilderConfig,开启 INT8 + 挂 Calibrator ----------
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_SIZE)

    # 检查硬件是否支持 INT8(你的 3060 肯定支持)
    if not builder.platform_has_fast_int8:
        print("[警告] 当前平台不支持快速 INT8")

    # 开启 INT8 flag
    config.set_flag(trt.BuilderFlag.INT8)

    # 创建 Calibrator 实例并挂到 config 上
    calibrator = Yolov5EntropyCalibrator(
        calib_dir=calib_dir,
        cache_path=cache_path,
        input_shape=INPUT_SHAPE
    )
    config.int8_calibrator = calibrator

    print(f"[4/5] BuilderConfig 设置完成 (Workspace={WORKSPACE_SIZE / (1<<30):.1f} GB, 精度=INT8)")

    # ---------- 第五步:构建 engine 并序列化保存 ----------
    print(f"[5/5] 开始构建 INT8 engine")
    print("      第一次构建会跑完整校准(500张图),预计需要 3-10 分钟,请耐心等待...")
    start_time = time.time()

    serialized_engine = builder.build_serialized_network(network, config)

    elapsed = time.time() - start_time

    if serialized_engine is None:
        print("[错误] Engine 构建失败")
        return False

    # 确保输出目录存在
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    # 写入磁盘
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"\n[OK] INT8 Engine 构建成功!")
    print(f"   保存路径:  {engine_path}")
    print(f"   文件大小:  {size_mb:.2f} MB")
    print(f"   构建耗时:  {elapsed:.2f} 秒 ({elapsed/60:.1f} 分钟)")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv5 TensorRT Engine 构建(INT8 PTQ)")
    print("=" * 60)
    print(f"TensorRT 版本: {trt.__version__}")
    print(f"ONNX 输入:    {ONNX_PATH}")
    print(f"校准集目录:   {CALIB_DATA_DIR}")
    print(f"Engine 输出:  {ENGINE_PATH}")
    print(f"校准缓存:     {CALIB_CACHE_PATH}")
    print("=" * 60)

    success = build_int8_engine(
        onnx_path=ONNX_PATH,
        engine_path=ENGINE_PATH,
        calib_dir=CALIB_DATA_DIR,
        cache_path=CALIB_CACHE_PATH
    )
    exit(0 if success else 1)
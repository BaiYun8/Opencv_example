"""
YOLOv5 ONNX -> TensorRT Engine 构建脚本(支持 FP32 / FP16)
================================================
流程(五步):
  1. 创建 Logger
  2. 创建 Builder 和 Network
  3. 创建 ONNX Parser,解析 onnx 文件
  4. 创建 BuilderConfig,设置 workspace 和精度
  5. 构建 engine 并序列化保存

相对于 FP32 版本的变化:
  - 新增 precision 参数,支持 "fp32" / "fp16"
  - 第四步根据精度设置对应的 flag
  - 输出文件名根据精度自动区分
"""

import os
import time
import tensorrt as trt

# ===================== 路径配置 =====================
ONNX_PATH  = r"F:\Test\yolo\yolov5s.onnx"        # 输入 onnx 路径
OUTPUT_DIR = r"F:\Test\yolo5s_TensorRT"          # engine 输出目录

# Workspace 大小(字节),4GB
WORKSPACE_SIZE = 4 * (1 << 30)


def build_engine(onnx_path: str, output_dir: str, precision: str = "fp32") -> bool:
    """
    将 ONNX 模型构建成 TensorRT Engine 并保存到磁盘
    
    Args:
        onnx_path:  输入的 onnx 文件路径
        output_dir: engine 输出目录
        precision:  精度模式,可选 "fp32" 或 "fp16"
    """

    # 参数合法性检查
    precision = precision.lower()
    if precision not in ("fp32", "fp16"):
        print(f"[错误] 不支持的精度: {precision},只支持 fp32 / fp16")
        return False

    # 输出文件名根据精度自动区分
    engine_path = os.path.join(output_dir, f"yolov5s_{precision}.engine")

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

    # 打印网络的输入输出,确认和 Netron 里看到的一致
    print("    === 网络输入 ===")
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        print(f"      {tensor.name} : shape={tensor.shape}, dtype={tensor.dtype}")

    print("    === 网络输出 ===")
    for i in range(network.num_outputs):
        tensor = network.get_output(i)
        print(f"      {tensor.name} : shape={tensor.shape}, dtype={tensor.dtype}")

    # ---------- 第四步:创建 BuilderConfig ----------
    config = builder.create_builder_config()

    # 设置 workspace 大小
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_SIZE)

    # 根据精度设置 flag
    if precision == "fp16":
        # 先检查硬件是否支持快速 FP16(你的 3060 是支持的)
        if not builder.platform_has_fast_fp16:
            print("[警告] 当前平台不支持快速 FP16,但仍会尝试启用")
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"[4/5] BuilderConfig 设置完成 (Workspace={WORKSPACE_SIZE / (1<<30):.1f} GB, 精度=FP16)")
    else:
        # FP32 是默认精度,不需要任何 flag
        print(f"[4/5] BuilderConfig 设置完成 (Workspace={WORKSPACE_SIZE / (1<<30):.1f} GB, 精度=FP32)")

    # ---------- 第五步:构建 engine 并序列化保存 ----------
    print(f"[5/5] 开始构建 {precision.upper()} engine,请耐心等待...")
    start_time = time.time()

    serialized_engine = builder.build_serialized_network(network, config)

    elapsed = time.time() - start_time

    if serialized_engine is None:
        print("[错误] Engine 构建失败")
        return False

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 写入磁盘
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"\n✅ {precision.upper()} Engine 构建成功!")
    print(f"   保存路径:  {engine_path}")
    print(f"   文件大小:  {size_mb:.2f} MB")
    print(f"   构建耗时:  {elapsed:.2f} 秒")
    return True


if __name__ == "__main__":
    import sys

    # 支持命令行参数切换精度,默认 fp16
    # 用法: python build_engine.py fp32
    #       python build_engine.py fp16
    precision = sys.argv[1] if len(sys.argv) > 1 else "fp16"

    print("=" * 60)
    print(f"YOLOv5 TensorRT Engine 构建({precision.upper()})")
    print("=" * 60)
    print(f"TensorRT 版本: {trt.__version__}")
    print(f"ONNX 输入:    {ONNX_PATH}")
    print(f"输出目录:     {OUTPUT_DIR}")
    print("=" * 60)

    success = build_engine(ONNX_PATH, OUTPUT_DIR, precision)
    sys.exit(0 if success else 1)


# python onnx_cov_fp32.py fp16

#python onnx_cov_fp32.py fp32
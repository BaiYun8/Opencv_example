# Makefile for yolov5 TensorRT demo (MSVC / nmake)
# Usage:
#   nmake                 # build
#   nmake clean           # remove build artifacts
#   nmake run             # build and run
#
# 如果三方库路径变了，改下面这几个变量即可。

# ---------- Toolchain ----------
CXX      = cl
LINK     = link

# ---------- Third-party roots ----------
OPENCV_DIR   = F:/opencv/opencv/build
TENSORRT_DIR = F:/TensorRT/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6
CUDA_DIR     = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8

# ---------- Include / Lib paths ----------
INCLUDES = /I "$(OPENCV_DIR)/include" \
           /I "$(TENSORRT_DIR)/include" \
           /I "$(CUDA_DIR)/include"

LIBPATHS = /LIBPATH:"$(OPENCV_DIR)/x64/vc16/lib" \
           /LIBPATH:"$(TENSORRT_DIR)/lib" \
           /LIBPATH:"$(CUDA_DIR)/lib/x64"

LIBS = opencv_world4120.lib nvinfer.lib cudart.lib

# ---------- Compiler / Linker flags ----------
CXXFLAGS = /nologo /EHsc /std:c++17 /O2 /MD /W3 /utf-8 $(INCLUDES)
LDFLAGS  = /nologo $(LIBPATHS)

# ---------- Targets ----------
TARGET = main_TensorRT_test.exe

OBJS = main_TensorRT_test.obj \
       yolov5_TensorRT.obj

all: $(TARGET)

$(TARGET): $(OBJS)
	$(LINK) $(LDFLAGS) /OUT:$@ $(OBJS) $(LIBS)

main_TensorRT_test.obj: main_TensorRT_test.cpp yolov5_TensorRT.h
	$(CXX) $(CXXFLAGS) /c main_TensorRT_test.cpp /Fo$@

yolov5_TensorRT.obj: yolov5_TensorRT.cpp yolov5_TensorRT.h
	$(CXX) $(CXXFLAGS) /c yolov5_TensorRT.cpp /Fo$@

run: $(TARGET)
	$(TARGET)

clean:
	-del /Q *.obj $(TARGET) 2>nul

.PHONY: all run clean

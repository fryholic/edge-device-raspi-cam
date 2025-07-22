# Makefile for Zero Copy OpenVINO Demo

# OpenVINO 환경 변수 설정
export LD_LIBRARY_PATH := /home/lee/openvino/build/install/usr/local/runtime/lib/aarch64:$(LD_LIBRARY_PATH)

# 컴파일러 설정
CXX = g++
CXXFLAGS = -std=c++17 -g -O2 -Wall

# --- 새로운 RTSP 데모 설정 ---
TARGET_RTSP_DEMO = zero_copy_rtsp_demo
SOURCE_RTSP_DEMO = zero_copy_rtsp_demo.cpp

# --- 새로운 RTSP 데모 설정 ---
TARGET_RTSP_DEMO_V4L2CONVERT = zero_copy_rtsp_demo_convert
SOURCE_RTSP_DEMO_V4L2CONVERT = zero_copy_rtsp_demo_convert.cpp

# --- 라이브러리 설정 ---
# OpenVINO 설정
OPENVINO_INCLUDE = -I/usr/local/runtime/include
OPENVINO_LIBS = -L/usr/local/runtime/lib/aarch64 -lopenvino

# OpenCV 설정
OPENCV_INCLUDE = -I/usr/local/include/opencv4
OPENCV_LIBS_HEADLESS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn

# libcamera 설정
LIBCAMERA_INCLUDE = -I/usr/include/libcamera
LIBCAMERA_LIBS = -lcamera -lcamera-base

# SQLite 설정
SQLITE_LIBS = -lSQLiteCpp -lsqlite3

# GStreamer 설정 (RTSP 스트리밍 버전용)
GSTREAMER_INCLUDE = `pkg-config --cflags gstreamer-1.0 gstreamer-rtsp-server-1.0 gstreamer-app-1.0`
GSTREAMER_LIBS = `pkg-config --libs gstreamer-1.0 gstreamer-rtsp-server-1.0 gstreamer-app-1.0`

# Eigen 헤더 경로 추가
EIGEN_INCLUDE = -I/usr/include/eigen3

# --- 전체 플래그 설정 ---
INCLUDES = $(OPENVINO_INCLUDE) $(OPENCV_INCLUDE) $(LIBCAMERA_INCLUDE) $(EIGEN_INCLUDE) $(GSTREAMER_INCLUDE)

# 새로운 RTSP 데모용 라이브러리 (OpenCV, OpenVINO 제외)
LIBS_RTSP_DEMO = $(LIBCAMERA_LIBS) $(GSTREAMER_LIBS) -lpthread

# --- 빌드 규칙 ---
all: rtsp-demo rtsp-demo-v4l2convert

# 새로운 Zero Copy + RTSP 데모 버전 빌드
rtsp-demo: $(TARGET_RTSP_DEMO)

$(TARGET_RTSP_DEMO): $(SOURCE_RTSP_DEMO)
	$(CXX) $(CXXFLAGS) $(LIBCAMERA_INCLUDE) $(GSTREAMER_INCLUDE) -o $(TARGET_RTSP_DEMO) $(SOURCE_RTSP_DEMO) $(LIBS_RTSP_DEMO)

# 새로운 Zero Copy + RTSP 데모 버전 빌드
rtsp-demo-v4l2convert: $(TARGET_RTSP_DEMO_V4L2CONVERT)

$(TARGET_RTSP_DEMO_V4L2CONVERT): $(SOURCE_RTSP_DEMO_V4L2CONVERT)
	$(CXX) $(CXXFLAGS) $(LIBCAMERA_INCLUDE) $(GSTREAMER_INCLUDE) -o $(TARGET_RTSP_DEMO_V4L2CONVERT) $(SOURCE_RTSP_DEMO_V4L2CONVERT) $(LIBS_RTSP_DEMO)

# --- 정리 ---
clean:
	rm -f $(TARGET_RTSP_DEMO) $(TARGET_RTSP_DEMO_V4L2CONVERT) # ... (다른 타겟들)

# --- 실행 ---
run-rtsp-demo: $(TARGET_RTSP_DEMO)
	./$(TARGET_RTSP_DEMO)

run-rtsp-demo-v4l2convert: $(TARGET_RTSP_DEMO_V4L2CONVERT)
	./$(TARGET_RTSP_DEMO_V4L2CONVERT)

# --- 도움말 ---
help:
	@echo "Available targets:"
	@echo "  rtsp-demo       - Build the standalone Zero Copy + RTSP streaming demo"
	@echo "  clean           - Remove built files"
	@echo "  run-rtsp-demo   - Build and run the standalone RTSP demo"
	@echo "  help            - Show this help message"

.PHONY: all rtsp-demo clean run-rtsp-demo help

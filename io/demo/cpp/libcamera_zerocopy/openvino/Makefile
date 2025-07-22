# Makefile for Zero Copy OpenVINO Demo

# OpenVINO 환경 변수 설정
export LD_LIBRARY_PATH := /home/lee/openvino/build/install/usr/local/runtime/lib/aarch64:$(LD_LIBRARY_PATH)

# 컴파일러 설정
CXX = g++
CXXFLAGS = -std=c++17 -g -O2 -Wall

# ====================================================================
#                           프로젝트 설정
# ====================================================================
TARGET = zero_copy_openvino_demo
SOURCE = zero_copy_openvino_demo.cpp

# --- 기존 타겟들 ---
TARGET_HEADLESS = zero_copy_openvino_demo_headless
SOURCE_HEADLESS = zero_copy_openvino_demo_headless.cpp
TARGET_SORT = zero_copy_openvino_sort_demo
SOURCE_SORT = zero_copy_openvino_sort_demo.cpp
TARGET_YOLO11N = zero_copy_openvino_yolo11n_demo_headless
SOURCE_YOLO11N = zero_copy_openvino_yolo11n_demo_headless.cpp
TARGET_V2 = zero_copy_openvino_demo_headless_v2
SOURCE_V2 = zero_copy_openvino_demo_headless_v2.cpp
TARGET_V5 = zero_copy_openvino_demo_headless_v5
SOURCE_V5 = zero_copy_openvino_demo_headless_v5.cpp
TARGET_V5_TRACKING = zero_copy_openvino_yolov5_tracking_demo
SOURCE_V5_TRACKING = zero_copy_openvino_yolov5_tracking_demo.cpp
TARGET_V5_LINE_CROSSING = zero_copy_openvino_yolov5_line_crossing_demo
SOURCE_V5_LINE_CROSSING = zero_copy_openvino_yolov5_line_crossing_demo.cpp
TARGET_V5_LINE_CROSSING_VISUAL = zero_copy_openvino_yolov5_line_crossing_visual_demo
SOURCE_V5_LINE_CROSSING_VISUAL = zero_copy_openvino_yolov5_line_crossing_visual_demo.cpp
TARGET_V5_RISK_ANALYSIS = zero_copy_openvino_yolov5_risk_analysis_demo
SOURCE_V5_RISK_ANALYSIS = zero_copy_openvino_yolov5_risk_analysis_demo.cpp
TARGET_V5_RTSP = zero_copy_openvino_yolov5_rtsp_demo
SOURCE_V5_RTSP = zero_copy_openvino_yolov5_rtsp_demo.cpp

# --- 신규 추가: RTSP + 추론 통합 데모 ---
TARGET_COMBINED = combined_demo
SOURCE_COMBINED = combined_rtsp_inference_demo.cpp

# ====================================================================
#                           라이브러리 설정
# ====================================================================

# OpenVINO 설정
OPENVINO_INCLUDE = -I/usr/local/runtime/include
OPENVINO_LIBS = -L/usr/local/runtime/lib/aarch64 -lopenvino

# OpenCV 설정
OPENCV_INCLUDE = -I/usr/local/include/opencv4
OPENCV_LIBS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_dnn
OPENCV_LIBS_HEADLESS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn

# libcamera 설정
LIBCAMERA_INCLUDE = -I/usr/include/libcamera
LIBCAMERA_LIBS = -lcamera -lcamera-base

# SQLite 설정
SQLITE_LIBS = -lSQLiteCpp -lsqlite3

# GStreamer 설정
GSTREAMER_INCLUDE = `pkg-config --cflags gstreamer-1.0 gstreamer-rtsp-1.0 gstreamer-rtsp-server-1.0 gstreamer-app-1.0`
GSTREAMER_LIBS = `pkg-config --libs gstreamer-1.0 gstreamer-rtsp-1.0 gstreamer-rtsp-server-1.0 gstreamer-app-1.0`

# Eigen 헤더 경로 추가
EIGEN_INCLUDE = -I/usr/include/eigen3

# 전체 플래그 설정
INCLUDES = $(OPENVINO_INCLUDE) $(OPENCV_INCLUDE) $(LIBCAMERA_INCLUDE) $(EIGEN_INCLUDE) $(GSTREAMER_INCLUDE)
CXXFLAGS += $(INCLUDES)

# 라이브러리 조합
LIBS_BASE = $(OPENVINO_LIBS) $(LIBCAMERA_LIBS) -lpthread
LIBS_GUI = $(LIBS_BASE) $(OPENCV_LIBS)
LIBS_HEADLESS = $(LIBS_BASE) $(OPENCV_LIBS_HEADLESS)
LIBS_WITH_SQLITE = $(LIBS_HEADLESS) $(SQLITE_LIBS)
LIBS_WITH_RTSP_OLD = $(LIBS_WITH_SQLITE) $(GSTREAMER_LIBS) # 이전 v5-rtsp용
LIBS_COMBINED = $(LIBS_HEADLESS) $(GSTREAMER_LIBS) # 신규 combined-demo용


# ====================================================================
#                           빌드 규칙
# ====================================================================

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LIBS_GUI)

# --- 신규 추가: RTSP + 추론 통합 데모 빌드 ---
combined-demo: $(TARGET_COMBINED)

$(TARGET_COMBINED): $(SOURCE_COMBINED) sort.cpp kalman_tracker.cpp object_tracker.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_COMBINED) $(SOURCE_COMBINED) sort.cpp kalman_tracker.cpp object_tracker.cpp $(LIBS_COMBINED)

# --- 기존 빌드 규칙들 ---
headless: $(TARGET_HEADLESS)
$(TARGET_HEADLESS): $(SOURCE_HEADLESS)
	$(CXX) $(CXXFLAGS) -o $(TARGET_HEADLESS) $(SOURCE_HEADLESS) $(LIBS_HEADLESS)

v2: $(TARGET_V2)
$(TARGET_V2): $(SOURCE_V2)
	$(CXX) $(CXXFLAGS) -o $(TARGET_V2) $(SOURCE_V2) $(LIBS_HEADLESS)

v5: $(TARGET_V5)
$(TARGET_V5): $(SOURCE_V5)
	$(CXX) $(CXXFLAGS) -o $(TARGET_V5) $(SOURCE_V5) $(LIBS_HEADLESS)

v5-tracking: $(TARGET_V5_TRACKING)
$(TARGET_V5_TRACKING): $(SOURCE_V5_TRACKING) sort.cpp kalman_tracker.cpp object_tracker.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_V5_TRACKING) $(SOURCE_V5_TRACKING) sort.cpp kalman_tracker.cpp object_tracker.cpp $(LIBS_HEADLESS)

v5-line-crossing: $(TARGET_V5_LINE_CROSSING)
$(TARGET_V5_LINE_CROSSING): $(SOURCE_V5_LINE_CROSSING) sort.cpp kalman_tracker.cpp object_tracker.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_V5_LINE_CROSSING) $(SOURCE_V5_LINE_CROSSING) sort.cpp kalman_tracker.cpp object_tracker.cpp $(LIBS_HEADLESS)

v5-line-crossing-visual: $(TARGET_V5_LINE_CROSSING_VISUAL)
$(TARGET_V5_LINE_CROSSING_VISUAL): $(SOURCE_V5_LINE_CROSSING_VISUAL) sort.cpp kalman_tracker.cpp object_tracker.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_V5_LINE_CROSSING_VISUAL) $(SOURCE_V5_LINE_CROSSING_VISUAL) sort.cpp kalman_tracker.cpp object_tracker.cpp $(LIBS_GUI)

v5-risk-analysis: $(TARGET_V5_RISK_ANALYSIS)
$(TARGET_V5_RISK_ANALYSIS): $(SOURCE_V5_RISK_ANALYSIS) sort.cpp kalman_tracker.cpp object_tracker.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_V5_RISK_ANALYSIS) $(SOURCE_V5_RISK_ANALYSIS) sort.cpp kalman_tracker.cpp object_tracker.cpp $(LIBS_WITH_SQLITE)

v5-rtsp: $(TARGET_V5_RTSP)
$(TARGET_V5_RTSP): $(SOURCE_V5_RTSP) sort.cpp kalman_tracker.cpp object_tracker.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET_V5_RTSP) $(SOURCE_V5_RTSP) sort.cpp kalman_tracker.cpp object_tracker.cpp $(LIBS_WITH_RTSP_OLD)

zero_copy_openvino_sort_demo: zero_copy_openvino_sort_demo.cpp sort.cpp kalman_tracker.cpp object_tracker.cpp
	$(CXX) $(CXXFLAGS) -o zero_copy_openvino_sort_demo zero_copy_openvino_sort_demo.cpp sort.cpp kalman_tracker.cpp object_tracker.cpp $(LIBS_HEADLESS)

yolo11n: $(TARGET_YOLO11N)
$(TARGET_YOLO11N): $(SOURCE_YOLO11N)
	$(CXX) $(CXXFLAGS) -o $(TARGET_YOLO11N) $(SOURCE_YOLO11N) $(LIBS_HEADLESS)

# ====================================================================
#                           유틸리티 규칙
# ====================================================================

# 정리
clean:
	rm -f $(TARGET) $(TARGET_HEADLESS) $(TARGET_YOLO11N) $(TARGET_V2) $(TARGET_V5) $(TARGET_V5_TRACKING) $(TARGET_V5_LINE_CROSSING) $(TARGET_V5_LINE_CROSSING_VISUAL) $(TARGET_V5_RISK_ANALYSIS) $(TARGET_V5_RTSP) zero_copy_openvino_sort_demo $(TARGET_COMBINED)

# 실행
run: $(TARGET)
	./$(TARGET)

run-combined: $(TARGET_COMBINED)
	./$(TARGET_COMBINED)

run-headless: $(TARGET_HEADLESS)
	./$(TARGET_HEADLESS)

# ... (다른 run-* 규칙들은 생략)

# 디버그 빌드
debug: CXXFLAGS += -DDEBUG -g3
debug: $(TARGET)

# 도움말
help:
	@echo "Available targets:"
	@echo "  all                      - Build the project (default)"
	@echo "  combined-demo            - Build the combined RTSP + Inference demo"
	@echo "  ... (other targets)"
	@echo "  clean                    - Remove built files"
	@echo "  run                      - Build and run the program"
	@echo "  run-combined             - Build and run the combined demo"
	@echo "  ... (other run targets)"
	@echo "  help                     - Show this help message"

.PHONY: all combined-demo headless yolo11n v2 v5 v5-tracking v5-line-crossing v5-line-crossing-visual v5-risk-analysis v5-rtsp clean run run-combined run-headless run-yolo11n run-v2 run-v5 run-v5-tracking run-v5-line-crossing run-v5-line-crossing-visual run-v5-risk-analysis run-v5-rtsp debug check-deps install-system help

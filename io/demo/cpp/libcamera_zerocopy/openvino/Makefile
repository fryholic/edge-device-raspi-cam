# Makefile for Zero Copy OpenVINO Demo

# 컴파일러 설정
CXX = g++
CXXFLAGS = -std=c++17 -g -O2 -Wall


# 프로젝트 설정
TARGET = zero_copy_openvino_demo
SOURCE = zero_copy_openvino_demo.cpp
TARGET_HEADLESS = zero_copy_openvino_demo_headless
SOURCE_HEADLESS = zero_copy_openvino_demo_headless.cpp

# SORT 통합 버전
TARGET_SORT = zero_copy_openvino_sort_demo
SOURCE_SORT = zero_copy_openvino_sort_demo.cpp

# YOLO11n 버전
TARGET_YOLO11N = zero_copy_openvino_yolo11n_demo_headless
SOURCE_YOLO11N = zero_copy_openvino_yolo11n_demo_headless.cpp

# openVION 방식 v2 버전
TARGET_V2 = zero_copy_openvino_demo_headless_v2
SOURCE_V2 = zero_copy_openvino_demo_headless_v2.cpp

# YOLOv5 버전 (mainv5.cpp 로직 적용)
TARGET_V5 = zero_copy_openvino_demo_headless_v5
SOURCE_V5 = zero_copy_openvino_demo_headless_v5.cpp

# YOLOv5 + SORT 트래킹 버전
TARGET_V5_TRACKING = zero_copy_openvino_yolov5_tracking_demo
SOURCE_V5_TRACKING = zero_copy_openvino_yolov5_tracking_demo.cpp

# OpenVINO 설정 (GLIBC 호환 버전)
OPENVINO_INCLUDE = -I/home/lee/openvino/build/install/usr/local/runtime/include
OPENVINO_LIBS = -L/home/lee/openvino/build/install/usr/local/runtime/lib/aarch64 -lopenvino

# OpenCV 설정 (헤드리스 버전은 highgui 제외)
OPENCV_INCLUDE = -I/usr/local/include/opencv4
OPENCV_LIBS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
OPENCV_LIBS_HEADLESS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn

# libcamera 설정
LIBCAMERA_INCLUDE = -I/usr/include/libcamera
LIBCAMERA_LIBS = -lcamera -lcamera-base


# Eigen 헤더 경로 추가
EIGEN_INCLUDE = -I/usr/include/eigen3

# 전체 플래그 설정
INCLUDES = $(OPENVINO_INCLUDE) $(OPENCV_INCLUDE) $(LIBCAMERA_INCLUDE) $(EIGEN_INCLUDE)
LIBS = $(OPENVINO_LIBS) $(OPENCV_LIBS) $(LIBCAMERA_LIBS) -lpthread
LIBS_HEADLESS = $(OPENVINO_LIBS) $(OPENCV_LIBS_HEADLESS) $(LIBCAMERA_LIBS) -lpthread


# 빌드 규칙
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBS)

# 헤드리스 버전 빌드
headless: $(TARGET_HEADLESS)

$(TARGET_HEADLESS): $(SOURCE_HEADLESS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET_HEADLESS) $(SOURCE_HEADLESS) $(LIBS_HEADLESS)

# openVION 방식 v2 버전 빌드
v2: $(TARGET_V2)

$(TARGET_V2): $(SOURCE_V2)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET_V2) $(SOURCE_V2) $(LIBS_HEADLESS)

# YOLOv5 버전 빌드 (mainv5.cpp 로직 적용)
v5: $(TARGET_V5)

$(TARGET_V5): $(SOURCE_V5)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET_V5) $(SOURCE_V5) $(LIBS_HEADLESS)

# YOLOv5 + SORT 트래킹 버전 빌드
v5-tracking: $(TARGET_V5_TRACKING)

$(TARGET_V5_TRACKING): $(SOURCE_V5_TRACKING) sort.cpp kalman_tracker.cpp object_tracker.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET_V5_TRACKING) $(SOURCE_V5_TRACKING) sort.cpp kalman_tracker.cpp object_tracker.cpp $(LIBS_HEADLESS)

# SORT 통합 버전 빌드
zero_copy_openvino_sort_demo: zero_copy_openvino_sort_demo.cpp sort.cpp kalman_tracker.cpp object_tracker.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o zero_copy_openvino_sort_demo zero_copy_openvino_sort_demo.cpp sort.cpp kalman_tracker.cpp object_tracker.cpp $(LIBS_HEADLESS)

# YOLO11n 버전 빌드
yolo11n: $(TARGET_YOLO11N)

$(TARGET_YOLO11N): $(SOURCE_YOLO11N)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET_YOLO11N) $(SOURCE_YOLO11N) $(LIBS_HEADLESS)

# 정리
clean:
	rm -f $(TARGET) $(TARGET_HEADLESS) $(TARGET_YOLO11N) $(TARGET_V2) $(TARGET_V5) $(TARGET_V5_TRACKING) zero_copy_openvino_sort_demo


# 실행
run: $(TARGET)
	./$(TARGET)

# 헤드리스 버전 실행
run-headless: $(TARGET_HEADLESS)
	./$(TARGET_HEADLESS)

# SORT 통합 버전 실행
run-sort: zero_copy_openvino_sort_demo
	./zero_copy_openvino_sort_demo

# YOLO11n 버전 실행
run-yolo11n: $(TARGET_YOLO11N)
	./$(TARGET_YOLO11N)

# openVION 방식 v2 버전 실행
run-v2: $(TARGET_V2)
	./$(TARGET_V2)

# YOLOv5 버전 실행 (mainv5.cpp 로직 적용)
run-v5: $(TARGET_V5)
	./$(TARGET_V5)

# YOLOv5 + SORT 트래킹 버전 실행
run-v5-tracking: $(TARGET_V5_TRACKING)
	./$(TARGET_V5_TRACKING)

# 디버그 빌드
debug: CXXFLAGS += -DDEBUG -g3
debug: $(TARGET)

# 종속성 확인
check-deps:
	@echo "Checking dependencies..."
	@echo "OpenVINO libraries:"
	@ls -la /home/lee/openvino/build/install/usr/local/runtime/lib/aarch64/libopenvino* 2>/dev/null || echo "OpenVINO not found"
	@echo "OpenCV libraries:"
	@pkg-config --libs opencv4 2>/dev/null || echo "OpenCV pkg-config not found"
	@echo "libcamera libraries:"
	@ldconfig -p | grep camera || echo "libcamera not found"
	@echo "YOLO model file:"
	@ls -la yolo5n_openvino_model/yolov5n.xml 2>/dev/null || echo "YOLO model not found"

# 설치 (시스템 라이브러리 사용 시)
install-system: OPENVINO_INCLUDE = 
install-system: OPENVINO_LIBS = -lopenvino
install-system: $(TARGET)

# 도움말
help:
	@echo "Available targets:"
	@echo "  all        - Build the project (default)"
	@echo "  headless   - Build headless version (no GUI)"
	@echo "  yolo11n    - Build YOLO11n version (headless)"
	@echo "  v2         - Build openVION-based v2 version (headless)"
	@echo "  v5         - Build YOLOv5 version (mainv5.cpp logic, headless)"
	@echo "  v5-tracking - Build YOLOv5 + SORT tracking version (headless)"
	@echo "  clean      - Remove built files"
	@echo "  run        - Build and run the program"
	@echo "  run-headless - Build and run the headless version"
	@echo "  run-yolo11n - Build and run the YOLO11n version"
	@echo "  run-v2     - Build and run the openVION-based v2 version"
	@echo "  run-v5     - Build and run the YOLOv5 version (mainv5.cpp logic)"
	@echo "  run-v5-tracking - Build and run the YOLOv5 + SORT tracking version"
	@echo "  debug      - Build with debug information"
	@echo "  check-deps - Check if all dependencies are available"
	@echo "  install-system - Build using system-installed OpenVINO"
	@echo "  help       - Show this help message"

.PHONY: all headless yolo11n v2 v5 v5-tracking clean run run-headless run-yolo11n run-v2 run-v5 run-v5-tracking debug check-deps install-system help

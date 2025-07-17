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

# OpenVINO 설정 (GLIBC 호환 버전)
OPENVINO_INCLUDE = -I/home/lee/openvino/build/install/usr/local/runtime/include
OPENVINO_LIBS = -L/home/lee/openvino/build/install/usr/local/runtime/lib/aarch64 -lopenvino

# OpenCV 설정 (헤드리스 버전은 highgui 제외)
OPENCV_INCLUDE = -I/usr/local/include/opencv4
OPENCV_LIBS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
OPENCV_LIBS_HEADLESS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs

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

# SORT 통합 버전 빌드
zero_copy_openvino_sort_demo: zero_copy_openvino_sort_demo.cpp sort.cpp kalman_tracker.cpp object_tracker.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o zero_copy_openvino_sort_demo zero_copy_openvino_sort_demo.cpp sort.cpp kalman_tracker.cpp object_tracker.cpp $(LIBS_HEADLESS)

# YOLO11n 버전 빌드
yolo11n: $(TARGET_YOLO11N)

$(TARGET_YOLO11N): $(SOURCE_YOLO11N)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET_YOLO11N) $(SOURCE_YOLO11N) $(LIBS_HEADLESS)

# 정리
clean:
	rm -f $(TARGET) $(TARGET_HEADLESS) $(TARGET_YOLO11N)


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
	@echo "  clean      - Remove built files"
	@echo "  run        - Build and run the program"
	@echo "  run-headless - Build and run the headless version"
	@echo "  run-yolo11n - Build and run the YOLO11n version"
	@echo "  debug      - Build with debug information"
	@echo "  check-deps - Check if all dependencies are available"
	@echo "  install-system - Build using system-installed OpenVINO"
	@echo "  help       - Show this help message"

.PHONY: all headless yolo11n clean run run-headless run-yolo11n debug check-deps install-system help

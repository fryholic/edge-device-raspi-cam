# Makefile for zero_copy_demo

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 $(OPENCV_FLAGS) -I/usr/include/libcamera
LDFLAGS = -lstdc++fs $(OPENCV_FLAGS) -lcamera -lopencv_imgcodecs

# FFmpeg flags for H264 version
FFMPEG_LIBS = -lavcodec -lavformat -lavutil -lswscale

# Target executables
TARGET = zero_copy_demo
TARGET_H264 = zero_copy_demo_h264
TARGET_DMABUF = zero_copy_dma_buf

# Source files
SRCS = zero_copy_demo.cpp
SRCS_H264 = zero_copy_demo_h264.cpp
SRCS_DMABUF = zero_copy_dma_buf.cpp

# Build rules
all: $(TARGET) $(TARGET_H264) $(TARGET_DMABUF)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_H264): $(SRCS_H264)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(FFMPEG_LIBS)

$(TARGET_DMABUF): $(SRCS_DMABUF)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET_H264) $(TARGET_DMABUF)

# Individual targets
.PHONY: h264 original dmabuf clean all
original: $(TARGET)
h264: $(TARGET_H264)
dmabuf: $(TARGET_DMABUF)

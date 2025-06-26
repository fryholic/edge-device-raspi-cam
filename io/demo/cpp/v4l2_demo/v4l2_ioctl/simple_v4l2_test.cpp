#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <cerrno>
#include <cstring>

int main() {
    const char* device = "/dev/video0";
    
    // 디바이스 열기
    int fd = open(device, O_RDWR);
    if (fd == -1) {
        std::cerr << "Failed to open device: " << strerror(errno) << std::endl;
        return 1;
    }
    
    std::cout << "Device opened successfully: " << device << std::endl;
    
    // 기본 능력 확인
    struct v4l2_capability cap = {};
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        std::cerr << "VIDIOC_QUERYCAP failed: " << strerror(errno) << std::endl;
        close(fd);
        return 1;
    }
    
    std::cout << "Driver: " << cap.driver << std::endl;
    std::cout << "Card: " << cap.card << std::endl;
    std::cout << "Bus info: " << cap.bus_info << std::endl;
    std::cout << "Capabilities: 0x" << std::hex << cap.capabilities << std::dec << std::endl;
    
    // 현재 포맷 확인
    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_G_FMT, &fmt) == -1) {
        std::cerr << "VIDIOC_G_FMT failed: " << strerror(errno) << std::endl;
    } else {
        std::cout << "Current format:" << std::endl;
        std::cout << "  Width: " << fmt.fmt.pix.width << std::endl;
        std::cout << "  Height: " << fmt.fmt.pix.height << std::endl;
        std::cout << "  Pixel format: 0x" << std::hex << fmt.fmt.pix.pixelformat << std::dec << std::endl;
        std::cout << "  Bytes per line: " << fmt.fmt.pix.bytesperline << std::endl;
        std::cout << "  Size image: " << fmt.fmt.pix.sizeimage << std::endl;
    }
    
    // 아무것도 변경하지 않고 기본 설정으로 버퍼 요청
    struct v4l2_requestbuffers req = {};
    req.count = 1;  // 최소한의 버퍼만 요청
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        std::cerr << "VIDIOC_REQBUFS failed: " << strerror(errno) << std::endl;
        close(fd);
        return 1;
    }
    
    std::cout << "Requested " << req.count << " buffers" << std::endl;
    
    // 버퍼 정보 조회
    struct v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    
    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
        std::cerr << "VIDIOC_QUERYBUF failed: " << strerror(errno) << std::endl;
        close(fd);
        return 1;
    }
    
    std::cout << "Buffer info:" << std::endl;
    std::cout << "  Length: " << buf.length << std::endl;
    std::cout << "  Offset: " << buf.m.offset << std::endl;
    
    // 버퍼를 큐에 추가
    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
        std::cerr << "VIDIOC_QBUF failed: " << strerror(errno) << std::endl;
        close(fd);
        return 1;
    }
    
    std::cout << "Buffer queued successfully" << std::endl;
    
    // 스트리밍 시작 시도
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        std::cerr << "VIDIOC_STREAMON failed: " << strerror(errno) << " (errno: " << errno << ")" << std::endl;
        close(fd);
        return 1;
    }
    
    std::cout << "Streaming started successfully!" << std::endl;
    
    // 스트리밍 중지
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
        std::cerr << "VIDIOC_STREAMOFF failed: " << strerror(errno) << std::endl;
    } else {
        std::cout << "Streaming stopped successfully" << std::endl;
    }
    
    close(fd);
    return 0;
}

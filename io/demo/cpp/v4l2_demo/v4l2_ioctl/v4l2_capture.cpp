#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <cerrno>
#include <cstring>
#include <cstdio>  // printf 함수를 위해 추가

#include <opencv2/opencv.hpp> // OpenCV 헤더 추가
#include "rg10_decoder.hpp"    // RG10 디코더 추가

// 캡처된 프레임 정보를 담을 구조체
struct CapturedFrame {
    void* data;
    size_t size;
    int    buffer_index;
};

// V4L2 카메라 제어 클래스
class V4L2Camera {
public:
    // 생성자: 장치 경로, 해상도, 포맷을 받아 초기화
    V4L2Camera(const std::string& device, int width, int height, uint32_t format)
        : device_path_(device), width_(width), height_(height), format_(format), fd_(-1), is_streaming_(false) {
        open_device();
        init_device();
        init_mmap();
    }

    // 소멸자: RAII 패턴에 따라 자원 자동 해제
    ~V4L2Camera() {
        if (is_streaming_) {
            stop_streaming();
        }
        unmap_buffers();
        if (fd_ != -1) {
            close(fd_);
        }
    }

    // 스트리밍 시작
    void start_streaming() {
        if (is_streaming_) return;

        // 버퍼를 큐에 추가
        for (size_t i = 0; i < buffers_.size(); ++i) {
            struct v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            
            try {
                xioctl(VIDIOC_QBUF, &buf);
                std::cout << "Buffer " << i << " queued successfully." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to queue buffer " << i << ": " << e.what() << std::endl;
                throw;
            }
        }

        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        try {
            xioctl(VIDIOC_STREAMON, &type);
            std::cout << "Streaming started successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to start streaming: " << e.what() << std::endl;
            throw;
        }
        is_streaming_ = true;
    }

    // 스트리밍 중지
    void stop_streaming() {
        if (!is_streaming_) return;
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(VIDIOC_STREAMOFF, &type);
        is_streaming_ = false;
        std::cout << "Streaming stopped." << std::endl;
    }

    // 프레임 캡처 (데이터 포인터와 크기 반환)
    CapturedFrame capture_frame() {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);

        struct timeval tv = {};
        tv.tv_sec = 2; // 2초 타임아웃

        int r = select(fd_ + 1, &fds, NULL, NULL, &tv);
        if (r == -1) {
            throw std::system_error(errno, std::generic_category(), "select");
        }
        if (r == 0) {
            throw std::runtime_error("select timeout");
        }

        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        // 큐에서 채워진 버퍼 가져오기 (Dequeue)
        xioctl(VIDIOC_DQBUF, &buf);

        return {buffers_[buf.index].start, buf.bytesused, (int)buf.index};
    }

    // 사용 완료된 프레임 버퍼를 다시 큐에 반환
    void release_frame(const CapturedFrame& frame) {
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = frame.buffer_index;
        xioctl(VIDIOC_QBUF, &buf);
    }

private:
    // ioctl 래퍼 함수 (에러 시 예외 발생)
    void xioctl(int request, void* arg) {
        int r;
        do {
            r = ioctl(fd_, request, arg);
        } while (r == -1 && errno == EINTR);

        if (r == -1) {
            throw std::system_error(errno, std::generic_category(), "ioctl");
        }
    }

    void open_device() {
        fd_ = open(device_path_.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ == -1) {
            throw std::system_error(errno, std::generic_category(), "Failed to open device");
        }
        std::cout << "Device opened successfully: " << device_path_ << std::endl;
    }

    // 장치가 실제로 스트리밍 가능한지 확인하는 함수 추가
    bool test_streaming_capability() {
        struct v4l2_requestbuffers test_req = {};
        test_req.count = 1;
        test_req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        test_req.memory = V4L2_MEMORY_MMAP;
        
        try {
            xioctl(VIDIOC_REQBUFS, &test_req);
            if (test_req.count == 0) {
                std::cerr << "Device reports 0 buffers available - streaming not supported" << std::endl;
                return false;
            }
            
            // 테스트 버퍼 해제
            test_req.count = 0;
            xioctl(VIDIOC_REQBUFS, &test_req);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Streaming capability test failed: " << e.what() << std::endl;
            return false;
        }
    }

    void init_device() {
        struct v4l2_capability cap;
        xioctl(VIDIOC_QUERYCAP, &cap);
        
        std::cout << "Device: " << cap.driver << " (" << cap.card << ")" << std::endl;
        
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            throw std::runtime_error("Device does not support video capture");
        }
        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
            throw std::runtime_error("Device does not support streaming");
        }

        // 실제 스트리밍 가능성 테스트
        if (!test_streaming_capability()) {
            throw std::runtime_error("Device does not support actual streaming (possibly no camera connected)");
        }

        struct v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.pixelformat = format_;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        xioctl(VIDIOC_S_FMT, &fmt);

        // 실제로 설정된 값들 확인
        std::cout << "Requested: " << width_ << "x" << height_ << std::endl;
        std::cout << "Actually set: " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << std::endl;
        std::cout << "Bytes per line: " << fmt.fmt.pix.bytesperline << std::endl;
        std::cout << "Size image: " << fmt.fmt.pix.sizeimage << std::endl;

        if (fmt.fmt.pix.pixelformat != format_) {
            throw std::runtime_error("Requested pixel format not supported");
        }
        
        // 실제 설정된 값으로 업데이트
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
    }

    void init_mmap() {
        struct v4l2_requestbuffers req = {};
        req.count = 4; // 4개의 버퍼 요청
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        xioctl(VIDIOC_REQBUFS, &req);

        if (req.count < 2) {
            throw std::runtime_error("Insufficient buffer memory");
        }

        buffers_.resize(req.count);

        for (size_t i = 0; i < buffers_.size(); ++i) {
            struct v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            xioctl(VIDIOC_QUERYBUF, &buf);

            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);

            if (buffers_[i].start == MAP_FAILED) {
                throw std::system_error(errno, std::generic_category(), "mmap");
            }
            
            std::cout << "Buffer " << i << " mapped: length=" << buf.length << " offset=" << buf.m.offset << std::endl;
        }
        std::cout << buffers_.size() << " buffers mapped." << std::endl;
    }

    void unmap_buffers() {
        for (auto& buffer : buffers_) {
            if (buffer.start && buffer.start != MAP_FAILED) {
                munmap(buffer.start, buffer.length);
            }
        }
        buffers_.clear();
    }

    struct Buffer {
        void* start;
        size_t length;
    };

    std::string device_path_;
    int width_, height_;
    uint32_t format_;
    int fd_;
    bool is_streaming_;
    std::vector<Buffer> buffers_;
};

// --- 메인 함수 ---
int main() {
    const std::string device = "/dev/video0";
    const int width = 1920;   // 센서의 실제 해상도
    const int height = 1080;
    // 실제 V4L2 장치가 사용하는 포맷 (RG10)
    const uint32_t format = V4L2_PIX_FMT_SRGGB10;  // RG10에 해당 

    try {
        V4L2Camera camera(device, width, height, format);

        camera.start_streaming();

        // 50 프레임 캡처 예제
        //for (int i = 0; i < 50; ++i) {
        for( ; ;) {
            CapturedFrame frame = camera.capture_frame();
            //std::cout << "Frame " << i << " captured! Size: " << frame.size << " bytes, Buffer Index: " << frame.buffer_index << std::endl;
            std::cout << "Frame captured! Size: " << frame.size << " bytes, Buffer Index: " << frame.buffer_index << std::endl;

            // --- 여기서 프레임 데이터 처리 ---
            // 상세한 RG10 포맷 분석 및 다양한 디코딩 시도
            std::cout << "Frame size: " << frame.size << " bytes" << std::endl;
            std::cout << "Expected size: " << width * height * 5 / 4 << " bytes (for packed RG10)" << std::endl;
            
            // Raw 데이터 분석 (첫 64바이트 출력)
            uint8_t* raw_data = (uint8_t*)frame.data;
            std::cout << "First 64 bytes: " << std::endl;
            for (int i = 0; i < 64 && i < frame.size; i++) {
                printf("%02x ", raw_data[i]);
                if ((i + 1) % 16 == 0) printf("\n");
            }
            std::cout << std::endl;
            
            // 방법 1: 정확한 RG10 패킹 해제
            cv::Mat bayer_method1(height, width, CV_8UC1);
            std::vector<uint16_t> unpacked_rg10(width * height);
            unpack_v4l2_rg10(raw_data, unpacked_rg10.data(), width, height);
            
            // 방법 2: 간단한 16비트 방식
            cv::Mat bayer_method2(height, width, CV_8UC1);
            std::vector<uint16_t> unpacked_16bit(width * height);
            unpack_simple_16bit(raw_data, unpacked_16bit.data(), width, height);
            
            // 방법 3: 현재 방식 (원본)
            cv::Mat bayer_method3(height, width, CV_8UC1);
            uint16_t* src16 = (uint16_t*)frame.data;
            
            // 각 방법의 통계 수집
            uint16_t min1 = 1023, max1 = 0, min2 = 1023, max2 = 0, min3 = 1023, max3 = 0;
            
            for (int i = 0; i < width * height; i++) {
                // 방법 1: RG10 패킹 해제
                uint16_t val1 = unpacked_rg10[i];
                if (val1 < min1) min1 = val1;
                if (val1 > max1) max1 = val1;
                bayer_method1.data[i] = val1 >> 2; // 10비트 -> 8비트
                
                // 방법 2: 16비트 간단
                uint16_t val2 = unpacked_16bit[i];
                if (val2 < min2) min2 = val2;
                if (val2 > max2) max2 = val2;
                bayer_method2.data[i] = val2 >> 2; // 10비트 -> 8비트
                
                // 방법 3: 기존 방식
                uint16_t val3 = src16[i] & 0x3FF;
                if (val3 < min3) min3 = val3;
                if (val3 > max3) max3 = val3;
                bayer_method3.data[i] = val3 >> 2; // 10비트 -> 8비트
            }
            
            std::cout << "Method 1 (RG10 unpack) range: " << min1 << " - " << max1 << std::endl;
            std::cout << "Method 2 (16bit simple) range: " << min2 << " - " << max2 << std::endl;
            std::cout << "Method 3 (current) range: " << min3 << " - " << max3 << std::endl;
            
            // 각 방법으로 Bayer 변환 (RG 패턴 사용)
            cv::Mat bgr1, bgr2, bgr3;
            cv::cvtColor(bayer_method1, bgr1, cv::COLOR_BayerRG2BGR);
            cv::cvtColor(bayer_method2, bgr2, cv::COLOR_BayerRG2BGR);
            cv::cvtColor(bayer_method3, bgr3, cv::COLOR_BayerRG2BGR);
            
            // 히스토그램 스트레칭 적용
            cv::Mat enhanced1, enhanced2, enhanced3;
            cv::convertScaleAbs(bgr1, enhanced1, 4.0, 50);
            cv::convertScaleAbs(bgr2, enhanced2, 4.0, 50);
            cv::convertScaleAbs(bgr3, enhanced3, 4.0, 50);
            
            cv::imshow("Method 1: RG10 Unpack", enhanced1);
            cv::imshow("Method 2: 16bit Simple", enhanced2);
            cv::imshow("Method 3: Current", enhanced3);
            cv::imshow("Raw Bayer (Method 2)", bayer_method2);
            
            if (cv::waitKey(1) == 27) break; // ESC 키로 종료
            // --------------------------------
            
            if (cv::waitKey(1) == 27) break; // ESC 키로 종료
            // --------------------------------

            camera.release_frame(frame);
        }

        // camera.stop_streaming(); // 소멸자가 자동으로 호출해 줌

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
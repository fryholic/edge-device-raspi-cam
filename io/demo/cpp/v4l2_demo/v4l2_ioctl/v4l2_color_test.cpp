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
#include <cstdio>

#include <opencv2/opencv.hpp>

// 캡처된 프레임 정보를 담을 구조체
struct CapturedFrame {
    void* data;
    size_t size;
    int    buffer_index;
};

// V4L2 카메라 제어 클래스 (간단한 버전)
class V4L2Camera {
public:
    V4L2Camera(const std::string& device) : device_path_(device), fd_(-1), is_streaming_(false) {
        open_device();
        init_device();
        init_mmap();
    }

    ~V4L2Camera() {
        if (is_streaming_) {
            stop_streaming();
        }
        unmap_buffers();
        if (fd_ != -1) {
            close(fd_);
        }
    }

    void start_streaming() {
        if (is_streaming_) return;

        for (size_t i = 0; i < buffers_.size(); ++i) {
            struct v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            
            if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
                throw std::runtime_error("Failed to queue buffer " + std::to_string(i));
            }
        }

        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) == -1) {
            throw std::runtime_error("Failed to start streaming");
        }
        
        std::cout << "Streaming started successfully." << std::endl;
        is_streaming_ = true;
    }

    void stop_streaming() {
        if (!is_streaming_) return;
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_, VIDIOC_STREAMOFF, &type);
        is_streaming_ = false;
    }

    CapturedFrame capture_frame() {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);

        struct timeval tv = {};
        tv.tv_sec = 2;

        int r = select(fd_ + 1, &fds, NULL, NULL, &tv);
        if (r <= 0) {
            throw std::runtime_error("select timeout or error");
        }

        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
            throw std::runtime_error("Failed to dequeue buffer");
        }

        return {buffers_[buf.index].start, buf.bytesused, (int)buf.index};
    }

    void release_frame(const CapturedFrame& frame) {
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = frame.buffer_index;
        ioctl(fd_, VIDIOC_QBUF, &buf);
    }

    int get_width() const { return width_; }
    int get_height() const { return height_; }
    int get_bytes_per_line() const { return bytes_per_line_; }

private:
    void open_device() {
        fd_ = open(device_path_.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ == -1) {
            throw std::runtime_error("Failed to open device");
        }
    }

    void init_device() {
        // 현재 형식 확인
        struct v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_G_FMT, &fmt) == -1) {
            throw std::runtime_error("Failed to get format");
        }
        
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
        format_ = fmt.fmt.pix.pixelformat;
        bytes_per_line_ = fmt.fmt.pix.bytesperline;
        
        std::cout << "Camera format: " << width_ << "x" << height_ << std::endl;
        std::cout << "Pixel format: 0x" << std::hex << format_ << std::dec << std::endl;
        std::cout << "Bytes per line: " << bytes_per_line_ << std::endl;
    }

    void init_mmap() {
        struct v4l2_requestbuffers req = {};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1) {
            throw std::runtime_error("Failed to request buffers");
        }

        buffers_.resize(req.count);

        for (size_t i = 0; i < buffers_.size(); ++i) {
            struct v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            
            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) == -1) {
                throw std::runtime_error("Failed to query buffer");
            }

            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);

            if (buffers_[i].start == MAP_FAILED) {
                throw std::runtime_error("mmap failed");
            }
        }
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
    int bytes_per_line_;
    uint32_t format_;
    int fd_;
    bool is_streaming_;
    std::vector<Buffer> buffers_;
};

int main() {
    try {
        V4L2Camera camera("/dev/video0");
        camera.start_streaming();
        
        // 한 프레임 캡처
        CapturedFrame frame = camera.capture_frame();
        std::cout << "Frame captured! Size: " << frame.size << " bytes" << std::endl;

        uint8_t* raw_data = (uint8_t*)frame.data;
        int width = camera.get_width();
        int height = camera.get_height();
        int bytes_per_line = camera.get_bytes_per_line();
        
        // Raw 데이터 통계
        uint32_t sum = 0;
        uint8_t min_val = 255, max_val = 0;
        for (int i = 0; i < 10000 && i < frame.size; i++) {
            sum += raw_data[i];
            min_val = std::min(min_val, raw_data[i]);
            max_val = std::max(max_val, raw_data[i]);
        }
        std::cout << "Raw data stats: min=" << (int)min_val 
                  << " max=" << (int)max_val << " avg=" << (sum/10000) << std::endl;
        
        // Stride 고려해서 Mat 생성
        cv::Mat bayer;
        if (bytes_per_line == width) {
            bayer = cv::Mat(height, width, CV_8UC1, raw_data);
        } else {
            bayer = cv::Mat(height, bytes_per_line, CV_8UC1, raw_data);
            bayer = bayer(cv::Rect(0, 0, width, height));
        }
        
        // 모든 Bayer 패턴을 시도해서 가장 자연스러운 색상을 찾기
        std::vector<std::pair<int, std::string>> bayer_patterns = {
            {cv::COLOR_BayerBG2BGR, "BGGR"},
            {cv::COLOR_BayerGB2BGR, "GBRG"},
            {cv::COLOR_BayerRG2BGR, "RGGB"},
            {cv::COLOR_BayerGR2BGR, "GRBG"}
        };
        
        for (auto& pattern : bayer_patterns) {
            cv::Mat bgr, resized, enhanced;
            
            // Bayer to BGR 변환
            cv::cvtColor(bayer, bgr, pattern.first);
            
            // 크기 축소 (1/4 크기)
            cv::resize(bgr, resized, cv::Size(width/4, height/4));
            
            // 강한 향상 적용
            cv::convertScaleAbs(resized, enhanced, 5.0, 50);
            
            // 파일명에 패턴 이름 포함
            std::string filename = "bayer_test_" + pattern.second + ".jpg";
            cv::imwrite(filename, enhanced);
            std::cout << "Saved: " << filename << " (pattern: " << pattern.second << ")" << std::endl;
            
            // 색상 채널 분석
            std::vector<cv::Mat> channels;
            cv::split(enhanced, channels);
            
            cv::Scalar mean_b = cv::mean(channels[0]);
            cv::Scalar mean_g = cv::mean(channels[1]);
            cv::Scalar mean_r = cv::mean(channels[2]);
            
            std::cout << "  " << pattern.second << " - B:" << (int)mean_b[0] 
                      << " G:" << (int)mean_g[0] << " R:" << (int)mean_r[0] << std::endl;
            
            // 추가로 히스토그램 평활화 버전도 저장
            for (auto& ch : channels) {
                cv::equalizeHist(ch, ch);
            }
            cv::Mat enhanced_eq;
            cv::merge(channels, enhanced_eq);
            
            std::string filename_eq = "bayer_test_" + pattern.second + "_eq.jpg";
            cv::imwrite(filename_eq, enhanced_eq);
        }
        
        // Raw Bayer 패턴도 시각화
        cv::Mat bayer_vis;
        cv::resize(bayer, bayer_vis, cv::Size(width/4, height/4));
        cv::convertScaleAbs(bayer_vis, bayer_vis, 8.0, 0);
        cv::imwrite("bayer_raw_pattern.jpg", bayer_vis);
        
        std::cout << "\n=== 색상 테스트 완료 ===" << std::endl;
        std::cout << "저장된 이미지들을 확인해서 가장 자연스러운 색상의 패턴을 찾으세요:" << std::endl;
        std::cout << "- bayer_test_BGGR.jpg" << std::endl;
        std::cout << "- bayer_test_GBRG.jpg" << std::endl;
        std::cout << "- bayer_test_RGGB.jpg" << std::endl;
        std::cout << "- bayer_test_GRBG.jpg" << std::endl;
        std::cout << "- bayer_raw_pattern.jpg (Raw Bayer pattern)" << std::endl;

        camera.release_frame(frame);
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

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

// V4L2 카메라 제어 클래스
class V4L2Camera {
public:
    V4L2Camera(const std::string& device, int width, int height, uint32_t format)
        : device_path_(device), width_(width), height_(height), bytes_per_line_(0), format_(format), fd_(-1), is_streaming_(false) {
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

        // 버퍼를 큐에 추가
        for (size_t i = 0; i < buffers_.size(); ++i) {
            struct v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            
            if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
                std::cerr << "VIDIOC_QBUF failed for buffer " << i << ": " << strerror(errno) << std::endl;
                throw std::runtime_error("Failed to queue buffer " + std::to_string(i));
            }
            std::cout << "Buffer " << i << " queued successfully." << std::endl;
        }

        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) == -1) {
            std::cerr << "VIDIOC_STREAMON failed: " << strerror(errno) << std::endl;
            throw std::runtime_error("Failed to start streaming");
        }
        
        std::cout << "Streaming started successfully." << std::endl;
        is_streaming_ = true;
    }

    void stop_streaming() {
        if (!is_streaming_) return;
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        
        if (ioctl(fd_, VIDIOC_STREAMOFF, &type) == -1) {
            std::cerr << "VIDIOC_STREAMOFF failed: " << strerror(errno) << std::endl;
        } else {
            std::cout << "Streaming stopped." << std::endl;
        }
        is_streaming_ = false;
    }

    CapturedFrame capture_frame() {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);

        struct timeval tv = {};
        tv.tv_sec = 2;

        int r = select(fd_ + 1, &fds, NULL, NULL, &tv);
        if (r == -1) {
            std::cerr << "select failed: " << strerror(errno) << std::endl;
            throw std::system_error(errno, std::generic_category(), "select");
        }
        if (r == 0) {
            throw std::runtime_error("select timeout");
        }

        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_DQBUF, &buf) == -1) {
            std::cerr << "VIDIOC_DQBUF failed: " << strerror(errno) << std::endl;
            throw std::runtime_error("Failed to dequeue buffer");
        }

        return {buffers_[buf.index].start, buf.bytesused, (int)buf.index};
    }

    void release_frame(const CapturedFrame& frame) {
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = frame.buffer_index;
        
        if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
            std::cerr << "VIDIOC_QBUF (release) failed: " << strerror(errno) << std::endl;
            // 에러가 발생해도 계속 진행 (프로그램 중단 방지)
        }
    }

    // 카메라 컨트롤 설정 함수
    void set_camera_controls() {
        std::cout << "\n=== Setting camera controls ===" << std::endl;
        
        // 자동 노출 활성화
        struct v4l2_control ctrl = {};
        ctrl.id = V4L2_CID_EXPOSURE_AUTO;
        ctrl.value = V4L2_EXPOSURE_AUTO;
        if (ioctl(fd_, VIDIOC_S_CTRL, &ctrl) == 0) {
            std::cout << "Auto exposure enabled" << std::endl;
        } else {
            std::cout << "Failed to set auto exposure: " << strerror(errno) << std::endl;
        }
        
        // 자동 화이트 밸런스 활성화
        ctrl.id = V4L2_CID_AUTO_WHITE_BALANCE;
        ctrl.value = 1;
        if (ioctl(fd_, VIDIOC_S_CTRL, &ctrl) == 0) {
            std::cout << "Auto white balance enabled" << std::endl;
        } else {
            std::cout << "Failed to set auto white balance: " << strerror(errno) << std::endl;
        }
        
        // 게인 설정 시도
        ctrl.id = V4L2_CID_GAIN;
        ctrl.value = 100;  // 게인 증가
        if (ioctl(fd_, VIDIOC_S_CTRL, &ctrl) == 0) {
            std::cout << "Gain set to " << ctrl.value << std::endl;
        } else {
            std::cout << "Failed to set gain: " << strerror(errno) << std::endl;
        }
        
        // 밝기 설정
        ctrl.id = V4L2_CID_BRIGHTNESS;
        ctrl.value = 128;  // 중간값
        if (ioctl(fd_, VIDIOC_S_CTRL, &ctrl) == 0) {
            std::cout << "Brightness set to " << ctrl.value << std::endl;
        } else {
            std::cout << "Failed to set brightness: " << strerror(errno) << std::endl;
        }
        
        // 대비 설정
        ctrl.id = V4L2_CID_CONTRAST;
        ctrl.value = 128;  // 중간값
        if (ioctl(fd_, VIDIOC_S_CTRL, &ctrl) == 0) {
            std::cout << "Contrast set to " << ctrl.value << std::endl;
        } else {
            std::cout << "Failed to set contrast: " << strerror(errno) << std::endl;
        }
        
        std::cout << "=== Camera controls setting completed ===" << std::endl;
    }

private:
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

    void init_device() {
        struct v4l2_capability cap = {};
        
        // 장치 능력 확인
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) == -1) {
            std::cerr << "VIDIOC_QUERYCAP failed: " << strerror(errno) << std::endl;
            throw std::runtime_error("Failed to query device capabilities");
        }
        
        std::cout << "Device: " << cap.driver << " (" << cap.card << ")" << std::endl;
        std::cout << "Bus info: " << cap.bus_info << std::endl;
        std::cout << "Capabilities: 0x" << std::hex << cap.capabilities << std::dec << std::endl;
        
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            throw std::runtime_error("Device does not support video capture");
        }
        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
            throw std::runtime_error("Device does not support streaming");
        }

        // 지원되는 형식들 확인
        std::cout << "\n=== Supported formats ===" << std::endl;
        struct v4l2_fmtdesc fmtdesc = {};
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmtdesc.index = 0;
        
        while (ioctl(fd_, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
            std::cout << "Format " << fmtdesc.index << ": " << fmtdesc.description 
                      << " (0x" << std::hex << fmtdesc.pixelformat << std::dec << ")" << std::endl;
            fmtdesc.index++;
        }
        std::cout << "========================\n" << std::endl;

        // 현재 형식 먼저 확인
        struct v4l2_format current_fmt = {};
        current_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_G_FMT, &current_fmt) == -1) {
            std::cerr << "VIDIOC_G_FMT failed: " << strerror(errno) << std::endl;
        } else {
            std::cout << "Current format: " << std::hex << current_fmt.fmt.pix.pixelformat << std::dec << std::endl;
            std::cout << "Current size: " << current_fmt.fmt.pix.width << "x" << current_fmt.fmt.pix.height << std::endl;
        }

        // 새 형식 설정 시도
        struct v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.pixelformat = format_;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        
        std::cout << "Requesting format: 0x" << std::hex << format_ << std::dec << std::endl;
        
        // 먼저 TRY_FMT로 형식이 지원되는지 확인
        struct v4l2_format try_fmt = fmt;
        if (ioctl(fd_, VIDIOC_TRY_FMT, &try_fmt) == -1) {
            std::cerr << "VIDIOC_TRY_FMT failed: " << strerror(errno) << std::endl;
            std::cout << "Requested format not supported, using current format..." << std::endl;
            
            // 현재 형식을 그대로 사용
            if (ioctl(fd_, VIDIOC_G_FMT, &fmt) == -1) {
                throw std::runtime_error("Failed to get current format");
            }
            std::cout << "Using current format without modification" << std::endl;
        } else {
            std::cout << "TRY_FMT suggests: " << try_fmt.fmt.pix.width << "x" << try_fmt.fmt.pix.height 
                      << " format 0x" << std::hex << try_fmt.fmt.pix.pixelformat << std::dec << std::endl;
            
            // TRY_FMT가 성공하면 실제로 설정
            if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
                std::cerr << "VIDIOC_S_FMT failed: " << strerror(errno) << std::endl;
                std::cout << "S_FMT failed, using current format..." << std::endl;
                
                if (ioctl(fd_, VIDIOC_G_FMT, &fmt) == -1) {
                    throw std::runtime_error("Failed to get current format");
                }
            }
        }

        std::cout << "Requested: " << width_ << "x" << height_ << std::endl;
        std::cout << "Actually set: " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << std::endl;
        std::cout << "Bytes per line: " << fmt.fmt.pix.bytesperline << std::endl;
        std::cout << "Size image: " << fmt.fmt.pix.sizeimage << std::endl;
        std::cout << "Final format: 0x" << std::hex << fmt.fmt.pix.pixelformat << std::dec << std::endl;
        
        // 실제 설정된 값으로 업데이트
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
        format_ = fmt.fmt.pix.pixelformat;
        bytes_per_line_ = fmt.fmt.pix.bytesperline;  // stride 정보 저장
        
        // 추가: field 설정 확인
        std::cout << "Field setting: " << fmt.fmt.pix.field << std::endl;
        
        // 카메라 컨트롤 설정
        set_camera_controls();
    }

    void init_mmap() {
        // 먼저 기존 버퍼들을 해제
        struct v4l2_requestbuffers req_clear = {};
        req_clear.count = 0;
        req_clear.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req_clear.memory = V4L2_MEMORY_MMAP;
        ioctl(fd_, VIDIOC_REQBUFS, &req_clear); // 에러 무시
        
        struct v4l2_requestbuffers req = {};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1) {
            std::cerr << "VIDIOC_REQBUFS failed: " << strerror(errno) << std::endl;
            throw std::runtime_error("Failed to request buffers");
        }

        if (req.count < 2) {
            throw std::runtime_error("Insufficient buffer memory (got " + std::to_string(req.count) + " buffers)");
        }

        std::cout << "Allocated " << req.count << " buffers" << std::endl;
        buffers_.resize(req.count);

        for (size_t i = 0; i < buffers_.size(); ++i) {
            struct v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            
            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) == -1) {
                std::cerr << "VIDIOC_QUERYBUF failed for buffer " << i << ": " << strerror(errno) << std::endl;
                throw std::runtime_error("Failed to query buffer " + std::to_string(i));
            }

            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);

            if (buffers_[i].start == MAP_FAILED) {
                std::cerr << "mmap failed for buffer " << i << ": " << strerror(errno) << std::endl;
                throw std::system_error(errno, std::generic_category(), "mmap");
            }
            
            std::cout << "Buffer " << i << " mapped: length=" << buf.length << " offset=" << buf.m.offset << std::endl;
        }
        std::cout << buffers_.size() << " buffers mapped successfully." << std::endl;
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
    int bytes_per_line_;  // stride 정보
    uint32_t format_;
    int fd_;
    bool is_streaming_;
    std::vector<Buffer> buffers_;
};

// 간단한 화이트 밸런스 적용 함수
cv::Mat apply_simple_white_balance(const cv::Mat& src) {
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    
    // 각 채널의 평균값 계산
    cv::Scalar mean_b = cv::mean(channels[0]);
    cv::Scalar mean_g = cv::mean(channels[1]);
    cv::Scalar mean_r = cv::mean(channels[2]);
    
    // 그린 채널을 기준으로 보정 (더 보수적으로)
    double target = (mean_b[0] + mean_g[0] + mean_r[0]) / 3.0;
    
    double gain_b = (mean_b[0] > 10) ? std::min(2.0, target / mean_b[0]) : 1.0;
    double gain_g = (mean_g[0] > 10) ? std::min(2.0, target / mean_g[0]) : 1.0;
    double gain_r = (mean_r[0] > 10) ? std::min(2.0, target / mean_r[0]) : 1.0;
    
    channels[0] *= gain_b;
    channels[1] *= gain_g;
    channels[2] *= gain_r;
    
    cv::Mat result;
    cv::merge(channels, result);
    return result;
}

int main() {
    const std::string device = "/dev/video0";
    // dmesg에서 확인된 실제 센서 해상도 사용
    const int width = 3280;
    const int height = 2464;
    
    std::cout << "=== Using sensor native resolution: " << width << "x" << height << " ===" << std::endl;
    
    // 현재 설정된 형식(0x42474752 = 'RGGB')을 우선 시도
    std::vector<std::pair<uint32_t, std::string>> formats_to_try = {
        {0x42474752, "RGGB8 (current)"},        // 현재 설정된 형식
        {V4L2_PIX_FMT_SRGGB8, "RGGB8"},         // 8비트 Bayer
        {V4L2_PIX_FMT_SGRBG8, "GRBG8"},         // 8비트 Bayer
        {V4L2_PIX_FMT_SGBRG8, "GBRG8"},         // 8비트 Bayer
        {V4L2_PIX_FMT_SBGGR8, "BGGR8"},         // 8비트 Bayer
        {V4L2_PIX_FMT_YUYV, "YUYV"},            // 일반적인 형식
        {V4L2_PIX_FMT_MJPEG, "MJPEG"}           // MJPEG
    };
    
    for (auto& format_pair : formats_to_try) {
        uint32_t format = format_pair.first;
        std::string format_name = format_pair.second;
        
        std::cout << "\n=== Trying format: " << format_name << " (0x" << std::hex << format << std::dec << ") ===" << std::endl;
        
        try {
            V4L2Camera camera(device, width, height, format);
            camera.start_streaming();
            
            std::cout << "SUCCESS! Using format: " << format_name << std::endl;

            for (int frame_count = 0; frame_count < 10; ++frame_count) {  // 처음에는 10프레임만 테스트
                CapturedFrame frame = camera.capture_frame();
                std::cout << "Frame " << frame_count << " captured! Size: " << frame.size << " bytes" << std::endl;

                uint8_t* raw_data = (uint8_t*)frame.data;
                
                // 첫 번째 프레임에서 Raw 데이터 일부 출력
                if (frame_count == 0) {
                    std::cout << "Raw data sample (first 32 bytes): ";
                    for (int i = 0; i < 32 && i < frame.size; i++) {
                        printf("%02x ", raw_data[i]);
                    }
                    std::cout << std::endl;
                }

                camera.release_frame(frame);
            }
            
            std::cout << "Format " << format_name << " works! Starting live view..." << std::endl;
            std::cout << "Press ESC to exit, 's' to save screenshot" << std::endl;
            
            // 성공한 형식으로 라이브 뷰 시작
            for (int frame_count = 0; ; ++frame_count) {
                CapturedFrame frame = camera.capture_frame();
                if (frame_count % 30 == 0) {  // 매 30프레임마다 상태 출력
                    std::cout << "Frame " << frame_count << " captured! Size: " << frame.size << " bytes" << std::endl;
                }

                uint8_t* raw_data = (uint8_t*)frame.data;
                
                // 첫 번째 프레임에서 실제 형식 정보 출력
                if (frame_count == 0) {
                    std::cout << "Live view using BGGR8 format" << std::endl;
                    std::cout << "Frame size: " << frame.size << " bytes" << std::endl;
                    
                    // Raw 데이터 통계
                    uint32_t sum = 0;
                    uint8_t min_val = 255, max_val = 0;
                    for (int i = 0; i < std::min((size_t)1000, frame.size); i++) {
                        sum += raw_data[i];
                        min_val = std::min(min_val, raw_data[i]);
                        max_val = std::max(max_val, raw_data[i]);
                    }
                    std::cout << "Raw data stats (first 1000 pixels): min=" << (int)min_val 
                              << " max=" << (int)max_val << " avg=" << (sum/1000) << std::endl;
                }
                
                // Raw Bayer 데이터를 OpenCV Mat으로 변환 (stride 고려)
                cv::Mat bayer;
                if (3296 == 3280) {
                    // Stride가 width와 같으면 단순하게 생성
                    bayer = cv::Mat(2464, 3280, CV_8UC1, raw_data);
                } else {
                    // Stride가 다르면 적절히 처리 (3296 stride, 3280 actual width)
                    bayer = cv::Mat(2464, 3296, CV_8UC1, raw_data);
                    bayer = bayer(cv::Rect(0, 0, 3280, 2464));  // crop to actual size
                }
                
                // BGGR8 형식이므로 COLOR_BayerBG2BGR 사용
                cv::Mat bgr;
                cv::cvtColor(bayer, bgr, cv::COLOR_BayerBG2BGR);
                
                // 이미지 크기를 줄여서 표시 (3280x2464 -> 820x616, 1/4 크기)
                cv::Mat resized;
                cv::resize(bgr, resized, cv::Size(820, 616));
                
                // 적절한 향상 적용 (노출이 개선되었으므로 더 적은 게인 사용)
                cv::Mat enhanced;
                cv::convertScaleAbs(resized, enhanced, 1.5, 10);  // 더 자연스러운 게인
                
                // 화이트 밸런스 적용
                cv::Mat wb_corrected = apply_simple_white_balance(resized);
                cv::Mat wb_enhanced;
                cv::convertScaleAbs(wb_corrected, wb_enhanced, 1.2, 5);
                
                // Raw Bayer 패턴도 표시 (크기 축소)
                cv::Mat bayer_display;
                cv::resize(bayer, bayer_display, cv::Size(410, 308));  // 1/8 크기
                cv::convertScaleAbs(bayer_display, bayer_display, 2.0, 0);  // 자연스러운 게인
                
                try {
                    // 창 제목 업데이트
                    cv::imshow("BGGR8 - Enhanced", enhanced);
                    cv::imshow("BGGR8 - White Balance", wb_enhanced);
                    cv::imshow("Raw Bayer", bayer_display);
                    
                    char key = cv::waitKey(1);
                    if (key == 27) break; // ESC로 종료
                    if (key == 's') {
                        // 스크린샷 저장 (원본 크기)
                        cv::imwrite("live_screenshot_full.jpg", bgr);
                        cv::imwrite("live_screenshot_enhanced.jpg", enhanced);
                        cv::imwrite("live_screenshot_wb.jpg", wb_enhanced);
                        std::cout << "Screenshots saved!" << std::endl;
                    }
                } catch (const cv::Exception& e) {
                    std::cout << "OpenCV GUI error: " << e.what() << std::endl;
                    std::cout << "Continuing without GUI..." << std::endl;
                    
                    // GUI 없이 계속 실행 (자동 스크린샷 저장)
                    if (frame_count % 60 == 0) {  // 매 60프레임마다 스크린샷 저장
                        cv::imwrite("auto_screenshot_" + std::to_string(frame_count) + ".jpg", enhanced);
                        std::cout << "Auto screenshot saved: frame " << frame_count << std::endl;
                    }
                    
                    // 짧은 대기
                    usleep(50000);  // 50ms
                }
                
                camera.release_frame(frame);
            }
            
            cv::destroyAllWindows();
            return 0;  // 성공적으로 종료
            
        } catch (const std::exception& e) {
            std::cerr << "Format " << format_name << " failed: " << e.what() << std::endl;
            continue;  // 다음 형식 시도
        }
    }
    
    std::cout << "All formats failed!" << std::endl;
    return 1;
}

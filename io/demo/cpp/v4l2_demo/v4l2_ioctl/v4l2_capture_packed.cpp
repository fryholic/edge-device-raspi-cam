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

// MIPI CSI-2 packed RG10 디코딩 함수
void unpack_mipi_rg10(const uint8_t* packed, uint16_t* unpacked, int width, int height) {
    // MIPI CSI-2 RG10 패킹: 4픽셀을 5바이트에 패킹
    // [P0_9:2][P1_9:2][P2_9:2][P3_9:2][P3_1:0|P2_1:0|P1_1:0|P0_1:0]
    
    int total_pixels = width * height;
    int packed_idx = 0;
    
    for (int i = 0; i < total_pixels; i += 4) {
        if (i + 3 < total_pixels && packed_idx + 4 < (total_pixels * 5) / 4) {
            // 5바이트 읽기
            uint8_t b0 = packed[packed_idx];     // P0 상위 8비트
            uint8_t b1 = packed[packed_idx + 1]; // P1 상위 8비트  
            uint8_t b2 = packed[packed_idx + 2]; // P2 상위 8비트
            uint8_t b3 = packed[packed_idx + 3]; // P3 상위 8비트
            uint8_t b4 = packed[packed_idx + 4]; // 4픽셀의 하위 2비트들
            
            // 10비트 픽셀 재구성 (MIPI 방식)
            unpacked[i]     = (b0 << 2) | ((b4 >> 0) & 0x03);
            unpacked[i + 1] = (b1 << 2) | ((b4 >> 2) & 0x03);
            unpacked[i + 2] = (b2 << 2) | ((b4 >> 4) & 0x03);
            unpacked[i + 3] = (b3 << 2) | ((b4 >> 6) & 0x03);
            
            packed_idx += 5;
        }
    }
}

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
        : device_path_(device), width_(width), height_(height), format_(format), fd_(-1), is_streaming_(false) {
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

    void stop_streaming() {
        if (!is_streaming_) return;
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(VIDIOC_STREAMOFF, &type);
        is_streaming_ = false;
        std::cout << "Streaming stopped." << std::endl;
    }

    CapturedFrame capture_frame() {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);

        struct timeval tv = {};
        tv.tv_sec = 2;

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

        xioctl(VIDIOC_DQBUF, &buf);

        return {buffers_[buf.index].start, buf.bytesused, (int)buf.index};
    }

    void release_frame(const CapturedFrame& frame) {
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = frame.buffer_index;
        xioctl(VIDIOC_QBUF, &buf);
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

        std::cout << "Requested: " << width_ << "x" << height_ << std::endl;
        std::cout << "Actually set: " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << std::endl;
        std::cout << "Bytes per line: " << fmt.fmt.pix.bytesperline << std::endl;
        std::cout << "Size image: " << fmt.fmt.pix.sizeimage << std::endl;

        if (fmt.fmt.pix.pixelformat != format_) {
            std::cout << "Requested format: " << std::hex << format_ << std::endl;
            std::cout << "Actually set format: " << std::hex << fmt.fmt.pix.pixelformat << std::endl;
            throw std::runtime_error("Requested pixel format not supported");
        }
        
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
    }

    void init_mmap() {
        struct v4l2_requestbuffers req = {};
        req.count = 4;
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

// 화이트 밸런스 적용 함수
cv::Mat apply_white_balance(const cv::Mat& src) {
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    
    // 각 채널의 평균값 계산
    cv::Scalar mean_b = cv::mean(channels[0]);
    cv::Scalar mean_g = cv::mean(channels[1]);
    cv::Scalar mean_r = cv::mean(channels[2]);
    
    // 그린 채널을 기준으로 보정
    double avg = (mean_b[0] + mean_g[0] + mean_r[0]) / 3.0;
    
    if (mean_b[0] > 0) channels[0] *= (avg / mean_b[0]);
    if (mean_g[0] > 0) channels[1] *= (avg / mean_g[0]);
    if (mean_r[0] > 0) channels[2] *= (avg / mean_r[0]);
    
    cv::Mat result;
    cv::merge(channels, result);
    return result;
}

int main() {
    const std::string device = "/dev/video0";
    const int width = 1920;
    const int height = 1080;
    
    // packed 형식과 unpacked 형식 모두 시도
    std::vector<std::pair<uint32_t, std::string>> formats = {
        {v4l2_fourcc('p','R','A','A'), "pRAA (MIPI Packed RG10)"},  // MIPI packed
        {V4L2_PIX_FMT_SRGGB10, "RG10 (Unpacked)"}                  // Unpacked
    };

    for (auto& format_pair : formats) {
        uint32_t format = format_pair.first;
        std::string format_name = format_pair.second;
        
        std::cout << "\n=== Trying format: " << format_name << " ===" << std::endl;
        
        try {
            V4L2Camera camera(device, width, height, format);
            camera.start_streaming();

            for (int frame_count = 0; frame_count < 10; ++frame_count) {
                CapturedFrame frame = camera.capture_frame();
                std::cout << "Frame " << frame_count << " captured! Size: " << frame.size << " bytes" << std::endl;

                uint8_t* raw_data = (uint8_t*)frame.data;
                
                // 첫 번째 줄의 일부 데이터 출력 (디버깅용)
                if (frame_count == 0) {
                    std::cout << "First 32 bytes: ";
                    for (int i = 0; i < 32 && i < frame.size; i++) {
                        printf("%02x ", raw_data[i]);
                    }
                    std::cout << std::endl;
                }

                cv::Mat bayer(height, width, CV_8UC1);
                std::vector<uint16_t> unpacked(width * height);
                
                if (format == v4l2_fourcc('p','R','A','A')) {
                    // MIPI packed 형식 처리
                    std::cout << "Processing MIPI packed format..." << std::endl;
                    unpack_mipi_rg10(raw_data, unpacked.data(), width, height);
                    
                    // 통계 계산
                    uint16_t min_val = 1023, max_val = 0;
                    uint32_t sum = 0;
                    for (int i = 0; i < width * height; i++) {
                        uint16_t val = unpacked[i];
                        if (val < min_val) min_val = val;
                        if (val > max_val) max_val = val;
                        sum += val;
                        bayer.data[i] = val >> 2; // 10비트 -> 8비트
                    }
                    uint32_t avg_val = sum / (width * height);
                    std::cout << "MIPI values range: " << min_val << " - " << max_val << ", avg: " << avg_val << std::endl;
                    
                } else {
                    // Unpacked 형식 처리
                    std::cout << "Processing unpacked format..." << std::endl;
                    uint16_t* src16 = (uint16_t*)raw_data;
                    
                    uint16_t min_val = 1023, max_val = 0;
                    uint32_t sum = 0;
                    for (int i = 0; i < width * height; i++) {
                        uint16_t val = src16[i] & 0x3FF;
                        unpacked[i] = val;
                        if (val < min_val) min_val = val;
                        if (val > max_val) max_val = val;
                        sum += val;
                        bayer.data[i] = val >> 2; // 10비트 -> 8비트
                    }
                    uint32_t avg_val = sum / (width * height);
                    std::cout << "Unpacked values range: " << min_val << " - " << max_val << ", avg: " << avg_val << std::endl;
                }

                // 여러 Bayer 패턴 시도
                std::vector<std::pair<int, std::string>> bayer_patterns = {
                    {cv::COLOR_BayerRG2BGR, "RG"},
                    {cv::COLOR_BayerGR2BGR, "GR"},
                    {cv::COLOR_BayerGB2BGR, "GB"},
                    {cv::COLOR_BayerBG2BGR, "BG"}
                };

                for (auto& pattern : bayer_patterns) {
                    cv::Mat bgr, enhanced;
                    cv::cvtColor(bayer, bgr, pattern.first);
                    
                    // 화이트 밸런스 적용
                    cv::Mat wb_bgr = apply_white_balance(bgr);
                    
                    // 히스토그램 스트레칭
                    cv::convertScaleAbs(wb_bgr, enhanced, 3.0, 30);
                    
                    // 감마 보정
                    cv::Mat gamma_corrected;
                    wb_bgr.convertTo(gamma_corrected, CV_32F, 1.0/255.0);
                    cv::pow(gamma_corrected, 0.6, gamma_corrected);
                    gamma_corrected.convertTo(gamma_corrected, CV_8U, 255.0);
                    
                    std::string window_name = format_name + " - " + pattern.second;
                    cv::imshow(window_name, enhanced);
                    cv::imshow(window_name + " (Gamma)", gamma_corrected);
                }
                
                if (cv::waitKey(30) == 27) break; // ESC로 종료
                camera.release_frame(frame);
            }

            break; // 성공한 형식을 찾으면 중단
            
        } catch (const std::exception& e) {
            std::cerr << "Error with " << format_name << ": " << e.what() << std::endl;
            cv::destroyAllWindows();
            continue; // 다음 형식 시도
        }
    }

    cv::destroyAllWindows();
    return 0;
}

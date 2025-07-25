#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <system_error>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <thread>

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
    bool   show_timing;  // 타이밍 정보 출력 여부
};

// V4L2 카메라 제어 클래스
class V4L2Camera {
public:
    V4L2Camera(const std::string& device_path, int width, int height, uint32_t format, bool use_dmabuf = false);
    ~V4L2Camera();

    CapturedFrame capture_frame(bool show_timing = false);
    void release_frame(const CapturedFrame& frame);
    void start_streaming();
    void stop_streaming();

    int get_width() const { return width_; }
    int get_height() const { return height_; }
    uint32_t get_format() const { return format_; }
    bool is_using_dmabuf() const { return use_dmabuf_; }

private:
    void open_device() {
        fd_ = open(device_path_.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (fd_ == -1) {
            throw std::system_error(errno, std::generic_category(), "Failed to open device");
        }
        std::cout << "Device opened successfully: " << device_path_ << std::endl;
    }

    void init_device() {
        struct v4l2_capability cap = {};

        // 장치 능력 확인
        std::cout << "Querying device capabilities..." << std::endl;
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) == -1) {
            std::cerr << "VIDIOC_QUERYCAP failed: " << strerror(errno) << " (errno: " << errno << ")" << std::endl;
            std::cerr << "Possible reasons: device is not a V4L2 device, or insufficient permissions." << std::endl;
            throw std::runtime_error("Failed to query device capabilities");
        }

        std::cout << "Device capabilities queried successfully." << std::endl;
        std::cout << "Device: " << cap.driver << " (" << cap.card << ")" << std::endl;
        std::cout << "Bus info: " << cap.bus_info << std::endl;

        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            std::cerr << "Device does not support video capture." << std::endl;
            throw std::runtime_error("Device does not support video capture");
        }
        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
            std::cerr << "Device does not support streaming." << std::endl;
            throw std::runtime_error("Device does not support streaming");
        }

        std::cout << "Device supports video capture and streaming." << std::endl;

        // 지원되는 포맷과 해상도 조회
        list_supported_formats();
        check_supported_resolutions();
        
        // 하드웨어 제한 및 프레임률 확인
        check_hardware_capabilities();
        check_and_set_framerate();

        // 현재 형식 확인
        struct v4l2_format current_fmt = {};
        current_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        std::cout << "Getting current format..." << std::endl;
        if (ioctl(fd_, VIDIOC_G_FMT, &current_fmt) == -1) {
            std::cout << "Could not get current format, proceeding..." << std::endl;
        } else {
            std::cout << "Current format: 0x" << std::hex << current_fmt.fmt.pix.pixelformat << std::dec;
            print_fourcc_format(current_fmt.fmt.pix.pixelformat);
            std::cout << "Current size: " << current_fmt.fmt.pix.width << "x" << current_fmt.fmt.pix.height << std::endl;
            std::cout << "Bytes per line: " << current_fmt.fmt.pix.bytesperline << std::endl;
        }

        // 1920x1080 해상도와 YUYV 포맷 설정 시도
        struct v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        // 현재 포맷을 기본으로 가져오기
        std::cout << "Getting default format..." << std::endl;
        if (ioctl(fd_, VIDIOC_G_FMT, &fmt) == -1) {
            throw std::runtime_error("Failed to get format");
        }

        // 1920x1080 해상도와 MJPEG 포맷 설정 시도 (더 높은 FPS를 위해)
        fmt.fmt.pix.width = 1920;
        fmt.fmt.pix.height = 1080;
        fmt.fmt.pix.pixelformat = v4l2_fourcc('M', 'J', 'P', 'G');

        std::cout << "\nTrying to set 1920x1080 resolution with MJPEG format..." << std::endl;

        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
            std::cerr << "MJPEG format failed, trying YUYV: " << strerror(errno) << std::endl;
            // MJPEG 실패 시 YUYV로 폴백
            fmt.fmt.pix.pixelformat = v4l2_fourcc('Y', 'U', 'Y', 'V');
            std::cout << "Trying to set 1920x1080 resolution with YUYV format..." << std::endl;
            
            if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
                std::cerr << "VIDIOC_S_FMT failed: " << strerror(errno) << std::endl;
                // 실패 시 현재 설정으로 계속 진행
                if (ioctl(fd_, VIDIOC_G_FMT, &fmt) == -1) {
                    throw std::runtime_error("Failed to get format after S_FMT failure");
                }
            } else {
                std::cout << "Successfully set to 1920x1080 with YUYV!" << std::endl;
            }
        } else {
            std::cout << "Successfully set to 1920x1080 with MJPEG!" << std::endl;
        }

        // 실제 설정된 값으로 업데이트
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
        format_ = fmt.fmt.pix.pixelformat;
        bytes_per_line_ = fmt.fmt.pix.bytesperline;

        std::cout << "Final format: " << width_ << "x" << height_ << std::endl;
        std::cout << "Format: 0x" << std::hex << format_ << std::dec;
        print_fourcc_format(format_);
        std::cout << "Bytes per line: " << bytes_per_line_ << std::endl;
    }

    void init_mmap() {
        if (use_dmabuf_) {
            init_dmabuf();
            return;
        }
        
        struct v4l2_requestbuffers req = {};
        req.count = 3;  // 최소한의 버퍼로 오버헤드 감소 (3개면 충분)
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

    void list_supported_formats() {
        std::cout << "Supported formats:" << std::endl;
        struct v4l2_fmtdesc fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        for (fmt.index = 0; ioctl(fd_, VIDIOC_ENUM_FMT, &fmt) == 0; fmt.index++) {
            std::cout << "  " << fmt.index << ": " << fmt.description 
                      << " (0x" << std::hex << fmt.pixelformat << std::dec << ")";
            print_fourcc_format(fmt.pixelformat);
        }
    }

    void print_fourcc_format(uint32_t fourcc) {
        char fourcc_str[5];
        fourcc_str[0] = fourcc & 0xFF;
        fourcc_str[1] = (fourcc >> 8) & 0xFF;
        fourcc_str[2] = (fourcc >> 16) & 0xFF;
        fourcc_str[3] = (fourcc >> 24) & 0xFF;
        fourcc_str[4] = '\0';
        std::cout << " [" << fourcc_str << "]" << std::endl;
    }

    void check_supported_resolutions() {
        std::cout << "\nChecking supported resolutions for common formats..." << std::endl;
        
        // 일반적인 해상도들 체크
        std::vector<std::pair<int, int>> resolutions = {
            {1920, 1080}, {1640, 1232}, {1280, 720}, {640, 480}, {320, 240}
        };
        
        // 현재 포맷으로 해상도 체크
        struct v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_G_FMT, &fmt) == -1) {
            return;
        }
        
        uint32_t current_format = fmt.fmt.pix.pixelformat;
        
        for (const auto& res : resolutions) {
            fmt.fmt.pix.width = res.first;
            fmt.fmt.pix.height = res.second;
            fmt.fmt.pix.pixelformat = current_format;
            
            // TRY_FMT으로 지원 여부 확인 (실제 설정하지 않음)
            if (ioctl(fd_, VIDIOC_TRY_FMT, &fmt) == 0) {
                if (fmt.fmt.pix.width == res.first && fmt.fmt.pix.height == res.second) {
                    std::cout << "  ✓ " << res.first << "x" << res.second << " supported" << std::endl;
                } else {
                    std::cout << "  ~ " << res.first << "x" << res.second << " -> adjusted to " 
                              << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << std::endl;
                }
            } else {
                std::cout << "  ✗ " << res.first << "x" << res.second << " not supported" << std::endl;
            }
        }
    }

    void check_and_set_framerate() {
        std::cout << "\nChecking current framerate capabilities..." << std::endl;
        
        // 현재 프레임률 확인
        struct v4l2_streamparm parm = {};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        
        if (ioctl(fd_, VIDIOC_G_PARM, &parm) == 0) {
            if (parm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME) {
                double current_fps = (double)parm.parm.capture.timeperframe.denominator / 
                                   parm.parm.capture.timeperframe.numerator;
                std::cout << "Current FPS: " << current_fps << std::endl;
                std::cout << "Timeperframe: " << parm.parm.capture.timeperframe.numerator 
                         << "/" << parm.parm.capture.timeperframe.denominator << std::endl;
            } else {
                std::cout << "Device does not support framerate control" << std::endl;
            }
        }
        
        // 30fps 설정 시도
        std::cout << "Attempting to set 30 FPS..." << std::endl;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = 30;
        
        if (ioctl(fd_, VIDIOC_S_PARM, &parm) == 0) {
            std::cout << "Successfully requested 30 FPS" << std::endl;
            
            // 실제 설정된 값 확인
            if (ioctl(fd_, VIDIOC_G_PARM, &parm) == 0) {
                double actual_fps = (double)parm.parm.capture.timeperframe.denominator / 
                                  parm.parm.capture.timeperframe.numerator;
                std::cout << "Actual FPS set to: " << actual_fps << std::endl;
            }
        } else {
            std::cerr << "Failed to set framerate: " << strerror(errno) << std::endl;
        }
    }

    void check_hardware_capabilities() {
        std::cout << "\nChecking hardware capabilities..." << std::endl;
        
        // 버퍼 정보 확인
        struct v4l2_requestbuffers req = {};
        req.count = 0;  // 현재 할당된 버퍼 수 확인
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == 0) {
            std::cout << "Current allocated buffers: " << req.count << std::endl;
        }
        
        // 드라이버 정보 재확인
        struct v4l2_capability cap = {};
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) == 0) {
            std::cout << "Driver version: " << ((cap.version >> 16) & 0xFF) << "."
                      << ((cap.version >> 8) & 0xFF) << "." << (cap.version & 0xFF) << std::endl;
        }
    }

    struct Buffer {
        void* start;
        size_t length;
        int dmabuf_fd;  // DMA-BUF 파일 디스크립터
    };

    void init_dmabuf() {
        std::cout << "Attempting to initialize DMA-BUF..." << std::endl;
        
        struct v4l2_requestbuffers req = {};
        req.count = 3;  // 최소한의 버퍼로 오버헤드 감소
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_DMABUF;
        
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1) {
            std::cerr << "DMA-BUF not supported, falling back to MMAP: " << strerror(errno) << std::endl;
            use_dmabuf_ = false;
            init_mmap_fallback();
            return;
        }

        if (req.count < 2) {
            throw std::runtime_error("Insufficient DMA-BUF buffer memory");
        }

        std::cout << "DMA-BUF: Allocated " << req.count << " buffers" << std::endl;
        buffers_.resize(req.count);

        // DMA-BUF의 경우 실제 버퍼 할당은 별도로 처리해야 함
        // 여기서는 단순히 기본 구현으로 폴백
        std::cout << "DMA-BUF initialization complete, but using MMAP for buffer mapping." << std::endl;
        use_dmabuf_ = false;  // 실제 DMA-BUF 구현 완료 시까지 MMAP 사용
        init_mmap_fallback();
    }

    void init_mmap_fallback() {
        struct v4l2_requestbuffers req = {};
        req.count = 3;  // 최소한의 버퍼로 오버헤드 감소
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
            buffers_[i].dmabuf_fd = -1;  // MMAP에서는 사용하지 않음
            buffers_[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);

            if (buffers_[i].start == MAP_FAILED) {
                std::cerr << "mmap failed for buffer " << i << ": " << strerror(errno) << std::endl;
                throw std::system_error(errno, std::generic_category(), "mmap");
            }
            
            std::cout << "Buffer " << i << " mapped: length=" << buf.length << " offset=" << buf.m.offset << std::endl;
        }
        std::cout << buffers_.size() << " buffers mapped successfully." << std::endl;
    }

    std::string device_path_;
    int width_, height_;
    int bytes_per_line_;
    uint32_t format_;
    int fd_;
    bool is_streaming_;
    bool use_dmabuf_;
    std::vector<Buffer> buffers_;
};

V4L2Camera::V4L2Camera(const std::string& device_path, int width, int height, uint32_t format, bool use_dmabuf)
    : device_path_(device_path), width_(width), height_(height), format_(format), fd_(-1), is_streaming_(false), use_dmabuf_(use_dmabuf) {
    try {
        open_device();
        init_device();
        init_mmap();
    } catch (const std::exception& e) {
        if (fd_ != -1) {
            close(fd_);
        }
        throw;
    }
}

V4L2Camera::~V4L2Camera() {
    if (fd_ != -1) {
        if (is_streaming_) {
            try {
                stop_streaming();
            } catch (const std::exception& e) {
                std::cerr << "Error in stop_streaming during destruction: " << e.what() << std::endl;
            }
        }
        unmap_buffers();
        close(fd_);
        std::cout << "Device closed." << std::endl;
    }
}

void V4L2Camera::start_streaming() {
    if (is_streaming_) {
        std::cout << "Already streaming." << std::endl;
        return;
    }

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
        throw std::runtime_error("Failed to start streaming (VIDIOC_STREAMON)");
    }

    is_streaming_ = true;
    std::cout << "Streaming started." << std::endl;
}

void V4L2Camera::stop_streaming() {
    if (!is_streaming_) {
        return;
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMOFF, &type) == -1) {
        std::cerr << "Warning: Failed to stop streaming (VIDIOC_STREAMOFF): " << strerror(errno) << std::endl;
    } else {
        std::cout << "Streaming stopped." << std::endl;
    }

    is_streaming_ = false;
}

CapturedFrame V4L2Camera::capture_frame(bool show_timing) {
    auto capture_start_time = std::chrono::high_resolution_clock::now();

    // 직접 폴링 방식: select() 없이 바로 DQBUF 시도
    struct v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    // EAGAIN이 발생할 때까지 계속 시도
    int retry_count = 0;
    const int max_retries = 1000;  // 최대 재시도 횟수
    
    while (retry_count < max_retries) {
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) == 0) {
            // 성공한 경우
            auto dqbuf_end_time = std::chrono::high_resolution_clock::now();
            if (show_timing) {
                auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(dqbuf_end_time - capture_start_time).count();
                std::cout << "Timing: direct polling VIDIOC_DQBUF took " << total_duration << " us (retries: " << retry_count << ")." << std::endl;
            }
            return {buffers_[buf.index].start, buf.bytesused, (int)buf.index, show_timing};
        }
        
        if (errno != EAGAIN) {
            // EAGAIN이 아닌 다른 오류
            throw std::runtime_error("Failed to dequeue buffer: " + std::string(strerror(errno)));
        }
        
        retry_count++;
        
        // 짧은 대기 (CPU 부하 감소)
        if (retry_count % 100 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));  // 1마이크로초 대기
        }
    }
    
    throw std::runtime_error("Polling timeout after " + std::to_string(max_retries) + " retries");
}

void V4L2Camera::release_frame(const CapturedFrame& frame) {
    struct v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = frame.buffer_index;

    if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
        throw std::runtime_error("Failed to re-queue buffer (VIDIOC_QBUF)");
    }
}

// MJPEG 프레임을 OpenCV Mat으로 디코딩하는 함수
cv::Mat decode_mjpeg_frame(const CapturedFrame& frame) {
    try {
        // MJPEG 데이터를 OpenCV Mat으로 변환
        std::vector<uchar> jpeg_data(static_cast<uchar*>(frame.data), 
                                   static_cast<uchar*>(frame.data) + frame.size);
        
        // JPEG 디코딩
        cv::Mat decoded_image = cv::imdecode(jpeg_data, cv::IMREAD_COLOR);
        
        if (decoded_image.empty()) {
            std::cerr << "Warning: Failed to decode MJPEG frame" << std::endl;
            return cv::Mat();
        }
        
        return decoded_image;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in decode_mjpeg_frame: " << e.what() << std::endl;
        return cv::Mat();
    } catch (const std::exception& e) {
        std::cerr << "Exception in decode_mjpeg_frame: " << e.what() << std::endl;
        return cv::Mat();
    }
}

// YUYV 프레임을 OpenCV Mat으로 변환하는 함수
cv::Mat convert_yuyv_frame(const CapturedFrame& frame, int width, int height) {
    try {
        // YUYV 데이터를 OpenCV Mat으로 변환
        cv::Mat yuyv(height, width, CV_8UC2, frame.data);
        cv::Mat bgr;
        cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUYV);
        return bgr;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception in convert_yuyv_frame: " << e.what() << std::endl;
        return cv::Mat();
    } catch (const std::exception& e) {
        std::cerr << "Exception in convert_yuyv_frame: " << e.what() << std::endl;
        return cv::Mat();
    }
}

// 프레임을 적절한 형식으로 변환하는 함수
cv::Mat convert_frame_to_mat(const CapturedFrame& frame, uint32_t format, int width, int height) {
    if (format == v4l2_fourcc('J', 'P', 'E', 'G') || format == v4l2_fourcc('M', 'J', 'P', 'G')) {
        return decode_mjpeg_frame(frame);
    } else if (format == v4l2_fourcc('Y', 'U', 'Y', 'V')) {
        return convert_yuyv_frame(frame, width, height);
    } else {
        std::cerr << "Unsupported format for display: 0x" << std::hex << format << std::dec << std::endl;
        return cv::Mat();
    }
}

int main() {
    const std::string device = "/dev/video0";
    bool use_dmabuf = true;  // DMA-BUF 사용 시도
    
    std::cout << "Starting V4L2 Camera Test with DIRECT POLLING + OpenCV Display..." << std::endl;
    
    try {
        std::cout << "Attempting to use " << (use_dmabuf ? "DMA-BUF" : "MMAP") << " for buffer management..." << std::endl;
        std::cout.flush();
        
        // DMA-BUF 지원 시도, 실패 시 MMAP으로 폴백
        V4L2Camera camera(device, 0, 0, 0, use_dmabuf);  // 크기와 형식은 현재 설정 사용
        
        std::cout << "Using " << (camera.is_using_dmabuf() ? "DMA-BUF" : "MMAP") << " for buffer management." << std::endl;
        
        camera.start_streaming();
        
        std::cout << "Camera streaming started. Starting real-time display with OpenCV..." << std::endl;
        std::cout << "Press 'q' to quit, 's' to save frame" << std::endl;
        std::cout.flush();
        
        // OpenCV 윈도우 생성
        const std::string window_name = "V4L2 Camera Live Feed";
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        
        // FPS 측정을 위한 변수들
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_print_time = start_time;
        int total_frames = 0;
        int display_frames = 0;
        int saved_frames = 0;
        const int timing_frames = 50;  // 50프레임마다 FPS 출력
        
        while (true) {
            try {
                auto frame_start = std::chrono::high_resolution_clock::now();
                
                // 프레임 캡처
                bool show_detailed_timing = (total_frames < 3); // 처음 3프레임만 상세 타이밍
                CapturedFrame frame = camera.capture_frame(show_detailed_timing);
                total_frames++;
                
                auto capture_end = std::chrono::high_resolution_clock::now();
                
                // 프레임을 OpenCV Mat으로 변환
                auto convert_start = std::chrono::high_resolution_clock::now();
                cv::Mat display_image = convert_frame_to_mat(frame, camera.get_format(), 
                                                           camera.get_width(), camera.get_height());
                auto convert_end = std::chrono::high_resolution_clock::now();
                
                // 프레임 반납 (빠른 반납을 위해 디스플레이 전에 수행)
                auto release_start = std::chrono::high_resolution_clock::now();
                camera.release_frame(frame);
                auto release_end = std::chrono::high_resolution_clock::now();
                
                // 화면에 표시
                auto display_start = std::chrono::high_resolution_clock::now();
                if (!display_image.empty()) {
                    // 디스플레이 크기 조정 (선택적)
                    cv::Mat resized_image;
                    if (display_image.cols > 1280 || display_image.rows > 720) {
                        // 큰 해상도는 축소해서 표시 (성능 향상)
                        double scale = std::min(1280.0 / display_image.cols, 720.0 / display_image.rows);
                        cv::resize(display_image, resized_image, cv::Size(), scale, scale, cv::INTER_LINEAR);
                        cv::imshow(window_name, resized_image);
                    } else {
                        cv::imshow(window_name, display_image);
                    }
                    display_frames++;
                } else {
                    std::cerr << "Warning: Empty frame, skipping display" << std::endl;
                }
                auto display_end = std::chrono::high_resolution_clock::now();
                
                // 키 입력 처리 (1ms 대기)
                auto key_start = std::chrono::high_resolution_clock::now();
                int key = cv::waitKey(1) & 0xFF;
                auto key_end = std::chrono::high_resolution_clock::now();
                
                if (key == 'q' || key == 27) {  // 'q' 또는 ESC 키
                    std::cout << "User requested exit." << std::endl;
                    break;
                } else if (key == 's' && !display_image.empty()) {  // 's' 키로 프레임 저장
                    std::string filename = "saved_frame_" + std::to_string(saved_frames++) + ".jpg";
                    cv::imwrite(filename, display_image);
                    std::cout << "Frame saved as: " << filename << std::endl;
                }
                
                // 상세한 타이밍 정보 (처음 3프레임만)
                if (total_frames <= 3) {
                    auto capture_time = std::chrono::duration_cast<std::chrono::microseconds>(capture_end - frame_start).count();
                    auto convert_time = std::chrono::duration_cast<std::chrono::microseconds>(convert_end - convert_start).count();
                    auto release_time = std::chrono::duration_cast<std::chrono::microseconds>(release_end - release_start).count();
                    auto display_time = std::chrono::duration_cast<std::chrono::microseconds>(display_end - display_start).count();
                    auto key_time = std::chrono::duration_cast<std::chrono::microseconds>(key_end - key_start).count();
                    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(key_end - frame_start).count();
                    
                    std::cout << "Frame " << total_frames << " timing breakdown:" << std::endl;
                    std::cout << "  Capture: " << capture_time << " us" << std::endl;
                    std::cout << "  Convert: " << convert_time << " us" << std::endl;
                    std::cout << "  Release: " << release_time << " us" << std::endl;
                    std::cout << "  Display: " << display_time << " us" << std::endl;
                    std::cout << "  Key check: " << key_time << " us" << std::endl;
                    std::cout << "  Total: " << total_time << " us (" << (1000000.0 / total_time) << " FPS potential)" << std::endl;
                }
                
                // 주기적 FPS 정보 출력
                if (total_frames % timing_frames == 0 && total_frames > 3) {
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> elapsed = now - start_time;
                    double capture_fps = total_frames / elapsed.count();
                    double display_fps = display_frames / elapsed.count();
                    std::cout << std::fixed << std::setprecision(2) 
                              << "Elapsed: " << elapsed.count() << "s, "
                              << "Capture FPS: " << capture_fps << ", "
                              << "Display FPS: " << display_fps << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::string error_msg = e.what();
                
                // EAGAIN이나 timeout 오류는 조용히 처리
                if (error_msg.find("EAGAIN") != std::string::npos || 
                    error_msg.find("No frame available") != std::string::npos ||
                    error_msg.find("Frame capture timeout") != std::string::npos ||
                    error_msg.find("timeout") != std::string::npos ||
                    error_msg.find("Polling timeout") != std::string::npos) {
                    continue;
                }
                
                std::cerr << "Error during frame processing: " << error_msg << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        // 정리
        cv::destroyAllWindows();
        std::cout << "OpenCV windows closed." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        cv::destroyAllWindows();
    }
    
    return 0;
}

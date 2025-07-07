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
#include <linux/dma-heap.h>  // DMA-BUF 할당을 위한 헤더 추가

// V4L2 인코더 헤더가 있는 경우에만 포함
#ifdef __has_include
    #if __has_include(<linux/v4l2-encoder.h>)
        #include <linux/v4l2-encoder.h>
        #define HAS_V4L2_ENCODER
    #endif
#endif

// DMA-BUF 할당을 위한 헬퍼 함수
int dma_buf_alloc(size_t size) {
    int heap_fd = open("/dev/dma_heap/linux,cma", O_RDWR | O_CLOEXEC);
    if (heap_fd < 0) {
        throw std::system_error(errno, std::generic_category(), "Failed to open DMA heap");
    }

    struct dma_heap_allocation_data alloc = {};
    alloc.len = size;
    alloc.fd_flags = O_RDWR | O_CLOEXEC;

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
        close(heap_fd);
        throw std::system_error(errno, std::generic_category(), "DMA_HEAP_IOCTL_ALLOC failed");
    }

    close(heap_fd);
    return alloc.fd;
}

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
        
        // 버퍼 개수 설정
        struct v4l2_control ctrl = {};
        ctrl.id = V4L2_CID_MIN_BUFFERS_FOR_OUTPUT;
        ctrl.value = 4;
        if (ioctl(fd_, VIDIOC_S_CTRL, &ctrl) == -1) {
            std::cerr << "WARNING: Failed to set buffer count: " << strerror(errno) << std::endl;
        }

        // 버퍼 크기 강제 설정 (4MB)
        // fmt.fmt.pix.sizeimage = 4 * 1024 * 1024;  // 이 라인을 나중에 처리



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

        // H.264 비트레이트 설정
        struct v4l2_control bitrate_control = {};
        bitrate_control.id = V4L2_CID_MPEG_VIDEO_BITRATE;
        bitrate_control.value = 1000000; // 1Mbps
        if (ioctl(fd_, VIDIOC_S_CTRL, &bitrate_control) == -1) {
            std::cerr << "Failed to set bitrate: " << strerror(errno) << std::endl;
        }

        // H.264 프로파일 설정
        struct v4l2_control profile_control = {};
        profile_control.id = V4L2_CID_MPEG_VIDEO_H264_PROFILE;
        profile_control.value = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
        if (ioctl(fd_, VIDIOC_S_CTRL, &profile_control) == -1) {
            std::cerr << "Failed to set H.264 profile: " << strerror(errno) << std::endl;
        }

        // 1920x1080 해상도와 H.264 포맷 설정 시도
        struct v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        // 현재 포맷을 기본으로 가져오기
        std::cout << "Getting default format..." << std::endl;
        if (ioctl(fd_, VIDIOC_G_FMT, &fmt) == -1) {
            throw std::runtime_error("Failed to get format");
        }

        // 1920x1080 해상도와 H.264 포맷 설정
        fmt.fmt.pix.width = 1920;
        fmt.fmt.pix.height = 1080;
        fmt.fmt.pix.pixelformat = v4l2_fourcc('H', '2', '6', '4');
        fmt.fmt.pix.sizeimage = 1920 * 1080 * 2;  // 적절한 버퍼 크기 설정

        std::cout << "\nTrying to set 1920x1080 resolution with H.264 format..." << std::endl;

        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
            std::cerr << "H.264 format failed, falling back to YUYV: " << strerror(errno) << std::endl;
            // H.264 실패 시 YUYV로 폴백
            fmt.fmt.pix.pixelformat = v4l2_fourcc('Y', 'U', 'Y', 'V');
            fmt.fmt.pix.sizeimage = fmt.fmt.pix.width * fmt.fmt.pix.height * 2;  // YUYV 크기
            
            if (ioctl(fd_, VIDIOC_S_FMT, &fmt) == -1) {
                std::cerr << "YUYV format also failed: " << strerror(errno) << std::endl;
                // 실패 시 현재 설정으로 계속 진행
                if (ioctl(fd_, VIDIOC_G_FMT, &fmt) == -1) {
                    throw std::runtime_error("Failed to get format after S_FMT failure");
                }
            } else {
                std::cout << "Successfully set to 1920x1080 with YUYV (fallback)!" << std::endl;
            }
        } else {
            std::cout << "Successfully set to 1920x1080 with H.264!" << std::endl;
        }

        // 실제 설정된 값으로 업데이트
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
        format_ = fmt.fmt.pix.pixelformat;
        bytes_per_line_ = fmt.fmt.pix.bytesperline;
        buffer_size_ = fmt.fmt.pix.sizeimage;

        std::cout << "Final format: " << width_ << "x" << height_ << std::endl;
        std::cout << "Format: 0x" << std::hex << format_ << std::dec;
        print_fourcc_format(format_);
        std::cout << "Bytes per line: " << bytes_per_line_ << std::endl;
        std::cout << "Buffer size: " << fmt.fmt.pix.sizeimage << " bytes" << std::endl;
        std::cout << "Actual Buffer size: " << buffer_size_ << " bytes" << std::endl;
    }

    void init_mmap() {
        struct v4l2_requestbuffers req = {};
        req.count = 4;  // 버퍼 개수 증가 (성능 향상)
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1) {
            std::cerr << "VIDIOC_REQBUFS failed: " << strerror(errno) << std::endl;
            throw std::runtime_error("Failed to request buffers");
        }

        if (req.count < 2) {
            throw std::runtime_error("Insufficient buffer memory (got " + std::to_string(req.count) + " buffers)");
        }

        std::cout << "Allocated " << req.count << " MMAP buffers" << std::endl;
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
            
            std::cout << "Buffer " << i << " mapped: length=" << buf.length 
                      << " offset=" << buf.m.offset << std::endl;
        }
        std::cout << buffers_.size() << " MMAP buffers mapped successfully." << std::endl;
    }

    void init_dmabuf() {
        std::cout << "Initializing DMA-BUF..." << std::endl;
        
        struct v4l2_requestbuffers req = {};
        req.count = 4;  // 버퍼 개수 증가
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_DMABUF;
        
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) == -1) {
            std::cerr << "DMA-BUF not supported: " << strerror(errno) << std::endl;
            throw std::runtime_error("DMA-BUF not supported");
        }

        if (req.count < 2) {
            throw std::runtime_error("Insufficient DMA-BUF buffer memory");
        }

        std::cout << "DMA-BUF: Allocated " << req.count << " buffers" << std::endl;
        buffers_.resize(req.count);

        // 버퍼 크기 확인
        struct v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_G_FMT, &fmt) == -1) {
            throw std::runtime_error("Failed to get format for buffer size");
        }
        size_t buffer_size = std::max(static_cast<size_t>(4 * 1024 * 1024), buffer_size_);

        // DMA 버퍼 할당
        for (size_t i = 0; i < buffers_.size(); ++i) {
            try {
                int dma_fd = dma_buf_alloc(buffer_size);
                buffers_[i].dmabuf_fd = dma_fd;
                buffers_[i].length = buffer_size;
                buffers_[i].start = nullptr; // DMA-BUF는 mmap하지 않음

                // 버퍼를 V4L2에 등록
                struct v4l2_buffer buf = {};
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory = V4L2_MEMORY_DMABUF;
                buf.index = i;
                buf.m.fd = dma_fd;
                buf.length = buffer_size;

                std::cout << "Queuing DMA-BUF buffer " << i << ": fd=" << dma_fd 
                          << ", size=" << buffer_size << std::endl;

                if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
                    std::cerr << "VIDIOC_QBUF failed for DMA-BUF buffer " << i 
                              << ": " << strerror(errno) << " (errno=" << errno << ")" << std::endl;
                    close(dma_fd);
                    throw std::runtime_error("Failed to queue DMA-BUF buffer " + std::to_string(i));
                }
                std::cout << "DMA-BUF buffer " << i << " allocated and queued successfully, size=" 
                          << buffer_size << std::endl;
            } catch (const std::exception& e) {
                // 이미 할당된 버퍼 정리
                for (size_t j = 0; j < i; ++j) {
                    close(buffers_[j].dmabuf_fd);
                }
                throw;
            }
        }
    }

    void unmap_buffers() {
        for (auto& buffer : buffers_) {
            if (buffer.start && buffer.start != MAP_FAILED) {
                munmap(buffer.start, buffer.length);
            }
            if (use_dmabuf_ && buffer.dmabuf_fd != -1) {
                close(buffer.dmabuf_fd);
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
        int dmabuf_fd = -1;  // DMA-BUF 파일 디스크립터
    };

    std::string device_path_;
    int width_, height_;
    int bytes_per_line_;
    uint32_t format_;
    int fd_;
    bool is_streaming_;
    bool use_dmabuf_;
    std::vector<Buffer> buffers_;
    size_t buffer_size_;
};

V4L2Camera::V4L2Camera(const std::string& device_path, int width, int height, uint32_t format, bool use_dmabuf)
    : device_path_(device_path), width_(width), height_(height), format_(format), fd_(-1), 
      is_streaming_(false), use_dmabuf_(use_dmabuf) {
    try {
        open_device();
        init_device();
        
        if (use_dmabuf_) {
            try {
                init_dmabuf();
            } catch (const std::exception& e) {
                std::cerr << "DMA-BUF initialization failed: " << e.what() 
                          << ", falling back to MMAP" << std::endl;
                use_dmabuf_ = false;
                init_mmap();
            }
        } else {
            init_mmap();
        }
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

    if (format_ == v4l2_fourcc('H','2','6','4')) {
#ifdef HAS_V4L2_ENCODER
        struct v4l2_encoder_cmd enc_cmd = {};
        enc_cmd.cmd = V4L2_ENC_CMD_START;
        enc_cmd.flags = 0;
        
        if (ioctl(fd_, VIDIOC_ENCODER_CMD, &enc_cmd) == -1) {
            std::cerr << "WARNING: Encoder start failed: " << strerror(errno) << std::endl;
        } else {
            std::cout << "Encoder start command sent successfully" << std::endl;
        }
#else
        std::cout << "H.264 encoder commands not available (v4l2-encoder.h not found)" << std::endl;
#endif
    }



    // DMA-BUF는 이미 초기화 시점에 버퍼가 큐에 있음
    if (!use_dmabuf_) {
        for (size_t i = 0; i < buffers_.size(); ++i) {
            struct v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
                throw std::runtime_error("Failed to queue buffer " + std::to_string(i));
            }
        }
    }

    std::cout << "Starting streaming with " << buffers_.size() << " buffers (" 
              << (use_dmabuf_ ? "DMA-BUF" : "MMAP") << " mode)..." << std::endl;

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    
    // 스트리밍 시작 전 버퍼 상태 확인
    std::cout << "About to call VIDIOC_STREAMON..." << std::endl;
    
    if (ioctl(fd_, VIDIOC_STREAMON, &type) == -1) {
        std::string err = strerror(errno);
        std::cerr << "VIDIOC_STREAMON failed: " << err << " (errno=" << errno << ")" << std::endl;

        // 추가 디버깅 정보
        std::cerr << "Current process UID: " << getuid() << ", GID: " << getgid() << std::endl;
        std::cerr << "Buffer management mode: " << (use_dmabuf_ ? "DMA-BUF" : "MMAP") << std::endl;
        std::cerr << "Number of buffers: " << buffers_.size() << std::endl;

        struct v4l2_capability cap = {};
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) == 0) {
            std::cerr << "Driver status: " << cap.driver << " v" 
                      << ((cap.version >> 16) & 0xFF) << "."
                      << ((cap.version >> 8) & 0xFF) << "."
                      << (cap.version & 0xFF) << std::endl;
        }
        
        // 버퍼 상태 확인
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (use_dmabuf_) {
                std::cerr << "DMA-BUF buffer " << i << ": fd=" << buffers_[i].dmabuf_fd 
                          << ", size=" << buffers_[i].length << std::endl;
            }
        }
        
        throw std::runtime_error("Failed to start streaming (VIDIOC_STREAMON)");
    }

    is_streaming_ = true;
    std::cout << "Streaming started (" << (use_dmabuf_ ? "DMA-BUF" : "MMAP") << " mode)" << std::endl;
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

    struct v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = use_dmabuf_ ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;

    // EAGAIN이 발생할 때까지 계속 시도
    int retry_count = 0;
    const int max_retries = 1000;
    
    while (retry_count < max_retries) {
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) == 0) {
            auto dqbuf_end_time = std::chrono::high_resolution_clock::now();
            if (show_timing) {
                auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    dqbuf_end_time - capture_start_time).count();
                std::cout << "Timing: VIDIOC_DQBUF took " << total_duration 
                          << " us (retries: " << retry_count << ")." << std::endl;
            }
            
            // DMA-BUF인 경우 mmap을 통해 데이터 접근
            void* frame_data = nullptr;
            if (use_dmabuf_) {
                if (buffers_[buf.index].start == nullptr) {
                    buffers_[buf.index].start = mmap(NULL, buf.bytesused, PROT_READ, 
                                                    MAP_SHARED, buffers_[buf.index].dmabuf_fd, 0);
                    if (buffers_[buf.index].start == MAP_FAILED) {
                        throw std::system_error(errno, std::generic_category(), "mmap for DMA-BUF");
                    }
                }
                frame_data = buffers_[buf.index].start;
            } else {
                frame_data = buffers_[buf.index].start;
            }
            
            return {frame_data, buf.bytesused, (int)buf.index, show_timing};
        }
        
        if (errno != EAGAIN) {
            throw std::runtime_error("Failed to dequeue buffer: " + std::string(strerror(errno)));
        }
        
        retry_count++;
        
        // 짧은 대기 (CPU 부하 감소)
        if (retry_count % 100 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    
    throw std::runtime_error("Polling timeout after " + std::to_string(max_retries) + " retries");
}

void V4L2Camera::release_frame(const CapturedFrame& frame) {
    struct v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = use_dmabuf_ ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;
    buf.index = frame.buffer_index;

    // DMA-BUF인 경우 fd 설정
    if (use_dmabuf_) {
        buf.m.fd = buffers_[frame.buffer_index].dmabuf_fd;
    }

    if (ioctl(fd_, VIDIOC_QBUF, &buf) == -1) {
        throw std::runtime_error("Failed to re-queue buffer (VIDIOC_QBUF)");
    }
}

// H.264 프레임을 파일로 저장하는 함수
void save_h264_frame(const CapturedFrame& frame, int frame_count) {
    std::string filename = "frame_" + std::to_string(frame_count) + ".h264";
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write((char*)frame.data, frame.size);
        file.close();
        std::cout << "H.264 frame " << frame_count << " saved (" << frame.size << " bytes)" << std::endl;
    } else {
        std::cerr << "Failed to save H.264 frame " << frame_count << std::endl;
    }
}

int main() {
    const std::string device = "/dev/video0";
    bool use_dmabuf = false;  // MMAP 모드로 먼저 테스트
    
    std::cout << "Starting V4L2 Camera Test with H.264 FORMAT BENCHMARK..." << std::endl;
    
    try {
        std::cout << "Attempting to use " << (use_dmabuf ? "DMA-BUF" : "MMAP") << " for buffer management..." << std::endl;
        
        V4L2Camera camera(device, 0, 0, 0, use_dmabuf);
        std::cout << "Using " << (camera.is_using_dmabuf() ? "DMA-BUF" : "MMAP") << " for buffer management." << std::endl;
        
        // 실제 설정된 포맷 확인
        uint32_t format = camera.get_format();
        bool is_h264 = (format == v4l2_fourcc('H', '2', '6', '4'));
        bool is_yuv420 = (format == v4l2_fourcc('Y', 'U', '1', '2'));
        bool is_mjpeg = (format == v4l2_fourcc('J', 'P', 'E', 'G') || format == v4l2_fourcc('M', 'J', 'P', 'G'));
        bool is_yuyv = (format == v4l2_fourcc('Y', 'U', 'Y', 'V'));
        
        std::cout << "\n=== FORMAT ANALYSIS ===" << std::endl;
        std::cout << "Final format: 0x" << std::hex << format << std::dec << std::endl;
        if (is_h264) {
            std::cout << "✓ Successfully using H.264 format!" << std::endl;
        } else if (is_yuv420) {
            std::cout << "⚠ Fallback to YUV420 format" << std::endl;
        } else if (is_mjpeg) {
            std::cout << "⚠ Fallback to MJPEG format" << std::endl;
        } else if (is_yuyv) {
            std::cout << "⚠ Fallback to YUYV format" << std::endl;
        } else {
            std::cout << "⚠ Unknown format" << std::endl;
        }
        std::cout << "Resolution: " << camera.get_width() << "x" << camera.get_height() << std::endl;
        std::cout << "========================" << std::endl;
        
        camera.start_streaming();
        
        std::cout << "Camera streaming started. Starting FPS measurement..." << std::endl;
        
        // FPS 측정을 위한 변수들
        auto start_time = std::chrono::high_resolution_clock::now();
        int total_frames = 0;
        int saved_frames = 0;
        const int max_save_frames = 5;
        const int timing_frames = 50;
        
        while (true) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = now - start_time;
            
            // 30초 후 종료
            if (elapsed.count() > 30.0) {
                std::cout << "Test completed after " << elapsed.count() << " seconds." << std::endl;
                break;
            }
            
            try {
                // 프레임 캡처
                bool show_detailed_timing = (total_frames < 5);
                CapturedFrame frame = camera.capture_frame(show_detailed_timing);
                total_frames++;
                
                // H.264 프레임 저장
                if (is_h264 && saved_frames < max_save_frames) {
                    save_h264_frame(frame, saved_frames);
                    saved_frames++;
                }
                
                // 프레임 반납
                camera.release_frame(frame);
                
                // 주기적 FPS 정보 출력
                if (total_frames % timing_frames == 0) {
                    double fps = total_frames / elapsed.count();
                    std::cout << std::fixed << std::setprecision(2) 
                              << "Elapsed time: " << elapsed.count() << "s, FPS: " << fps << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::string error_msg = e.what();
                if (error_msg.find("EAGAIN") != std::string::npos || 
                    error_msg.find("timeout") != std::string::npos) {
                    continue;  // 일시적 오류는 무시
                }
                std::cerr << "Error during frame processing: " << error_msg << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        // 최종 통계
        auto final_time = std::chrono::high_resolution_clock::now();
        double final_seconds = std::chrono::duration<double>(final_time - start_time).count();
        std::cout << "\n=== FINAL BENCHMARK RESULTS ===" << std::endl;
        std::cout << "Format used: " 
                  << (is_h264 ? "H.264" : 
                     (is_yuv420 ? "YUV420" : 
                     (is_mjpeg ? "MJPEG" : "YUYV"))) << std::endl;
        std::cout << "Total frames: " << total_frames << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << final_seconds << " seconds" << std::endl;
        std::cout << "Average FPS: " << (total_frames / final_seconds) << std::endl;
        if (is_h264) {
            std::cout << "H.264 frames saved: " << saved_frames << std::endl;
        }
        std::cout << "===============================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    
    return 0;
}
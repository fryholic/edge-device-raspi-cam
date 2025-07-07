#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <sys/mman.h>
#include <unordered_map>
#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>
#include <fcntl.h>       // open, O_RDWR, O_NONBLOCK
#include <unistd.h>     // close
#include <sys/ioctl.h>  // ioctl
#include <linux/videodev2.h> // V4L2 관련 구조체 및 상수

// FFmpeg includes for H.264 decoding
extern "C" {
#include <libavcodec    // H.264 및 기타 포맷 시도
    std::vector<PixelFormat> formatOptions;
    
    // libcamera에서 지원되는 다양한 H.264 관련 포맷들 시도
    formatOptions.push_back(formats::YUV420);  // H.264 인코딩에 많이 사용됨
    formatOptions.push_back(formats::RGB888);
    formatOptions.push_back(formats::BGR888);
    
    std::cout << "Will try formats in order: YUV420, RGB888, BGR888" << std::endl;lude <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using namespace libcamera;
using namespace std::chrono;

// 전역 변수들
static StreamConfiguration g_streamConfig;
static std::shared_ptr<Camera> g_camera;
static high_resolution_clock::time_point g_startTime;
static size_t g_frameCount = 0;
static const size_t g_targetFrames = 50;
static double g_totalProcessTime = 0;
static double g_totalFpsTime = 0;
static std::unordered_map<Request*, size_t> g_requestToBufferIndex;

// Update g_frameBuffers to store raw pointers
static std::vector<FrameBuffer*> g_frameBuffers;

// Add a new global vector to store mapped memory pointers
static std::vector<void*> g_mappedBuffers;

// FFmpeg encoder globals for H.264 encoding
static AVCodec *g_h264Encoder = nullptr;
static AVCodecContext *g_encoderContext = nullptr;
static AVFrame *g_encodeFrame = nullptr;
static AVPacket *g_encodePacket = nullptr;
static SwsContext *g_encodeSwsContext = nullptr;
static bool g_encoderInitialized = false;

// 전역 스트림 포인터 추가
static Stream *g_stream;
static bool g_stopping = false;  // 종료 플래그 추가

// 함수 선언
bool initH264Encoder();
void cleanupH264Encoder();
bool encodeFrameToH264AndSaveJpeg(const cv::Mat& frame, size_t frameIndex);

// 디버깅 로그 레벨 설정
void setLogLevelToDebug() {
    std::cout << "Setting log level to debug for detailed output..." << std::endl;
    int fd_ = open("/dev/video0", O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        std::cerr << "Failed to open video device: " << strerror(errno) << std::endl;
        return;
    }

    // ioctl 호출 제거 (지원되지 않는 경우)
    // struct v4l2_dbg_chip_info chip = {};
    // if (ioctl(fd_, VIDIOC_DBG_G_CHIP_INFO, &chip) == 0) {
    //     std::cout << "Chip identified: " << chip.name << std::endl;
    // } else {
    //     std::cerr << "Failed to get chip information: " << strerror(errno) << std::endl;
    // }

    close(fd_);
}

// 콜백 함수 (전역 함수)
void onRequestCompleted(Request *request) {
    // 종료 중이면 콜백 무시
    if (g_stopping) {
        return;
    }

    // 상태 확인
    if (request->status() != Request::RequestComplete) {
        if (!g_stopping) {  // 종료 중이 아닐 때만 오류 출력
            std::cerr << "Request failed: " << request->status() << std::endl;
        }
        return;
    }

    auto processStart = high_resolution_clock::now();

    // 버퍼 인덱스 가져오기
    auto it = g_requestToBufferIndex.find(request);
    if (it == g_requestToBufferIndex.end()) {
        std::cerr << "Request not found in buffer index map!" << std::endl;
        return;
    }
    
    size_t bufferIndex = it->second;

    // 버퍼 데이터 처리
    void *mappedData = g_mappedBuffers[bufferIndex];
    const FrameBuffer::Plane &plane = g_frameBuffers[bufferIndex]->planes().front();
    size_t dataSize = plane.length; // 프레임 버퍼 크기 사용
    
    if (dataSize == 0) {
        std::cerr << "No data in frame!" << std::endl;
        return;
    }

    std::cout << "Processing frame " << g_frameCount << " (size: " << dataSize << " bytes, format: " << g_streamConfig.pixelFormat.toString() << ")" << std::endl;

    // 프레임을 OpenCV Mat으로 변환
    cv::Mat frame;
    
    if (g_streamConfig.pixelFormat == formats::BGR888) {
        cv::Mat bgr_frame(
            g_streamConfig.size.height,
            g_streamConfig.size.width,
            CV_8UC3,
            mappedData,
            g_streamConfig.stride
        );
        cv::cvtColor(bgr_frame, frame, cv::COLOR_RGB2BGR);
    } else if (g_streamConfig.pixelFormat == formats::RGB888) {
        cv::Mat rgb_frame(
            g_streamConfig.size.height,
            g_streamConfig.size.width,
            CV_8UC3,
            mappedData,
            g_streamConfig.stride
        );
        cv::cvtColor(rgb_frame, frame, cv::COLOR_RGB2BGR);
    } else if (g_streamConfig.pixelFormat == formats::YUV420) {
        cv::Mat yuv_frame(
            g_streamConfig.size.height * 3 / 2,
            g_streamConfig.size.width,
            CV_8UC1,
            mappedData,
            g_streamConfig.stride
        );
        cv::cvtColor(yuv_frame, frame, cv::COLOR_YUV2BGR_I420);
    }
    
    // 프레임을 H.264로 인코딩하고 JPEG로도 저장 (첫 4프레임만)
    if (!frame.empty() && g_encoderInitialized) {
        bool success = encodeFrameToH264AndSaveJpeg(frame, g_frameCount);
        if (!success) {
            std::cout << "Failed to encode frame " << g_frameCount << std::endl;
        }
    }

    // 성능 측정
    auto processEnd = high_resolution_clock::now();
    double processTime = duration_cast<microseconds>(processEnd - processStart).count() / 1000.0;
    
    // FPS 계산
    auto elapsedTime = duration_cast<milliseconds>(processEnd - g_startTime);
    double elapsedTimeMs = elapsedTime.count();
    double fps = 0;
    if (elapsedTimeMs > g_totalFpsTime) {
        fps = 1000.0 / (elapsedTimeMs - g_totalFpsTime);
    }
    
    g_totalProcessTime += processTime;
    g_totalFpsTime = elapsedTimeMs;
    
    std::cout << "Frame " << g_frameCount 
              << " | Process: " << processTime << "ms"
              << " | Avg: " << g_totalProcessTime / (g_frameCount + 1) << "ms";
    
    if (fps > 0) {
        std::cout << " | FPS: " << fps;
    }
    std::cout << std::endl;

    // 다음 요청 준비
    request->reuse(Request::ReuseFlag::ReuseBuffers);

    // 디버깅 로그 추가
    std::cout << "Reused request with buffer " << bufferIndex << " successfully" << std::endl;

    // 종료 중이 아닐 때만 요청을 다시 큐에 추가
    if (!g_stopping && g_frameCount < g_targetFrames) {
        g_camera->queueRequest(request);
    }
    
    g_frameCount++;

    if (g_frameCount >= g_targetFrames && !g_stopping) {
        std::cout << "Target frame count reached, stopping camera..." << std::endl;
        g_stopping = true;  // 종료 플래그 설정
        // 콜백에서는 stop()을 호출하지 않음 - 메인 루프에서 처리
    }
}

bool initH264Encoder() {
    std::cout << "Initializing H.264 encoder..." << std::endl;
    
    // Find H.264 encoder
    g_h264Encoder = (AVCodec*)avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!g_h264Encoder) {
        std::cerr << "H.264 encoder not found!" << std::endl;
        return false;
    }
    
    // Create encoder context
    g_encoderContext = avcodec_alloc_context3(g_h264Encoder);
    if (!g_encoderContext) {
        std::cerr << "Failed to allocate encoder context!" << std::endl;
        return false;
    }
    
    // Set encoder parameters
    g_encoderContext->bit_rate = 4000000;  // 4 Mbps
    g_encoderContext->width = 1920;
    g_encoderContext->height = 1080;
    g_encoderContext->time_base = {1, 30};  // 30 FPS
    g_encoderContext->framerate = {30, 1};
    g_encoderContext->gop_size = 10;
    g_encoderContext->max_b_frames = 1;
    g_encoderContext->pix_fmt = AV_PIX_FMT_YUV420P;
    
    // Open encoder
    if (avcodec_open2(g_encoderContext, g_h264Encoder, nullptr) < 0) {
        std::cerr << "Failed to open H.264 encoder!" << std::endl;
        avcodec_free_context(&g_encoderContext);
        return false;
    }
    
    // Allocate frame and packet
    g_encodeFrame = av_frame_alloc();
    g_encodePacket = av_packet_alloc();
    
    if (!g_encodeFrame || !g_encodePacket) {
        std::cerr << "Failed to allocate encode frame or packet!" << std::endl;
        return false;
    }
    
    // Set frame parameters
    g_encodeFrame->format = g_encoderContext->pix_fmt;
    g_encodeFrame->width = g_encoderContext->width;
    g_encodeFrame->height = g_encoderContext->height;
    
    if (av_frame_get_buffer(g_encodeFrame, 32) < 0) {
        std::cerr << "Failed to allocate frame data!" << std::endl;
        return false;
    }
    
    std::cout << "H.264 encoder initialized successfully!" << std::endl;
    g_encoderInitialized = true;
    return true;
}

// FFmpeg 리소스 정리
void cleanupH264Encoder() {
    if (g_encodeSwsContext) {
        sws_freeContext(g_encodeSwsContext);
        g_encodeSwsContext = nullptr;
    }
    
    if (g_encodeFrame) {
        av_frame_free(&g_encodeFrame);
    }
    
    if (g_encodePacket) {
        av_packet_free(&g_encodePacket);
    }
    
    if (g_encoderContext) {
        avcodec_free_context(&g_encoderContext);
    }
    
    g_encoderInitialized = false;
    std::cout << "H.264 encoder cleanup completed." << std::endl;
}

// 프레임을 H.264로 인코딩하고 JPEG로도 저장
bool encodeFrameToH264AndSaveJpeg(const cv::Mat& frame, size_t frameIndex) {
    if (!g_encoderInitialized) {
        std::cerr << "Encoder not initialized!" << std::endl;
        return false;
    }
    
    // Create SWS context for color conversion
    if (!g_encodeSwsContext) {
        g_encodeSwsContext = sws_getContext(
            frame.cols, frame.rows, AV_PIX_FMT_BGR24,
            g_encoderContext->width, g_encoderContext->height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        
        if (!g_encodeSwsContext) {
            std::cerr << "Failed to create encode SWS context!" << std::endl;
            return false;
        }
    }
    
    // Make frame writable
    if (av_frame_make_writable(g_encodeFrame) < 0) {
        std::cerr << "Failed to make frame writable!" << std::endl;
        return false;
    }
    
    // Convert BGR to YUV420P
    const uint8_t* srcData[4] = {frame.data, nullptr, nullptr, nullptr};
    int srcLinesize[4] = {static_cast<int>(frame.step[0]), 0, 0, 0};
    
    sws_scale(g_encodeSwsContext, srcData, srcLinesize, 0, frame.rows,
              g_encodeFrame->data, g_encodeFrame->linesize);
    
    g_encodeFrame->pts = frameIndex;
    
    // Send frame to encoder
    int ret = avcodec_send_frame(g_encoderContext, g_encodeFrame);
    if (ret < 0) {
        std::cerr << "Error sending frame to encoder: " << ret << std::endl;
        return false;
    }
    
    // Receive encoded packet
    ret = avcodec_receive_packet(g_encoderContext, g_encodePacket);
    if (ret < 0) {
        if (ret == AVERROR(EAGAIN)) {
            // Need more frames
            return true;  // Not an error, just need more input
        }
        std::cerr << "Error receiving packet from encoder: " << ret << std::endl;
        return false;
    }
    
    // Save H.264 data to file
    std::string h264Filename = "libcamera_frame_" + std::to_string(frameIndex) + ".h264";
    std::ofstream h264File(h264Filename, std::ios::binary);
    if (h264File.is_open()) {
        h264File.write(reinterpret_cast<const char*>(g_encodePacket->data), g_encodePacket->size);
        h264File.close();
        std::cout << "Saved H.264 frame to " << h264Filename << std::endl;
    } else {
        std::cerr << "Failed to save H.264 frame to " << h264Filename << std::endl;
    }
    
    // Save JPEG (first 4 frames only)
    if (frameIndex < 4) {
        std::string jpegFilename = "libcamera_frame_" + std::to_string(frameIndex) + ".jpg";
        
        // JPEG 압축 품질 설정 (90% 품질로 4Mbps 상당의 고품질)
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(90);
        
        bool success = cv::imwrite(jpegFilename, frame, compression_params);
        if (success) {
            std::cout << "Saved high-quality JPEG to " << jpegFilename << std::endl;
        } else {
            std::cerr << "Failed to save JPEG to " << jpegFilename << std::endl;
        }
    }
    
    av_packet_unref(g_encodePacket);
    return true;
}

int main() {
    std::cout << "Starting libcamera zero-copy demo with H.264 encoding..." << std::endl;
    
    // FFmpeg 인코더 초기화
    if (!initH264Encoder()) {
        std::cerr << "Failed to initialize H.264 encoder!" << std::endl;
        return 1;
    }
    
    // 디버깅 로그 레벨 설정
    setLogLevelToDebug();

    // 카메라 매니저 초기화
    CameraManager *manager = new CameraManager();
    std::cout << "Created camera manager" << std::endl;
    
    if (manager->start()) {
        std::cerr << "Failed to start camera manager!" << std::endl;
        return 1;
    }
    std::cout << "Camera manager started successfully" << std::endl;

    // 카메라 선택
    if (manager->cameras().empty()) {
        std::cerr << "No cameras available!" << std::endl;
        manager->stop();
        return 1;
    }
    std::cout << "Found " << manager->cameras().size() << " camera(s)" << std::endl;
    
    g_camera = manager->cameras()[0];
    std::cout << "Selected camera: " << g_camera->id() << std::endl;
    
    if (g_camera->acquire()) {
        std::cerr << "Failed to acquire camera!" << std::endl;
        manager->stop();
        return 1;
    }
    std::cout << "Camera acquired successfully" << std::endl;

    // 720p 해상도 설정 -> 1080p 해상도 설정
    std::cout << "Generating camera configuration..." << std::endl;
    std::unique_ptr<CameraConfiguration> config = g_camera->generateConfiguration({StreamRole::Viewfinder});
    if (!config) {
        std::cerr << "Failed to generate configuration" << std::endl;
        g_camera->release();
        manager->stop();
        return 1;
    }
    std::cout << "Configuration generated successfully" << std::endl;

    StreamConfiguration &streamConfig = config->at(0);
    std::cout << "Default stream configuration: " << streamConfig.toString() << std::endl;
    
    // H.264 포맷 우선 시도, 지원되지 않으면 다른 포맷 사용
    std::vector<PixelFormat> formatOptions;
    
    // H.264 포맷을 문자열로 생성 시도
    try {
        PixelFormat h264Format = PixelFormat::fromString("H264");
        formatOptions.push_back(h264Format);
        std::cout << "Added H.264 format to options" << std::endl;
    } catch (...) {
        std::cout << "H.264 format creation failed, trying alternatives" << std::endl;
    }
    
    // 대안 포맷들
    formatOptions.push_back(formats::YUV420);
    formatOptions.push_back(formats::RGB888);
    formatOptions.push_back(formats::BGR888);
    
    bool formatSet = false;
    for (const auto& format : formatOptions) {
        std::cout << "Trying format: " << format.toString() << std::endl;
        streamConfig.pixelFormat = format;
        streamConfig.size = {1920, 1080};  // 1080p 해상도로 변경
        
        CameraConfiguration::Status status = config->validate();
        std::cout << "Validation status: " << (int)status << std::endl;
        
        if (status != CameraConfiguration::Invalid) {
            formatSet = true;
            std::cout << "Using pixel format: " << format.toString() << std::endl;
            std::cout << "Final size: " << streamConfig.size.width << "x" << streamConfig.size.height << std::endl;
            break;
        }
    }
    
    if (!formatSet) {
        std::cerr << "No supported pixel format found!" << std::endl;
        g_camera->release();
        manager->stop();
        return 1;
    }
    
    CameraConfiguration::Status status = config->validate();
    if (status == CameraConfiguration::Invalid) {
        std::cerr << "Camera configuration is invalid!" << std::endl;
        g_camera->release();
        manager->stop();
        return 1;
    }
    
    if (status == CameraConfiguration::Adjusted) {
        std::cout << "Camera configuration was adjusted:" << std::endl;
        std::cout << "  Adjusted: " << streamConfig.toString() << std::endl;
    }
    
    // 검증 후 실제 설정된 값을 전역 변수에 저장
    g_streamConfig = streamConfig;
    std::cout << "Configuring camera..." << std::endl;
    
    if (g_camera->configure(config.get()) < 0) {
        std::cerr << "Camera configuration failed!" << std::endl;
        g_camera->release();
        manager->stop();
        return 1;
    }

    std::cout << "Final stream configuration: " << g_streamConfig.toString() << std::endl;

    // 프레임 버퍼 할당 - 실제 설정된 스트림 사용
    FrameBufferAllocator *allocator = new FrameBufferAllocator(g_camera);
    
    // 설정된 스트림을 직접 사용 (복사본이 아닌 원본)
    Stream *stream = streamConfig.stream();
    
    // 카메라 설정 완료 후 전역 스트림 포인터 설정
    g_stream = stream;
    std::cout << "Global stream pointer set: " << g_stream << std::endl;
    std::cout << "Allocating buffers for stream: " << stream << std::endl;
    
    if (allocator->allocate(stream) < 0) {
        std::cerr << "Buffer allocation failed!" << std::endl;
        delete allocator;
        g_camera->release();
        manager->stop();
        return 1;
    }
    std::cout << "Buffer allocation successful!" << std::endl;

    // 메모리 매핑 준비
    const std::vector<std::unique_ptr<FrameBuffer>> &buffers = allocator->buffers(stream);
    std::cout << "Mapping " << buffers.size() << " buffers..." << std::endl;
    
    for (size_t i = 0; i < buffers.size(); ++i) {
        const FrameBuffer::Plane &plane = buffers[i]->planes().front();
        void *mappedData = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
        if (mappedData == MAP_FAILED) {
            std::cerr << "mmap failed for buffer " << i << std::endl;
            delete allocator;
            g_camera->release();
            manager->stop();
            return 1;
        }

        g_frameBuffers.push_back(buffers[i].get());
        g_mappedBuffers.push_back(mappedData); // Store the mapped pointer
        std::cout << "Buffer " << i << " mapped successfully (size: " << plane.length << " bytes)" << std::endl;
    }

    // 요청 생성 및 버퍼 연결
    std::vector<std::unique_ptr<Request>> requests;
    std::cout << "Creating requests..." << std::endl;
    
    for (size_t i = 0; i < buffers.size(); ++i) {
        std::unique_ptr<Request> request = g_camera->createRequest();
        if (!request) {
            std::cerr << "Failed to create request" << std::endl;
            delete allocator;
            g_camera->release();
            manager->stop();
            return 1;
        }

        // 디버깅 로그 추가: 요청 생성 확인
        std::cout << "Created request " << i << " successfully" << std::endl;

        if (request->addBuffer(stream, buffers[i].get())) {
            std::cerr << "Failed to add buffer to request for buffer " << i << std::endl;
            std::cerr << "Stream: " << stream << ", Buffer: " << buffers[i].get() << std::endl;
            delete allocator;
            g_camera->release();
            manager->stop();
            return 1;
        }

        // 디버깅 로그 추가: 버퍼 추가 확인
        std::cout << "Added buffer " << i << " to request successfully" << std::endl;

        g_requestToBufferIndex[request.get()] = i;
        requests.push_back(std::move(request));
        std::cout << "Request " << i << " created and configured" << std::endl;
    }

    // 카메라 시작
    std::cout << "Starting camera..." << std::endl;
    if (g_camera->start()) {
        std::cerr << "Failed to start camera!" << std::endl;
        delete allocator;
        g_camera->release();
        manager->stop();
        return 1;
    }
    std::cout << "Camera started successfully!" << std::endl;

    // 요청 완료 콜백 연결 (전역 함수 사용)
    std::cout << "Connecting callback..." << std::endl;
    g_camera->requestCompleted.connect(onRequestCompleted);
    std::cout << "Callback connected!" << std::endl;

    // 초기 요청 시작
    std::cout << "Queuing initial requests..." << std::endl;
    g_startTime = high_resolution_clock::now();
    for (auto &request : requests) {
        g_camera->queueRequest(request.get());
    }
    std::cout << "Initial requests queued, starting capture loop..." << std::endl;

    // 메인 루프
    while (g_frameCount < g_targetFrames && !g_stopping) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Capture completed, cleaning up..." << std::endl;
    
    // 종료 플래그 설정 (이미 설정되어 있지만 안전을 위해)
    g_stopping = true;
    
    // 콜백에서 새로운 요청이 추가되지 않도록 충분히 대기
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 콜백 분리
    std::cout << "Disconnecting callback..." << std::endl;
    g_camera->requestCompleted.disconnect(onRequestCompleted);

    // 추가 대기
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // 정리
    std::cout << "Stopping camera..." << std::endl;
    g_camera->stop();
    
    // 카메라 정지 후 추가 대기
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "Unmapping buffers..." << std::endl;
    std::cout << "Unmapping buffers..." << std::endl;
    for (size_t i = 0; i < g_mappedBuffers.size(); ++i) {
        if (g_mappedBuffers[i] != nullptr) {
            munmap(g_mappedBuffers[i], allocator->buffers(stream)[i]->planes().front().length);
        }
    }
    
    std::cout << "Cleaning up allocator..." << std::endl;
    delete allocator;
    std::cout << "Releasing camera..." << std::endl;
    g_camera->release();
    std::cout << "Stopping manager..." << std::endl;
    manager->stop();
    
    // FFmpeg 인코더 정리
    cleanupH264Encoder();
    
    // cv::destroyAllWindows(); // GUI 미사용으로 주석 처리

    std::cout << "Demo completed successfully!" << std::endl;
    return 0;
}
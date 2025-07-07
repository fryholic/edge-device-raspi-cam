#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <sys/mman.h>
#include <unordered_map>
#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

// FFmpeg includes for H.264 decoding (disabled)
// extern "C" {
// #include <libavcodec/avcodec.h>
// #include <libavformat/avformat.h>
// #include <libavutil/imgutils.h>
// #include <libswscale/swscale.h>
// }

// FOURCC macro if not defined
#ifndef FOURCC
#define FOURCC(a, b, c, d) ((uint32_t)(a) | ((uint32_t)(b) << 8) | ((uint32_t)(c) << 16) | ((uint32_t)(d) << 24))
#endif

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
static std::unordered_map<Request*, size_t> g_requestToBufferIndex; // 레거시 호환용

// Frame buffers (legacy for compatibility)
static std::vector<FrameBuffer*> g_frameBuffers;
static std::vector<void*> g_mappedBuffers;

// Stream configurations
static StreamConfiguration g_h264StreamConfig;
static StreamConfiguration g_yuvStreamConfig;

// FFmpeg decoder globals for H.264 decoding (disabled)
// static AVCodec *g_h264Decoder = nullptr;
// static AVCodecContext *g_decoderContext = nullptr;
// static AVFrame *g_frame = nullptr;
// static AVPacket *g_packet = nullptr;
// static SwsContext *g_swsContext = nullptr;
// static bool g_decoderInitialized = false;
// static size_t g_jpegFramesSaved = 0;

// 전역 스트림 포인터 및 종료 플래그
static Stream *g_h264Stream = nullptr;  // H.264 스트림
static Stream *g_yuvStream = nullptr;   // YUV 스트림 (옵션)
static bool g_stopping = false;
static bool g_hasH264Stream = false;
static bool g_hasYuvStream = false;

// 다중 스트림 지원을 위한 버퍼 관리
static std::vector<FrameBuffer*> g_h264FrameBuffers;
static std::vector<void*> g_h264MappedBuffers;
static std::vector<FrameBuffer*> g_yuvFrameBuffers;
static std::vector<void*> g_yuvMappedBuffers;
static std::unordered_map<Request*, std::vector<size_t>> g_requestToBufferIndices;

// V4L2 H.264 인코더 관련 변수들
static int g_encoderFd = -1;
static bool g_encoderInitialized = false;
static std::ofstream g_h264OutputFile;

// 함수 선언
bool initV4L2H264Encoder();
void cleanupV4L2H264Encoder();
bool encodeToH264(const uint8_t* yuvData, size_t dataSize, size_t frameIndex);

/*
// H.264 디코더 초기화 (비활성화)
bool initH264Decoder() {
    std::cout << "Initializing H.264 decoder..." << std::endl;
    
    // Find H.264 decoder
    g_h264Decoder = (AVCodec*)avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!g_h264Decoder) {
        std::cerr << "H.264 decoder not found!" << std::endl;
        return false;
    }
    
    // Create decoder context
    g_decoderContext = avcodec_alloc_context3(g_h264Decoder);
    if (!g_decoderContext) {
        std::cerr << "Failed to allocate decoder context!" << std::endl;
        return false;
    }
    
    // Open decoder
    if (avcodec_open2(g_decoderContext, g_h264Decoder, nullptr) < 0) {
        std::cerr << "Failed to open H.264 decoder!" << std::endl;
        avcodec_free_context(&g_decoderContext);
        return false;
    }
    
    // Allocate frame and packet
    g_frame = av_frame_alloc();
    g_packet = av_packet_alloc();
    
    if (!g_frame || !g_packet) {
        std::cerr << "Failed to allocate frame or packet!" << std::endl;
        return false;
    }
    
    std::cout << "H.264 decoder initialized successfully!" << std::endl;
    g_decoderInitialized = true;
    return true;
}

// FFmpeg 리소스 정리
void cleanupH264Decoder() {
    if (g_swsContext) {
        sws_freeContext(g_swsContext);
        g_swsContext = nullptr;
    }
    
    if (g_frame) {
        av_frame_free(&g_frame);
    }
    
    if (g_packet) {
        av_packet_free(&g_packet);
    }
    
    if (g_decoderContext) {
        avcodec_free_context(&g_decoderContext);
    }
    
    g_decoderInitialized = false;
    std::cout << "H.264 decoder cleanup completed." << std::endl;
}

// H.264 프레임을 디코딩하고 JPEG로 저장
bool decodeH264AndSaveJpeg(const uint8_t* h264Data, size_t dataSize, size_t frameIndex) {
    if (!g_decoderInitialized) {
        std::cerr << "Decoder not initialized!" << std::endl;
        return false;
    }
    
    // Set packet data
    g_packet->data = (uint8_t*)h264Data;
    g_packet->size = dataSize;
    
    // Send packet to decoder
    int ret = avcodec_send_packet(g_decoderContext, g_packet);
    if (ret < 0) {
        std::cerr << "Error sending packet to decoder: " << ret << std::endl;
        return false;
    }
    
    // Receive frame from decoder
    ret = avcodec_receive_frame(g_decoderContext, g_frame);
    if (ret < 0) {
        if (ret == AVERROR(EAGAIN)) {
            // Need more data
            return false;
        }
        std::cerr << "Error receiving frame from decoder: " << ret << std::endl;
        return false;
    }
    
    // Convert to BGR format for OpenCV
    if (!g_swsContext) {
        g_swsContext = sws_getContext(
            g_frame->width, g_frame->height, (AVPixelFormat)g_frame->format,
            g_frame->width, g_frame->height, AV_PIX_FMT_BGR24,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        
        if (!g_swsContext) {
            std::cerr << "Failed to create SWS context!" << std::endl;
            return false;
        }
    }
    
    // Allocate BGR frame
    int bgrLinesize[4];
    uint8_t* bgrData[4];
    int bgrBufferSize = av_image_alloc(bgrData, bgrLinesize, g_frame->width, g_frame->height, AV_PIX_FMT_BGR24, 32);
    
    if (bgrBufferSize < 0) {
        std::cerr << "Failed to allocate BGR buffer!" << std::endl;
        return false;
    }
    
    // Convert YUV to BGR
    sws_scale(g_swsContext, g_frame->data, g_frame->linesize, 0, g_frame->height, bgrData, bgrLinesize);
    
    // Create OpenCV Mat
    cv::Mat bgrFrame(g_frame->height, g_frame->width, CV_8UC3, bgrData[0], bgrLinesize[0]);
    
    // Save as JPEG with high quality
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(90);  // High quality
    
    std::string filename = "libcamera_frame_" + std::to_string(frameIndex) + ".jpg";
    bool success = cv::imwrite(filename, bgrFrame, compression_params);
    
    if (success) {
        std::cout << "Saved decoded H.264 frame to " << filename << std::endl;
        g_jpegFramesSaved++;
    } else {
        std::cerr << "Failed to save frame to " << filename << std::endl;
    }
    
    // Free BGR buffer
    av_freep(&bgrData[0]);
    
    return success;
}
*/

// V4L2 H.264 인코더 초기화
bool initV4L2H264Encoder() {
    std::cout << "Initializing V4L2 H.264 encoder..." << std::endl;
    
    // Raspberry Pi의 H.264 인코더 디바이스 열기
    const char* encoder_device = "/dev/video11"; // RPi H.264 encoder
    g_encoderFd = open(encoder_device, O_RDWR);
    if (g_encoderFd < 0) {
        std::cerr << "Failed to open H.264 encoder device: " << encoder_device << std::endl;
        return false;
    }
    
    // 인코더 capability 확인
    struct v4l2_capability cap;
    if (ioctl(g_encoderFd, VIDIOC_QUERYCAP, &cap) < 0) {
        std::cerr << "Failed to query encoder capabilities" << std::endl;
        close(g_encoderFd);
        g_encoderFd = -1;
        return false;
    }
    
    std::cout << "H.264 encoder device: " << cap.card << std::endl;
    std::cout << "Driver: " << cap.driver << std::endl;
    
    // H.264 출력 파일 열기
    g_h264OutputFile.open("libcamera_encoded.h264", std::ios::binary);
    if (!g_h264OutputFile.is_open()) {
        std::cerr << "Failed to open H.264 output file!" << std::endl;
        close(g_encoderFd);
        g_encoderFd = -1;
        return false;
    }
    
    std::cout << "V4L2 H.264 encoder initialized successfully!" << std::endl;
    std::cout << "H.264 output will be saved to: libcamera_encoded.h264" << std::endl;
    g_encoderInitialized = true;
    return true;
}

// V4L2 H.264 인코더 정리
void cleanupV4L2H264Encoder() {
    if (g_h264OutputFile.is_open()) {
        g_h264OutputFile.close();
        std::cout << "H.264 output file closed." << std::endl;
    }
    
    if (g_encoderFd >= 0) {
        close(g_encoderFd);
        g_encoderFd = -1;
    }
    g_encoderInitialized = false;
    std::cout << "V4L2 H.264 encoder cleanup completed." << std::endl;
}

// YUYV를 H.264로 인코딩 (단순 버전)
bool encodeToH264(const uint8_t* yuvData, size_t dataSize, size_t frameIndex) {
    if (!g_encoderInitialized) {
        return false;
    }
    
    // 현재는 단순히 원본 파일과 함께 인코더가 초기화되었다는 로그만 출력
    std::cout << "H.264 encoder ready for frame " << frameIndex << std::endl;
    return true;
}

// 콜백 함수
void onRequestCompleted(Request *request) {
    if (g_stopping) {
        return;
    }

    if (request->status() != Request::RequestComplete) {
        if (!g_stopping) {
            std::cerr << "Request failed: " << request->status() << std::endl;
        }
        return;
    }

    auto processStart = high_resolution_clock::now();

    std::cout << "Processing frame " << g_frameCount << std::endl;

    // H.264 스트림 처리
    if (g_hasH264Stream && g_h264Stream) {
        FrameBuffer *h264Buffer = request->findBuffer(g_h264Stream);
        if (h264Buffer) {
            const FrameBuffer::Plane &h264Plane = h264Buffer->planes().front();
            size_t h264DataSize = h264Plane.length;
            
            if (h264DataSize > 0) {
                // H.264 버퍼를 메모리에서 직접 읽기
                auto it = std::find(g_h264FrameBuffers.begin(), g_h264FrameBuffers.end(), h264Buffer);
                if (it != g_h264FrameBuffers.end()) {
                    size_t bufferIndex = std::distance(g_h264FrameBuffers.begin(), it);
                    void *h264Data = g_h264MappedBuffers[bufferIndex];
                    
                    // H.264 데이터를 파일에 직접 저장
                    if (g_h264OutputFile.is_open()) {
                        g_h264OutputFile.write(static_cast<const char*>(h264Data), h264DataSize);
                        std::cout << "Saved H.264 frame " << g_frameCount << " (size: " << h264DataSize << " bytes)" << std::endl;
                    }
                }
            }
        }
    }

    // YUV 스트림 처리 (있는 경우)
    if (g_hasYuvStream && g_yuvStream) {
        FrameBuffer *yuvBuffer = request->findBuffer(g_yuvStream);
        if (yuvBuffer) {
            const FrameBuffer::Plane &yuvPlane = yuvBuffer->planes().front();
            size_t yuvDataSize = yuvPlane.length;
            
            if (yuvDataSize > 0) {
                auto it = std::find(g_yuvFrameBuffers.begin(), g_yuvFrameBuffers.end(), yuvBuffer);
                if (it != g_yuvFrameBuffers.end()) {
                    size_t bufferIndex = std::distance(g_yuvFrameBuffers.begin(), it);
                    void *yuvData = g_yuvMappedBuffers[bufferIndex];
                    
                    // YUV 데이터를 파일로 저장 (프리뷰용)
                    std::string yuvFilename = "libcamera_frame_" + std::to_string(g_frameCount) + ".yuv420";
                    std::ofstream yuvFile(yuvFilename, std::ios::binary);
                    if (yuvFile.is_open()) {
                        yuvFile.write(static_cast<const char*>(yuvData), yuvDataSize);
                        yuvFile.close();
                        std::cout << "Saved YUV frame " << g_frameCount << " (size: " << yuvDataSize << " bytes)" << std::endl;
                    }
                }
            }
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

    // 종료 중이 아닐 때만 요청을 다시 큐에 추가
    if (!g_stopping && g_frameCount < g_targetFrames) {
        g_camera->queueRequest(request);
    }
    
    g_frameCount++;

    if (g_frameCount >= g_targetFrames && !g_stopping) {
        std::cout << "Target frame count reached, stopping camera..." << std::endl;
        g_stopping = true;
    }
}

int main() {
    std::cout << "Starting libcamera zero-copy demo with H.264 encoding..." << std::endl;
    
    // V4L2 H.264 인코더 초기화 시도
    if (initV4L2H264Encoder()) {
        std::cout << "V4L2 H.264 encoder available" << std::endl;
    } else {
        std::cout << "V4L2 H.264 encoder not available, continuing with raw format" << std::endl;
    }

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

    // 카메라 설정을 위한 다중 스트림 접근 (YUV + H.264)
    std::cout << "Generating camera configuration..." << std::endl;
    
    // 이중 스트림 설정: VideoRecording (H.264) + Viewfinder (YUV)
    std::unique_ptr<CameraConfiguration> config = g_camera->generateConfiguration({StreamRole::VideoRecording, StreamRole::Viewfinder});
    if (!config || config->size() < 2) {
        std::cout << "Dual stream configuration failed, trying single VideoRecording..." << std::endl;
        // VideoRecording 역할로 시도
        config = g_camera->generateConfiguration({StreamRole::VideoRecording});
        if (!config) {
            std::cout << "VideoRecording role failed, trying Viewfinder..." << std::endl;
            // Viewfinder 역할로 시도
            config = g_camera->generateConfiguration({StreamRole::Viewfinder});
            if (!config) {
                std::cerr << "Failed to generate configuration" << std::endl;
                g_camera->release();
                manager->stop();
                return 1;
            }
        }
    }
    std::cout << "Configuration generated successfully with " << config->size() << " stream(s)" << std::endl;

    // 각 스트림의 기본 설정 확인
    for (size_t i = 0; i < config->size(); ++i) {
        std::cout << "Stream " << i << " default: " << config->at(i).toString() << std::endl;
        
        // 사용 가능한 포맷들 확인
        const StreamFormats &formats = config->at(i).formats();
        std::cout << "Available pixel formats for stream " << i << ":" << std::endl;
        for (const auto& pixelFormat : formats.pixelformats()) {
            std::cout << "  - " << pixelFormat.toString() << std::endl;
        }
    }
    
    // 첫 번째 스트림을 H.264로 설정 (하드웨어 인코딩)
    StreamConfiguration &h264Config = config->at(0);
    const StreamFormats &h264Formats = h264Config.formats();
    
    // H.264 포맷 시도
    bool h264Available = false;
    for (const auto& pixelFormat : h264Formats.pixelformats()) {
        if (pixelFormat.toString() == "H264") {
            h264Config.pixelFormat = pixelFormat;
            h264Available = true;
            std::cout << "H.264 format found and set for stream 0" << std::endl;
            break;
        }
    }
    
    if (!h264Available) {
        std::cout << "H.264 format not available, using YUV420 for stream 0" << std::endl;
        h264Config.pixelFormat = formats::YUV420;
    }
    
    h264Config.size = {1920, 1080};
    std::cout << "Set stream 0 to " << h264Config.pixelFormat.toString() << " 1920x1080" << std::endl;
    
    // 두 번째 스트림이 있으면 YUV로 설정 (프리뷰용)
    if (config->size() > 1) {
        StreamConfiguration &yuvConfig = config->at(1);
        yuvConfig.pixelFormat = formats::YUV420;
        yuvConfig.size = {640, 480}; // 작은 크기로 설정
        std::cout << "Set stream 1 to YUV420 640x480 for preview" << std::endl;
    }
    
    // 설정 검증
    CameraConfiguration::Status status = config->validate();
    std::cout << "Configuration validation status: " << (int)status << std::endl;
    
    if (status == CameraConfiguration::Invalid) {
        std::cerr << "Configuration validation failed!" << std::endl;
        g_camera->release();
        manager->stop();
        return 1;
    }
    
    // 설정 적용
    g_h264StreamConfig = config->at(0);
    if (config->size() > 1) {
        g_yuvStreamConfig = config->at(1);
        g_hasYuvStream = true;
    }
    g_hasH264Stream = true;
    
    std::cout << "Configuring camera..." << std::endl;
    
    if (g_camera->configure(config.get()) < 0) {
        std::cerr << "Camera configuration failed!" << std::endl;
        g_camera->release();
        manager->stop();
        return 1;
    }

    std::cout << "Final H.264 stream configuration: " << g_h264StreamConfig.toString() << std::endl;
    if (g_hasYuvStream) {
        std::cout << "Final YUV stream configuration: " << g_yuvStreamConfig.toString() << std::endl;
    }

    // 프레임 버퍼 할당
    FrameBufferAllocator *allocator = new FrameBufferAllocator(g_camera);
    
    // H.264 스트림 버퍼 할당
    g_h264Stream = g_h264StreamConfig.stream();
    if (allocator->allocate(g_h264Stream) < 0) {
        std::cerr << "H.264 buffer allocation failed!" << std::endl;
        delete allocator;
        g_camera->release();
        manager->stop();
        return 1;
    }
    std::cout << "H.264 buffer allocation successful!" << std::endl;

    // YUV 스트림 버퍼 할당 (있는 경우)
    if (g_hasYuvStream) {
        g_yuvStream = g_yuvStreamConfig.stream();
        if (allocator->allocate(g_yuvStream) < 0) {
            std::cerr << "YUV buffer allocation failed!" << std::endl;
            delete allocator;
            g_camera->release();
            manager->stop();
            return 1;
        }
        std::cout << "YUV buffer allocation successful!" << std::endl;
    }

    // H.264 스트림 메모리 매핑
    const std::vector<std::unique_ptr<FrameBuffer>> &h264Buffers = allocator->buffers(g_h264Stream);
    std::cout << "Mapping " << h264Buffers.size() << " H.264 buffers..." << std::endl;
    
    for (size_t i = 0; i < h264Buffers.size(); ++i) {
        const FrameBuffer::Plane &plane = h264Buffers[i]->planes().front();
        void *mappedData = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
        if (mappedData == MAP_FAILED) {
            std::cerr << "mmap failed for H.264 buffer " << i << std::endl;
            delete allocator;
            g_camera->release();
            manager->stop();
            return 1;
        }

        g_h264FrameBuffers.push_back(h264Buffers[i].get());
        g_h264MappedBuffers.push_back(mappedData);
        std::cout << "H.264 buffer " << i << " mapped successfully (size: " << plane.length << " bytes)" << std::endl;
    }

    // YUV 스트림 메모리 매핑 (있는 경우)
    if (g_hasYuvStream) {
        const std::vector<std::unique_ptr<FrameBuffer>> &yuvBuffers = allocator->buffers(g_yuvStream);
        std::cout << "Mapping " << yuvBuffers.size() << " YUV buffers..." << std::endl;
        
        for (size_t i = 0; i < yuvBuffers.size(); ++i) {
            const FrameBuffer::Plane &plane = yuvBuffers[i]->planes().front();
            void *mappedData = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
            if (mappedData == MAP_FAILED) {
                std::cerr << "mmap failed for YUV buffer " << i << std::endl;
                delete allocator;
                g_camera->release();
                manager->stop();
                return 1;
            }

            g_yuvFrameBuffers.push_back(yuvBuffers[i].get());
            g_yuvMappedBuffers.push_back(mappedData);
            std::cout << "YUV buffer " << i << " mapped successfully (size: " << plane.length << " bytes)" << std::endl;
        }
    }

    // 요청 생성
    std::vector<std::unique_ptr<Request>> requests;
    std::cout << "Creating requests..." << std::endl;
    
    size_t numRequests = h264Buffers.size();
    for (size_t i = 0; i < numRequests; ++i) {
        std::unique_ptr<Request> request = g_camera->createRequest();
        if (!request) {
            std::cerr << "Failed to create request" << std::endl;
            delete allocator;
            g_camera->release();
            manager->stop();
            return 1;
        }

        // H.264 버퍼 추가
        if (request->addBuffer(g_h264Stream, h264Buffers[i].get())) {
            std::cerr << "Failed to add H.264 buffer to request for buffer " << i << std::endl;
            delete allocator;
            g_camera->release();
            manager->stop();
            return 1;
        }

        // YUV 버퍼 추가 (있는 경우)
        if (g_hasYuvStream && i < allocator->buffers(g_yuvStream).size()) {
            const std::vector<std::unique_ptr<FrameBuffer>> &yuvBuffers = allocator->buffers(g_yuvStream);
            if (request->addBuffer(g_yuvStream, yuvBuffers[i].get())) {
                std::cerr << "Failed to add YUV buffer to request for buffer " << i << std::endl;
                delete allocator;
                g_camera->release();
                manager->stop();
                return 1;
            }
        }

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

    // 콜백 연결
    g_camera->requestCompleted.connect(onRequestCompleted);

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
    
    g_stopping = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // 콜백 분리
    g_camera->requestCompleted.disconnect(onRequestCompleted);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // 정리
    g_camera->stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 버퍼 언매핑
    for (size_t i = 0; i < g_h264MappedBuffers.size(); ++i) {
        if (g_h264MappedBuffers[i] != nullptr) {
            munmap(g_h264MappedBuffers[i], allocator->buffers(g_h264Stream)[i]->planes().front().length);
        }
    }
    
    if (g_hasYuvStream) {
        for (size_t i = 0; i < g_yuvMappedBuffers.size(); ++i) {
            if (g_yuvMappedBuffers[i] != nullptr) {
                munmap(g_yuvMappedBuffers[i], allocator->buffers(g_yuvStream)[i]->planes().front().length);
            }
        }
    }
    
    delete allocator;
    g_camera->release();
    manager->stop();
    
    // V4L2 H.264 인코더 정리
    cleanupV4L2H264Encoder();

    std::cout << "Demo completed successfully!" << std::endl;
    return 0;
}

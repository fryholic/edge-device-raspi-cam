#include <iostream>
#include <chrono>
#include <thread>
#include <sys/mman.h>
#include <unordered_map>
#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>

using namespace libcamera;
using namespace std::chrono;

struct MappedBuffer {
    void *data;
    size_t size;
};

// 전역 변수들
static std::vector<MappedBuffer> g_mappedBuffers;
static StreamConfiguration g_streamConfig;
static std::shared_ptr<Camera> g_camera;
static high_resolution_clock::time_point g_startTime;
static size_t g_frameCount = 0;
static const size_t g_targetFrames = 50;
static double g_totalProcessTime = 0;
static double g_totalFpsTime = 0;
static std::unordered_map<Request*, size_t> g_requestToBufferIndex;

// 콜백 함수 (전역 함수)
void onRequestCompleted(Request *request) {
    // 상태 확인
    if (request->status() != Request::RequestComplete) {
        std::cerr << "Request failed: " << request->status() << std::endl;
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
    MappedBuffer &buf = g_mappedBuffers[bufferIndex];
    
    // OpenCV Mat 생성 (Zero-Copy)
    cv::Mat frame;
    
    if (g_streamConfig.pixelFormat == formats::BGR888) {
        frame = cv::Mat(
            g_streamConfig.size.height, 
            g_streamConfig.size.width, 
            CV_8UC3, 
            buf.data,
            g_streamConfig.stride
        );
    } else if (g_streamConfig.pixelFormat == formats::RGB888) {
        cv::Mat rgb_frame(
            g_streamConfig.size.height, 
            g_streamConfig.size.width, 
            CV_8UC3, 
            buf.data,
            g_streamConfig.stride
        );
        cv::cvtColor(rgb_frame, frame, cv::COLOR_RGB2BGR);
    } else if (g_streamConfig.pixelFormat == formats::YUV420) {
        cv::Mat yuv_frame(
            g_streamConfig.size.height * 3 / 2, 
            g_streamConfig.size.width, 
            CV_8UC1, 
            buf.data,
            g_streamConfig.stride
        );
        cv::cvtColor(yuv_frame, frame, cv::COLOR_YUV2BGR_I420);
    } else {
        std::cerr << "Unsupported pixel format in callback!" << std::endl;
        return;
    }
    
    // 프레임 처리
    if (!frame.empty()) {
        // GUI 대신 파일로 저장 (첫 10프레임만)
        if (g_frameCount < 10) {
            std::string filename = "libcamera_frame_" + std::to_string(g_frameCount) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "Saved frame to " << filename << std::endl;
        }
        
        // GUI는 주석 처리 (VNC 환경 문제 방지)
        // cv::imshow("LibCamera Zero-Copy (720p)", frame);
        // int key = cv::waitKey(1);
        // if (key == 'q') g_camera->stop();
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
    request->reuse();
    g_camera->queueRequest(request);
    g_frameCount++;

    if (g_frameCount >= g_targetFrames) g_camera->stop();
}

int main() {
    std::cout << "Starting libcamera zero-copy demo..." << std::endl;
    
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

    // 720p 해상도 설정
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
    
    // 지원되는 포맷 시도 (BGR888 -> RGB888 -> YUV420 순서)
    std::vector<PixelFormat> formatOptions = {
        formats::BGR888,
        formats::RGB888,
        formats::YUV420
    };
    
    bool formatSet = false;
    for (const auto& format : formatOptions) {
        std::cout << "Trying format: " << format.toString() << std::endl;
        streamConfig.pixelFormat = format;
        streamConfig.size = {1280, 720};
        
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
    g_mappedBuffers.resize(buffers.size());
    std::cout << "Mapping " << buffers.size() << " buffers..." << std::endl;
    
    for (size_t i = 0; i < buffers.size(); ++i) {
        const FrameBuffer::Plane &plane = buffers[i]->planes().front();
        g_mappedBuffers[i].data = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
        g_mappedBuffers[i].size = plane.length;
        
        if (g_mappedBuffers[i].data == MAP_FAILED) {
            std::cerr << "mmap failed for buffer " << i << std::endl;
            delete allocator;
            g_camera->release();
            manager->stop();
            return 1;
        }
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
        
        if (request->addBuffer(stream, buffers[i].get())) {
            std::cerr << "Failed to add buffer to request" << std::endl;
            delete allocator;
            g_camera->release();
            manager->stop();
            return 1;
        }
        
        // 버퍼 인덱스를 전역 맵에 저장
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
    while (g_frameCount < g_targetFrames) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // 정리
    for (auto &buf : g_mappedBuffers) {
        if (buf.data != MAP_FAILED) {
            munmap(buf.data, buf.size);
        }
    }
    
    delete allocator;
    g_camera->stop();
    g_camera->release();
    manager->stop();
    // cv::destroyAllWindows(); // GUI 미사용으로 주석 처리

    std::cout << "Demo completed successfully!" << std::endl;
    return 0;
}
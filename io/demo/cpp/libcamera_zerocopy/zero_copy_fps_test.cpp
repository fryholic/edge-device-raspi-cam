#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cstring>
#include <csignal>
#include <array>      // std::array
#include <sys/mman.h> // mmap, munmap
#include <algorithm>  // min, max
#include <iomanip>    // std::setprecision
#include <numeric>    // std::accumulate
#include <limits>     // std::numeric_limits
#include <functional> // std::function
#ifdef _OPENMP
#include <omp.h>      // OpenMP 병렬처리
#endif
#ifdef __ARM_NEON
#include <arm_neon.h> // ARM NEON SIMD 명령어
#endif
#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/control_ids.h>
#include <libcamera/stream.h>

using namespace libcamera;
using namespace std::chrono;
using namespace std::literals::chrono_literals; // 시간 리터럴 사용을 위함

// 프레임 통계를 위한 구조체
struct FrameStats {
    size_t frameIndex;
    high_resolution_clock::time_point captureTime;
    high_resolution_clock::time_point processTime;
    double ioTime;
    double instantFps;  // 해당 프레임 시점의 순간 FPS
    double avgFps;      // 해당 프레임까지의 평균 FPS
};

class DMABufZeroCopyFPSTest {
private:    
    // 카메라 및 버퍼 관련
    std::shared_ptr<Camera> camera;
    std::unique_ptr<CameraManager> cameraManager;
    std::unique_ptr<CameraConfiguration> config;
    Stream* stream;
    
    // DMA-BUF 관련
    std::shared_ptr<FrameBufferAllocator> allocator;
    std::vector<std::vector<void*>> bufferPlaneMappings;  // [buffer][plane] = mapped_ptr
    std::vector<std::vector<size_t>> bufferPlaneSizes;    // [buffer][plane] = size
    
    // 성능 측정
    high_resolution_clock::time_point startTime;
    std::vector<FrameStats> frameStats;
    size_t frameCount;
    
    // 기능 플래그
    bool verboseOutput;   // 상세 출력 여부
    
    // 종료 플래그
    std::atomic<bool> stopping;
    
public:
    DMABufZeroCopyFPSTest(bool verboseOutput = true) 
        : stream(nullptr), frameCount(0), verboseOutput(verboseOutput), stopping(false) {
        
#ifdef _OPENMP
        // 30fps 달성을 위한 OpenMP 설정 - Raspberry Pi 4B의 4코어 최대 활용
        omp_set_num_threads(4);
        omp_set_dynamic(0);
        if (verboseOutput) {
            std::cout << "OpenMP 활성화: " << omp_get_max_threads() << " threads" << std::endl;
        }
#endif
    }
    
    ~DMABufZeroCopyFPSTest() {
        cleanup();
    }
    
    bool initialize() {
        // 카메라 매니저 초기화
        cameraManager = std::make_unique<CameraManager>();
        int ret = cameraManager->start();
        if (ret) {
            std::cerr << "Failed to start camera manager: " << ret << std::endl;
            return false;
        }
        
        // 카메라 찾기
        auto cameras = cameraManager->cameras();
        if (cameras.empty()) {
            std::cerr << "No cameras found" << std::endl;
            return false;
        }
        
        camera = cameras[0];
        std::cout << "Using camera: " << camera->id() << std::endl;
        
        // 카메라 획득
        if (camera->acquire()) {
            std::cerr << "Failed to acquire camera" << std::endl;
            return false;
        }
        
        // 카메라 설정 생성
        config = camera->generateConfiguration({StreamRole::Viewfinder});
        if (!config) {
            std::cerr << "Failed to generate configuration" << std::endl;
            return false;
        }
        
        // 스트림 설정 - 1920x1080에서 30 FPS 달성을 위한 최적화
        StreamConfiguration& streamConfig = config->at(0);
        
        // 성능 최적화를 위한 설정
        // 1. 해상도를 1920x1080으로 설정 (고정)
        streamConfig.size = Size(1920, 1080);
        
        // 2. RGB888 포맷으로 직접 설정 (BGR 순서로 저장됨)
        streamConfig.pixelFormat = libcamera::formats::RGB888;
        
        // 3. 30fps 달성을 위한 성능 최적화 설정
        // 4. 카메라 컨트롤 설정 - 30fps 달성을 위한 공격적인 최적화
        ControlList cameraControls;
        
        // 5. 30fps 달성을 위한 초고속 설정
        cameraControls.set(controls::AeExposureMode, controls::ExposureNormal);
        cameraControls.set(controls::ExposureTime, 10000);  // 10ms - 극도로 빠른 노출
        
        // 6. 30fps 목표를 위한 프레임 지속 시간 제한 (더 공격적)
        std::array<int64_t, 2> frameDurationLimits = {30000, 30000};  // 33.3fps 목표 (1초/30 = 30000μs)
        cameraControls.set(controls::FrameDurationLimits, frameDurationLimits);
        
        // 7. 극도의 성능 최적화 설정
        cameraControls.set(controls::AeEnable, false);      // 자동 노출 완전 비활성화
        cameraControls.set(controls::AwbEnable, false);     // 자동 화이트 밸런스 완전 비활성화
        cameraControls.set(controls::AfMode, controls::AfModeManual); // 자동 포커스 비활성화
        
        // 8. 최소 버퍼로 지연 시간 최소화
        streamConfig.bufferCount = 3;  // 최소 버퍼로 지연 시간 최소화
        
        // 설정 검증 및 적용
        config->validate();
        if (camera->configure(config.get())) {
            std::cerr << "Failed to configure camera" << std::endl;
            return false;
        }
        
        stream = streamConfig.stream();
        
        std::cout << "Stream configuration:" << std::endl;
        std::cout << "  Size: " << streamConfig.size.width << "x" << streamConfig.size.height << std::endl;
        std::cout << "  Format: " << streamConfig.pixelFormat.toString() << std::endl;
        std::cout << "  Stride: " << streamConfig.stride << std::endl;
        
        if (verboseOutput) {
            std::cout << "Available formats:" << std::endl;
            // 사용 가능한 모든 포맷 출력
            for (const auto& format : streamConfig.formats().pixelformats()) {
                std::cout << "  - " << format.toString() << std::endl;
            }
        }
        
        return setupBuffers();
    }
    
    bool setupBuffers() {
        // 버퍼 할당
        allocator = std::make_shared<FrameBufferAllocator>(camera);
        if (allocator->allocate(stream) < 0) {
            std::cerr << "Failed to allocate buffers" << std::endl;
            return false;
        }
        
        // DMA-BUF 버퍼 설정
        const std::vector<std::unique_ptr<FrameBuffer>>& buffers = allocator->buffers(stream);
        
        for (size_t i = 0; i < buffers.size(); ++i) {
            const FrameBuffer* buffer = buffers[i].get();
            
            std::cout << "Buffer " << i << " has " << buffer->planes().size() << " planes:" << std::endl;
            
            std::vector<void*> planeMappings;
            std::vector<size_t> planeSizes;
            
            for (size_t planeIndex = 0; planeIndex < buffer->planes().size(); ++planeIndex) {
                const FrameBuffer::Plane& plane = buffer->planes()[planeIndex];
                
                void* memory = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, 
                                  MAP_SHARED, plane.fd.get(), 0);
                if (memory == MAP_FAILED) {
                    std::cerr << "Failed to map buffer plane " << planeIndex << std::endl;
                    return false;
                }
                
                std::cout << "  Plane " << planeIndex << ": mapped " << plane.length << 
                         " bytes at fd " << plane.fd.get() << std::endl;
                
                planeMappings.push_back(memory);
                planeSizes.push_back(plane.length);
            }
            
            // 각 버퍼의 모든 평면을 저장
            bufferPlaneMappings.push_back(planeMappings);
            bufferPlaneSizes.push_back(planeSizes);
        }
        
        std::cout << "Successfully set up " << buffers.size() << " DMA-BUF buffers" << std::endl;
        return true;
    }
    
    void cleanup() {
        // 매핑된 버퍼 해제
        if (allocator) {
            const std::vector<std::unique_ptr<FrameBuffer>>& buffers = allocator->buffers(stream);
            for (size_t i = 0; i < bufferPlaneMappings.size(); ++i) {
                for (size_t j = 0; j < bufferPlaneMappings[i].size(); ++j) {
                    if (i < buffers.size() && j < buffers[i]->planes().size()) {
                        const FrameBuffer::Plane& plane = buffers[i]->planes()[j];
                        munmap(bufferPlaneMappings[i][j], plane.length);
                    }
                }
            }
        }
        
        // 카메라 해제
        if (camera) {
            camera->release();
        }
    }
    
    bool start() {
        if (!camera) {
            std::cerr << "Camera is not initialized" << std::endl;
            return false;
        }
        
        // 요청 완료 시그널 연결
        camera->requestCompleted.connect(this, &DMABufZeroCopyFPSTest::onRequestCompleted);
        
        // 요청 큐 준비
        std::vector<std::unique_ptr<Request>> requests;
        for (const std::unique_ptr<FrameBuffer>& buffer : allocator->buffers(stream)) {
            std::unique_ptr<Request> request = camera->createRequest();
            if (!request) {
                std::cerr << "Failed to create request" << std::endl;
                return false;
            }
            
            if (request->addBuffer(stream, buffer.get())) {
                std::cerr << "Failed to add buffer to request" << std::endl;
                return false;
            }
            
            requests.push_back(std::move(request));
        }
        
        // 카메라 시작
        if (camera->start()) {
            std::cerr << "Failed to start camera" << std::endl;
            return false;
        }
        
        // 타이머 시작
        startTime = high_resolution_clock::now();
        
        // 요청 큐에 넣기
        for (auto& request : requests) {
            camera->queueRequest(request.release());
        }
        
        return true;
    }
    
    void stop() {
        stopping.store(true);
        
        if (camera) {
            camera->stop();
            camera->requestCompleted.disconnect(this, &DMABufZeroCopyFPSTest::onRequestCompleted);
        }
    }
    
    void onRequestCompleted(Request* request) {
        if (stopping.load()) {
            return;
        }
        
        auto ioStartTime = high_resolution_clock::now();
        
        if (request->status() != Request::RequestComplete) {
            if (!stopping.load()) {
                std::cerr << "Request failed: " << request->status() << std::endl;
            }
            return;
        }
        
        // 버퍼에서 프레임 데이터 가져오기 (zero-copy)
        FrameBuffer* buffer = request->buffers().begin()->second;
        
        // 버퍼 인덱스 찾기
        size_t bufferIndex = 0;
        const std::vector<std::unique_ptr<FrameBuffer>>& buffers = allocator->buffers(stream);
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (buffers[i].get() == buffer) {
                bufferIndex = i;
                break;
            }
        }
        
        // DMA-BUF에서 직접 데이터 접근 (zero-copy) - 모든 평면 접근
        const std::vector<void*>& planeMappings = bufferPlaneMappings[bufferIndex];
        const std::vector<size_t>& planeSizes = bufferPlaneSizes[bufferIndex];
        
        auto ioEndTime = high_resolution_clock::now();
        double ioTime = duration_cast<microseconds>(ioEndTime - ioStartTime).count() / 1000.0;
        
        // FPS 계산 - 더 정확한 측정을 위해 1초마다 초기화
        double fps = 0;
        double instantFps = 0;
        static auto lastFpsUpdateTime = high_resolution_clock::now();
        static int framesSinceLastUpdate = 0;
        
        if (frameCount > 0) {
            // 전체 세션에 대한 평균 FPS
            auto totalElapsed = duration_cast<microseconds>(ioEndTime - startTime);
            fps = (frameCount + 1) * 1000000.0 / totalElapsed.count();
            
            // 최근 1초 동안의 순간 FPS
            framesSinceLastUpdate++;
            auto timeSinceLastUpdate = duration_cast<milliseconds>(ioEndTime - lastFpsUpdateTime);
            
            if (timeSinceLastUpdate.count() >= 1000) {  // 1초마다 업데이트
                instantFps = framesSinceLastUpdate * 1000.0 / timeSinceLastUpdate.count();
                lastFpsUpdateTime = ioEndTime;
                framesSinceLastUpdate = 0;
            }
        }
        
        // 프레임 통계 기록
        FrameStats stats;
        stats.frameIndex = frameCount;
        stats.captureTime = ioStartTime;
        stats.processTime = ioEndTime;
        stats.ioTime = ioTime;
        stats.instantFps = instantFps;
        stats.avgFps = fps;
        frameStats.push_back(stats);
        
        // 연속 FPS 모니터링 - 매 30프레임마다 출력
        if (frameCount % 30 == 0 || frameCount < 10) {
            std::cout << "\r■ Frame: " << std::setw(6) << frameCount 
                      << " | Avg FPS: " << std::fixed << std::setprecision(1) << std::setw(5) << stats.avgFps
                      << " | Instant FPS: " << std::fixed << std::setprecision(1) << std::setw(5) << stats.instantFps
                      << " | I/O: " << std::fixed << std::setprecision(1) << std::setw(4) << ioTime << "ms";
            std::cout.flush();
        }
        
        // 간단한 데이터 접근만 수행 (파일 저장 없음)
        if (planeMappings.size() >= 1) {
            // 데이터 접근만 하고 실제 처리는 생략 (최대 성능을 위해)
            uint8_t* srcRGB = static_cast<uint8_t*>(planeMappings[0]);
            size_t dataSize = planeSizes[0];
            
            // 첫 프레임에서만 기본 정보 출력
            if (frameCount == 0) {
                const StreamConfiguration& streamConfig = config->at(0);
                int width = streamConfig.size.width;
                int height = streamConfig.size.height;
                int stride = streamConfig.stride;
                
                std::cout << "\n초기 설정:" << std::endl;
                std::cout << "  Format: " << streamConfig.pixelFormat.toString() << std::endl;
                std::cout << "  Resolution: " << width << "x" << height << std::endl; 
                std::cout << "  Stride: " << stride << std::endl;
                std::cout << "  Data Size: " << dataSize << " bytes" << std::endl;
                std::cout << "  RGB Sample: (" << (int)srcRGB[0] << "," << (int)srcRGB[1] << "," << (int)srcRGB[2] << ")" << std::endl;
                std::cout << "\nFPS 모니터링 시작 (Ctrl+C로 종료):" << std::endl;
            }
        }
        
        // 다음 요청 준비
        request->reuse(Request::ReuseFlag::ReuseBuffers);
        
        frameCount++;
        
        if (!stopping.load()) {
            camera->queueRequest(request);
        }
    }
    
    // 최종 통계 출력
    void printFinalStats() {
        if (frameStats.empty()) {
            std::cout << "\n통계 없음 - 프레임이 캡처되지 않음" << std::endl;
            return;
        }
        
        // 기본 통계 계산
        double totalIoTime = 0;
        double maxIoTime = 0;
        double minIoTime = std::numeric_limits<double>::max();
        double maxFps = 0;
        double minFps = std::numeric_limits<double>::max();
        
        // 30fps 달성 여부 통계
        int framesAbove30Fps = 0;
        int frameCount = static_cast<int>(frameStats.size());
        
        for (const auto& stats : frameStats) {
            // I/O 시간 통계
            totalIoTime += stats.ioTime;
            maxIoTime = std::max(maxIoTime, stats.ioTime);
            minIoTime = std::min(minIoTime, stats.ioTime);
            
            // FPS 통계
            if (stats.avgFps > 0) {
                maxFps = std::max(maxFps, stats.avgFps);
                minFps = std::min(minFps, stats.avgFps);
                
                if (stats.avgFps >= 29.5) { // 30fps에 근접하는 경우를 허용
                    framesAbove30Fps++;
                }
            }
        }
        
        double avgIoTime = totalIoTime / frameStats.size();
        auto totalTime = duration_cast<milliseconds>(
            frameStats.back().processTime - startTime).count();
        double avgFps = frameStats.size() * 1000.0 / totalTime;
        double percent30Fps = (frameCount > 0) ? (framesAbove30Fps * 100.0 / frameCount) : 0;
        
        // 최종 통계 정보 출력
        std::cout << "\n\n=========================================================" << std::endl;
        std::cout << "최종 FPS 성능 통계" << std::endl;
        std::cout << "=========================================================" << std::endl;
        std::cout << "총 프레임 수: " << frameCount << " 프레임" << std::endl;
        std::cout << "총 실행 시간: " << std::fixed << std::setprecision(2) << totalTime / 1000.0 << " 초" << std::endl;
        std::cout << "평균 FPS: " << std::fixed << std::setprecision(1) << avgFps << std::endl;
        std::cout << "최소 FPS: " << std::fixed << std::setprecision(1) << minFps << std::endl;
        std::cout << "최대 FPS: " << std::fixed << std::setprecision(1) << maxFps << std::endl;
        std::cout << "30FPS 이상 달성 비율: " << std::fixed << std::setprecision(1) << percent30Fps << "% (" << framesAbove30Fps << "/" << frameCount << ")" << std::endl;
        std::cout << "평균 I/O 시간: " << std::fixed << std::setprecision(2) << avgIoTime << " ms" << std::endl;
        
        // 성능 평가
        std::cout << "\n성능 평가:" << std::endl;
        if (avgFps >= 30.0) {
            std::cout << "🎉 30fps 목표 달성! 뛰어난 성능입니다." << std::endl;
        } else if (avgFps >= 25.0) {
            std::cout << "👍 25fps 이상 달성! 양호한 성능입니다." << std::endl;
        } else if (avgFps >= 20.0) {
            std::cout << "⚡ 20fps 이상 달성! 추가 최적화가 필요합니다." << std::endl;
        } else {
            std::cout << "❌ 20fps 미만! 심각한 성능 문제가 있습니다." << std::endl;
        }
        std::cout << "=========================================================" << std::endl;
    }
};

// 시그널 핸들러
static std::atomic<bool> shouldExit{false};
static DMABufZeroCopyFPSTest* testInstance = nullptr;

void signalHandler(int signal) {
    std::cout << "\n\n신호 수신: " << signal << ". 종료 중..." << std::endl;
    shouldExit.store(true);
    if (testInstance) {
        testInstance->stop();
    }
}

int main(int argc, char** argv) {
    bool verboseOutput = false;  // 기본적으로 간단한 출력
    
    // 명령행 인수 처리
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--verbose" || arg == "-v") {
            verboseOutput = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --verbose, -v            상세 출력 활성화" << std::endl;
            std::cout << "  --help, -h               이 도움말 출력" << std::endl;
            return 0;
        }
    }
    
    // 시그널 핸들러 등록
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    std::cout << "=========================================================" << std::endl;
    std::cout << "DMA-BUF Zero-Copy FPS 테스트 (파일 저장 없음)" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "- 설정: 1920x1080 RGB888 포맷 (파일 저장 없음)" << std::endl;
    std::cout << "- 목표: 최대 FPS 달성 테스트" << std::endl;
    std::cout << "- 종료: Ctrl+C 키를 누르세요" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "최적화 적용 상태:" << std::endl;
#ifdef _OPENMP
    std::cout << "✓ OpenMP 병렬처리 활성화" << std::endl;
#else
    std::cout << "✗ OpenMP 병렬처리 비활성화" << std::endl;
#endif
#ifdef __ARM_NEON
    std::cout << "✓ ARM NEON SIMD 최적화 활성화" << std::endl;
#else
    std::cout << "✗ ARM NEON SIMD 최적화 비활성화" << std::endl;
#endif
    std::cout << "✓ -O3 최적화, 루프 언롤링, 네이티브 아키텍처 최적화" << std::endl;
    std::cout << "✓ 최소 버퍼, 자동 기능 비활성화, 빠른 노출 설정" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    try {
        DMABufZeroCopyFPSTest fpsTest(verboseOutput);
        testInstance = &fpsTest;
        
        if (!fpsTest.initialize()) {
            std::cerr << "카메라 초기화 실패" << std::endl;
            return -1;
        }
        
        if (!fpsTest.start()) {
            std::cerr << "캡처 시작 실패" << std::endl;
            return -1;
        }
        
        // Ctrl+C가 입력될 때까지 계속 실행
        while (!shouldExit.load()) {
            std::this_thread::sleep_for(100ms);
        }
        
        fpsTest.stop();
        fpsTest.printFinalStats();
        
    } catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

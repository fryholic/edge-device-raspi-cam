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

// OpenCV 헤더 추가
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/ocl.hpp>

using namespace libcamera;
using namespace std::chrono;
using namespace std::literals::chrono_literals; // 시간 리터럴 사용을 위함

// 프레임 통계를 위한 구조체
struct FrameStats {
    size_t frameIndex;
    high_resolution_clock::time_point captureTime;
    high_resolution_clock::time_point processTime;
    double ioTime;
    double processOpenCVTime;  // OpenCV 처리 시간 추가
    double instantFps;  // 해당 프레임 시점의 순간 FPS
    double avgFps;      // 해당 프레임까지의 평균 FPS
};

class DMABufOpenCVDemo {
private:    
    // 카메라 및 버퍼 관련
    std::shared_ptr<Camera> camera;
    std::unique_ptr<CameraManager> cameraManager;
    std::unique_ptr<CameraConfiguration> config;
    Stream* stream;
    
    // 카메라 컨트롤
    ControlList cameraControls;
    
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
    bool enableOpenCV;    // OpenCV 처리 활성화 여부
    bool saveFrames;      // 프레임 저장 여부
    
    // 종료 플래그
    std::atomic<bool> stopping;
    
public:
    DMABufOpenCVDemo(bool verboseOutput = true, bool enableOpenCV = true, bool saveFrames = false) 
        : stream(nullptr), frameCount(0), verboseOutput(verboseOutput), 
          enableOpenCV(enableOpenCV), saveFrames(saveFrames), stopping(false) {
        
#ifdef _OPENMP
        // OpenMP 설정 - Raspberry Pi 4B의 4코어 최대 활용
        omp_set_num_threads(4);
        omp_set_dynamic(0);
        if (verboseOutput) {
            std::cout << "OpenMP 활성화: " << omp_get_max_threads() << " threads" << std::endl;
        }
#endif

#ifdef __ARM_NEON
        if (verboseOutput) {
            std::cout << "ARM NEON SIMD 최적화 활성화" << std::endl;
        }
#endif

        // OpenCV 하드웨어 최적화 설정
        if (enableOpenCV) {
            cv::setUseOptimized(true);  // OpenCV 내장 최적화 활성화
            cv::setNumThreads(4);       // 멀티스레드 활성화
            
            if (verboseOutput) {
                std::cout << "OpenCV 최적화 설정:" << std::endl;
                std::cout << "  - 하드웨어 최적화: " << (cv::useOptimized() ? "활성화" : "비활성화") << std::endl;
                std::cout << "  - 스레드 수: " << cv::getNumThreads() << std::endl;
                
                // OpenCL 지원 확인
                if (cv::ocl::haveOpenCL()) {
                    std::cout << "  - OpenCL 지원: 활성화" << std::endl;
                    cv::ocl::setUseOpenCL(true);
                } else {
                    std::cout << "  - OpenCL 지원: 비활성화" << std::endl;
                }
            }
        }
    }
    
    ~DMABufOpenCVDemo() {
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
        
        // 스트림 설정 - 기존 FPS 테스트와 동일한 설정
        StreamConfiguration& streamConfig = config->at(0);
        
        // 성능 최적화를 위한 설정
        // 1. 해상도를 1920x1080으로 설정 (고정)
        streamConfig.size = Size(1920, 1080);
        
        // 2. RGB888 포맷으로 직접 설정 (BGR 순서로 저장됨)
        streamConfig.pixelFormat = libcamera::formats::RGB888;
        
        // 3. 30fps 달성을 위한 성능 최적화 설정
        // 4. 카메라 컨트롤 설정 - 30fps 달성을 위한 공격적인 최적화
        
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
        camera->requestCompleted.connect(this, &DMABufOpenCVDemo::onRequestCompleted);
        
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
        if (camera->start(&cameraControls)) {
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
            camera->requestCompleted.disconnect(this, &DMABufOpenCVDemo::onRequestCompleted);
        }
    }
    
    // NEON 최적화된 OpenCV 이미지 처리 함수
    cv::Mat processWithOpenCV(const cv::Mat& inputImage) {
        auto processStart = high_resolution_clock::now();
        
        cv::Mat processedImage;
        
        // NEON과 하드웨어 가속 최적화된 OpenCV 처리
        try {
#ifdef __ARM_NEON
            // NEON이 활성화된 경우 더 적극적인 최적화
            cv::setUseOptimized(true);  // OpenCV 내장 최적화 활성화
            cv::setNumThreads(4);       // 4코어 활용
#endif

            // 1. 하드웨어 가속을 위한 작은 크기 처리 (GPU/ISP 친화적)
            cv::Mat smallImage;
            cv::resize(inputImage, smallImage, cv::Size(960, 540), 0, 0, cv::INTER_LINEAR);
            
            // 2. NEON 최적화된 간단한 블러 (작은 커널)
            cv::Mat blurred;
            cv::GaussianBlur(smallImage, blurred, cv::Size(5, 5), 0);
            
            // 3. 엣지 검출 (더 작은 크기에서)
            cv::Mat smallGray, smallEdges;
            cv::cvtColor(blurred, smallGray, cv::COLOR_RGB2GRAY);
            cv::Canny(smallGray, smallEdges, 30, 90);  // 임계값 낮춤
            
            // 4. 엣지를 원본 크기로 복원 (하드웨어 가속)
            cv::Mat edges;
            cv::resize(smallEdges, edges, inputImage.size(), 0, 0, cv::INTER_NEAREST);
            
            // 5. 컬러로 변환 (NEON 최적화)
            cv::Mat edgesColor;
            cv::cvtColor(edges, edgesColor, cv::COLOR_GRAY2RGB);
            
            // 6. 가중 합성 (NEON 최적화)
            cv::addWeighted(inputImage, 0.85, edgesColor, 0.15, 0, processedImage);
            
            // 7. 간단한 텍스트 (성능 최적화)
            if (frameCount % 10 == 0) {  // 10프레임마다만 텍스트 업데이트
                std::string text = "Frame: " + std::to_string(frameCount);
                cv::putText(processedImage, text, cv::Point(30, 50), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            } else {
                // 기존 텍스트 유지를 위해 원본과 합성만
                processedImage = inputImage.clone();
                cv::addWeighted(processedImage, 0.85, edgesColor, 0.15, 0, processedImage);
            }
            
        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV 처리 오류: " << e.what() << std::endl;
            // 오류 시 원본 이미지 반환
            processedImage = inputImage.clone();
        }
        
        auto processEnd = high_resolution_clock::now();
        double processTime = duration_cast<microseconds>(processEnd - processStart).count() / 1000.0;
        
        if (verboseOutput && frameCount % 30 == 0) {
            std::cout << " | OpenCV: " << std::fixed << std::setprecision(1) << processTime << "ms";
        }
        
        return processedImage;
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
        
        // DMA-BUF에서 직접 데이터 접근 (zero-copy)
        const std::vector<void*>& planeMappings = bufferPlaneMappings[bufferIndex];
        // const std::vector<size_t>& planeSizes = bufferPlaneSizes[bufferIndex]; // 현재 미사용
        
        double processOpenCVTime = 0;
        
        // OpenCV 처리 (활성화된 경우)
        if (enableOpenCV && planeMappings.size() >= 1) {
            auto cvProcessStart = high_resolution_clock::now();
            
            // DMA 버퍼 데이터를 OpenCV Mat으로 변환 (zero-copy)
            uint8_t* srcData = static_cast<uint8_t*>(planeMappings[0]);
            const StreamConfiguration& streamConfig = config->at(0);
            int width = streamConfig.size.width;
            int height = streamConfig.size.height;
            int stride = streamConfig.stride;
            
            // OpenCV Mat 생성 (zero-copy로 버퍼 공유)
            cv::Mat rawImage(height, width, CV_8UC3, srcData, stride);
            
            // 색상 포맷 확인 및 변환 (첫 프레임에서 디버그 정보 출력)
            cv::Mat workingImage;
            if (frameCount == 0) {
                std::cout << "  데이터 샘플 (첫 3픽셀): (" << (int)srcData[0] << "," 
                         << (int)srcData[1] << "," << (int)srcData[2] << ") ("
                         << (int)srcData[3] << "," << (int)srcData[4] << "," << (int)srcData[5] << ") ("
                         << (int)srcData[6] << "," << (int)srcData[7] << "," << (int)srcData[8] << ")" << std::endl;
            }
            
            // libcamera RGB888은 실제로 BGR 순서일 가능성이 높음
            cv::cvtColor(rawImage, workingImage, cv::COLOR_BGR2RGB);
            
            // NEON 최적화된 OpenCV 처리 수행
            cv::Mat processedImage = processWithOpenCV(workingImage);
            
            // 프레임 저장 (요청된 경우)
            if (saveFrames && frameCount % 30 == 0) {  // 30프레임마다 저장
                std::string filename = "opencv_frame_" + std::to_string(frameCount) + ".png";
                
                // 저장할 때는 BGR 순서로 변환
                cv::Mat bgrForSave;
                cv::cvtColor(processedImage, bgrForSave, cv::COLOR_RGB2BGR);
                
                if (cv::imwrite(filename, bgrForSave)) {
                    if (verboseOutput) {
                        std::cout << " | Saved: " << filename;
                    }
                } else {
                    std::cerr << " | 저장 실패: " << filename;
                }
            }
            
            auto cvProcessEnd = high_resolution_clock::now();
            processOpenCVTime = duration_cast<microseconds>(cvProcessEnd - cvProcessStart).count() / 1000.0;
        }
        
        auto ioEndTime = high_resolution_clock::now();
        double ioTime = duration_cast<microseconds>(ioEndTime - ioStartTime).count() / 1000.0;
        
        // FPS 계산
        double fps = 0;
        double instantFps = 0;
        static auto lastFpsUpdateTime = high_resolution_clock::now();
        static int framesSinceLastUpdate = 0;
        
        if (frameCount > 0) {
            auto totalElapsed = duration_cast<microseconds>(ioEndTime - startTime);
            fps = (frameCount + 1) * 1000000.0 / totalElapsed.count();
            
            framesSinceLastUpdate++;
            auto timeSinceLastUpdate = duration_cast<milliseconds>(ioEndTime - lastFpsUpdateTime);
            
            if (timeSinceLastUpdate.count() >= 1000) {
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
        stats.processOpenCVTime = processOpenCVTime;
        stats.instantFps = instantFps;
        stats.avgFps = fps;
        frameStats.push_back(stats);
        
        // 연속 모니터링 출력
        if (frameCount % 30 == 0 || frameCount < 10) {
            std::cout << "\r■ Frame: " << std::setw(6) << frameCount 
                      << " | Avg FPS: " << std::fixed << std::setprecision(1) << std::setw(5) << stats.avgFps
                      << " | Instant FPS: " << std::fixed << std::setprecision(1) << std::setw(5) << stats.instantFps
                      << " | I/O: " << std::fixed << std::setprecision(1) << std::setw(4) << ioTime << "ms";
            
            if (enableOpenCV) {
                std::cout << " | OpenCV: " << std::fixed << std::setprecision(1) << std::setw(4) << processOpenCVTime << "ms";
            }
            
            std::cout.flush();
        }
        
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
            std::cout << "  OpenCV 처리: " << (enableOpenCV ? "활성화" : "비활성화") << std::endl;
            std::cout << "  프레임 저장: " << (saveFrames ? "활성화" : "비활성화") << std::endl;
            std::cout << "\n모니터링 시작 (Ctrl+C로 종료):" << std::endl;
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
        double totalOpenCVTime = 0;
        double maxIoTime = 0;
        double minIoTime = std::numeric_limits<double>::max();
        double maxOpenCVTime = 0;
        double minOpenCVTime = std::numeric_limits<double>::max();
        double maxFps = 0;
        double minFps = std::numeric_limits<double>::max();
        
        int framesAbove30Fps = 0;
        int frameCount = static_cast<int>(frameStats.size());
        
        for (const auto& stats : frameStats) {
            totalIoTime += stats.ioTime;
            totalOpenCVTime += stats.processOpenCVTime;
            maxIoTime = std::max(maxIoTime, stats.ioTime);
            minIoTime = std::min(minIoTime, stats.ioTime);
            
            if (enableOpenCV && stats.processOpenCVTime > 0) {
                maxOpenCVTime = std::max(maxOpenCVTime, stats.processOpenCVTime);
                minOpenCVTime = std::min(minOpenCVTime, stats.processOpenCVTime);
            }
            
            if (stats.avgFps > 0) {
                maxFps = std::max(maxFps, stats.avgFps);
                minFps = std::min(minFps, stats.avgFps);
                
                if (stats.avgFps >= 29.5) {
                    framesAbove30Fps++;
                }
            }
        }
        
        double avgIoTime = totalIoTime / frameStats.size();
        double avgOpenCVTime = enableOpenCV ? (totalOpenCVTime / frameStats.size()) : 0;
        auto totalTime = duration_cast<milliseconds>(
            frameStats.back().processTime - startTime).count();
        double avgFps = frameStats.size() * 1000.0 / totalTime;
        double percent30Fps = (frameCount > 0) ? (framesAbove30Fps * 100.0 / frameCount) : 0;
        
        // 최종 통계 정보 출력
        std::cout << "\n\n=========================================================" << std::endl;
        std::cout << "OpenCV 데모 성능 통계" << std::endl;
        std::cout << "=========================================================" << std::endl;
        std::cout << "총 프레임 수: " << frameCount << " 프레임" << std::endl;
        std::cout << "총 실행 시간: " << std::fixed << std::setprecision(2) << totalTime / 1000.0 << " 초" << std::endl;
        std::cout << "평균 FPS: " << std::fixed << std::setprecision(1) << avgFps << std::endl;
        std::cout << "최소 FPS: " << std::fixed << std::setprecision(1) << minFps << std::endl;
        std::cout << "최대 FPS: " << std::fixed << std::setprecision(1) << maxFps << std::endl;
        std::cout << "30FPS 이상 달성 비율: " << std::fixed << std::setprecision(1) << percent30Fps << "% (" << framesAbove30Fps << "/" << frameCount << ")" << std::endl;
        std::cout << "평균 I/O 시간: " << std::fixed << std::setprecision(2) << avgIoTime << " ms" << std::endl;
        
        if (enableOpenCV) {
            std::cout << "평균 OpenCV 처리 시간: " << std::fixed << std::setprecision(2) << avgOpenCVTime << " ms" << std::endl;
            std::cout << "최소 OpenCV 처리 시간: " << std::fixed << std::setprecision(2) << minOpenCVTime << " ms" << std::endl;
            std::cout << "최대 OpenCV 처리 시간: " << std::fixed << std::setprecision(2) << maxOpenCVTime << " ms" << std::endl;
        }
        
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
        
        if (enableOpenCV) {
            std::cout << "\nOpenCV 처리 분석:" << std::endl;
            if (avgOpenCVTime < 10.0) {
                std::cout << "✅ OpenCV 처리가 매우 빠릅니다 (<10ms)" << std::endl;
            } else if (avgOpenCVTime < 20.0) {
                std::cout << "⚠️ OpenCV 처리가 다소 느립니다 (<20ms)" << std::endl;
            } else {
                std::cout << "❌ OpenCV 처리가 매우 느립니다 (>20ms)" << std::endl;
            }
        }
        
        std::cout << "=========================================================" << std::endl;
    }
};

// 시그널 핸들러
static std::atomic<bool> shouldExit{false};
static DMABufOpenCVDemo* demoInstance = nullptr;

void signalHandler(int signal) {
    std::cout << "\n\n신호 수신: " << signal << ". 종료 중..." << std::endl;
    shouldExit.store(true);
    if (demoInstance) {
        demoInstance->stop();
    }
}

int main(int argc, char** argv) {
    bool verboseOutput = false;
    bool enableOpenCV = true;
    bool saveFrames = false;
    
    // 명령행 인수 처리
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--verbose" || arg == "-v") {
            verboseOutput = true;
        } else if (arg == "--no-opencv") {
            enableOpenCV = false;
        } else if (arg == "--save-frames" || arg == "-s") {
            saveFrames = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --verbose, -v            상세 출력 활성화" << std::endl;
            std::cout << "  --no-opencv              OpenCV 처리 비활성화" << std::endl;
            std::cout << "  --save-frames, -s        프레임을 PNG 파일로 저장" << std::endl;
            std::cout << "  --help, -h               이 도움말 출력" << std::endl;
            return 0;
        }
    }
    
    // 시그널 핸들러 등록
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    std::cout << "=========================================================" << std::endl;
    std::cout << "DMA-BUF Zero-Copy OpenCV 데모" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "- 설정: 1920x1080 RGB888 포맷" << std::endl;
    std::cout << "- OpenCV 처리: " << (enableOpenCV ? "활성화" : "비활성화") << std::endl;
    std::cout << "- 프레임 저장: " << (saveFrames ? "활성화" : "비활성화") << std::endl;
    std::cout << "- 종료: Ctrl+C 키를 누르세요" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    try {
        DMABufOpenCVDemo demo(verboseOutput, enableOpenCV, saveFrames);
        demoInstance = &demo;
        
        if (!demo.initialize()) {
            std::cerr << "카메라 초기화 실패" << std::endl;
            return -1;
        }
        
        if (!demo.start()) {
            std::cerr << "캡처 시작 실패" << std::endl;
            return -1;
        }
        
        // Ctrl+C가 입력될 때까지 계속 실행
        while (!shouldExit.load()) {
            std::this_thread::sleep_for(100ms);
        }
        
        demo.stop();
        demo.printFinalStats();
        
    } catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}


/*

1. 컴파일:
make -f Makefile.opencv_demo

2. 실행 옵션들:
# 기본 실행 (OpenCV 처리 포함)
./zero_copy_opencv_demo

# 상세 출력
./zero_copy_opencv_demo --verbose

# OpenCV 처리 없이 (순수 FPS 테스트)
./zero_copy_opencv_demo --no-opencv

# 프레임 저장
./zero_copy_opencv_demo --save-frames


*/
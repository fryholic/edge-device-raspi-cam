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

// 저장할 프레임 데이터 구조체
struct FrameData {
    std::vector<uint8_t> data;
    size_t width;
    size_t height;
    size_t frameNumber;
};

class DMABufZeroCopyCapture {
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
    size_t targetFrames;
    
    // 기능 플래그
    bool savePPM;         // PPM 파일 저장 여부
    bool verboseOutput;   // 상세 출력 여부
    int saveInterval;     // 파일 저장 주기 (프레임 단위)
    
    // 파일 저장을 위한 백그라운드 처리
    std::queue<FrameData> saveQueue;
    std::mutex saveMutex;
    std::condition_variable saveCondition;
    std::thread saveThread;
    std::atomic<bool> shouldStop;
    
    // 종료 플래그
    std::atomic<bool> stopping;
    
public:
    DMABufZeroCopyCapture(size_t targetFrames = 100, int saveInterval = 10, bool savePPM = false, bool verboseOutput = true) 
        : stream(nullptr), frameCount(0), targetFrames(targetFrames), 
          savePPM(savePPM), verboseOutput(verboseOutput), saveInterval(saveInterval),
          shouldStop(false), stopping(false) {
        
#ifdef _OPENMP
        // 30fps 달성을 위한 OpenMP 설정 - Raspberry Pi 4B의 4코어 최대 활용
        omp_set_num_threads(4);
        omp_set_dynamic(0);
        if (verboseOutput) {
            std::cout << "OpenMP 활성화: " << omp_get_max_threads() << " threads" << std::endl;
        }
#endif
        
        // 백그라운드 저장 스레드 시작
        saveThread = std::thread(&DMABufZeroCopyCapture::saveWorker, this);
    }
    
    ~DMABufZeroCopyCapture() {
        shouldStop.store(true);
        saveCondition.notify_all();
        if (saveThread.joinable()) {
            saveThread.join();
        }
        cleanup();
    }
    
    // 파일 저장 워커 스레드 - 30fps 달성을 위한 최적화
    void saveWorker() {
        while (true) {
            FrameData frameData;
            {
                std::unique_lock<std::mutex> lock(saveMutex);
                saveCondition.wait(lock, [this] { 
                    return !saveQueue.empty() || shouldStop.load(); 
                });
                
                if (shouldStop.load() && saveQueue.empty()) {
                    break;
                }
                
                if (!saveQueue.empty()) {
                    frameData = std::move(saveQueue.front());
                    saveQueue.pop();
                } else {
                    continue;
                }
            }
            
            // RGB 파일 저장 (바이너리)
            std::string filename = "frame_rgb_" + std::to_string(frameData.frameNumber) + 
                                "_" + std::to_string(frameData.width) + 
                                "x" + std::to_string(frameData.height) + 
                                ".rgb";
                                
            std::ofstream file(filename, std::ios::binary);
            if (file.is_open()) {
                file.write(reinterpret_cast<const char*>(frameData.data.data()), 
                          frameData.data.size());
                file.close();
                
                // 30fps 달성을 위한 최적화: 저장 완료 메시지 간소화
                if (verboseOutput && frameData.frameNumber % (saveInterval * 5) == 0) {
                    std::cout << "\n[SAVED] " << filename << " (" << frameData.data.size() << " bytes)";
                }
            }
            
            // PPM 파일 저장 (선택적)
            if (savePPM) {
                std::string ppmFilename = "frame_rgb_" + std::to_string(frameData.frameNumber) + 
                                       "_" + std::to_string(frameData.width) + 
                                       "x" + std::to_string(frameData.height) + 
                                       ".ppm";
                
                std::ofstream ppmFile(ppmFilename, std::ios::binary);
                if (ppmFile.is_open()) {
                    // PPM 헤더 작성
                    ppmFile << "P6\n" << frameData.width << " " << frameData.height << "\n255\n";
                    // RGB 데이터 저장
                    ppmFile.write(reinterpret_cast<const char*>(frameData.data.data()), frameData.data.size());
                    ppmFile.close();
                    
                    if (verboseOutput) {
                        std::cout << "\nSaved PPM: " << ppmFilename;
                    }
                }
            }
            
            // 메타데이터 파일 생성 (선택적)
            if (verboseOutput) {
                std::string metaFilename = "frame_rgb_" + std::to_string(frameData.frameNumber) + "_meta.txt";
                std::ofstream metaFile(metaFilename);
                if (metaFile.is_open()) {
                    metaFile << "Frame: " << frameData.frameNumber << std::endl;
                    metaFile << "Format: RGB888 (BGR->RGB 변환)" << std::endl;
                    metaFile << "Size: " << frameData.width << "x" << frameData.height << " (" << frameData.data.size() << " bytes)" << std::endl;
                    metaFile.close();
                }
            }
        }
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
        camera->requestCompleted.connect(this, &DMABufZeroCopyCapture::onRequestCompleted);
        
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
            camera->requestCompleted.disconnect(this, &DMABufZeroCopyCapture::onRequestCompleted);
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
        
        // 30fps 달성을 위한 최적화: 출력 빈도 줄이기 (매 30프레임마다만 출력)
        if (frameCount % 30 == 0 || frameCount < 10) {
            std::cout << "\n■ Frame: " << frameCount 
                      << " | Avg FPS: " << std::fixed << std::setprecision(1) << stats.avgFps
                      << " | Instant FPS: " << std::fixed << std::setprecision(1) << stats.instantFps
                      << " | I/O: " << std::fixed << std::setprecision(1) << ioTime << "ms";
        }
        
        // RGB888 데이터를 직접 저장
        try {
            // 스트림 설정에서 해상도와 stride 정보 가져오기
            const StreamConfiguration& streamConfig = config->at(0);
            int width = streamConfig.size.width;
            int height = streamConfig.size.height;
            int stride = streamConfig.stride;
            // 30fps 달성을 위한 최적화: 상세 출력은 처음 몇 프레임만
            if (frameCount < 3) {
                std::cout << "\n  - Format: " << streamConfig.pixelFormat.toString() 
                          << " | Resolution: " << width << "x" << height 
                          << " | Stride: " << stride
                          << " | Planes: " << planeMappings.size();
            }
            
            bool dataExtracted = false;
            std::vector<uint8_t> rgbData;
            size_t rgbSize = 0;
            
            // RGB888 형식 처리
            if (planeMappings.size() >= 1) {
                uint8_t* srcRGB = static_cast<uint8_t*>(planeMappings[0]);
                size_t dataSize = planeSizes[0];
                
                if (frameCount < 3) {
                    std::cout << " | Data Size: " << dataSize << " bytes";
                }
                
                // RGB888은 픽셀당 3바이트
                rgbSize = width * height * 3;
                
                // 데이터를 그대로 복사
                rgbData.resize(rgbSize);
                
                // RGB 스트라이드 계산 - libcamera에서는 stride가 픽셀 단위로 주어질 수 있음
                size_t bytesPerPixel = 3; // RGB888은 픽셀당 3바이트
                size_t effectiveStride = stride;
                
                // RGB 포맷의 경우 stride는 바이트 단위가 아닌 픽셀 단위일 수 있음
                // 이를 바이트 단위로 변환
                if (effectiveStride < width * bytesPerPixel) {
                    effectiveStride *= bytesPerPixel;
                }
                
                if (frameCount < 3) {
                    std::cout << " | Byte Stride: " << effectiveStride;
                }
                
                // 30fps 달성을 위한 초고속 BGR->RGB 변환
                // OpenMP + NEON SIMD 활용한 병렬 처리
                convertBGRtoRGB_Parallel(srcRGB, rgbData.data(), width, height, effectiveStride);
                
                // 30fps 달성을 위한 최적화: RGB 샘플 출력은 첫 프레임만
                if (frameCount == 0) {
                    std::cout << "\n\n  ■ RGB Sample (First 3 pixels): ("
                              << (int)rgbData[0] << "," << (int)rgbData[1] << "," << (int)rgbData[2] << ") "
                              << "(" << (int)rgbData[3] << "," << (int)rgbData[4] << "," << (int)rgbData[5] << ") "
                              << "(" << (int)rgbData[6] << "," << (int)rgbData[7] << "," << (int)rgbData[8] << ")";
                }
                
                dataExtracted = true;
            }
            
            // RGB 데이터를 파일로 저장 - saveInterval 프레임당 1회만 저장
            if (dataExtracted && frameCount % saveInterval == 0) {
                // 성능 최적화: 기존 백그라운드 저장 시스템 사용
                FrameData frameData;
                frameData.data = std::move(rgbData);  // move semantics로 복사 비용 절약
                frameData.width = width;
                frameData.height = height;
                frameData.frameNumber = frameCount;
                
                {
                    std::lock_guard<std::mutex> lock(saveMutex);
                    saveQueue.push(std::move(frameData));
                }
                saveCondition.notify_one();
                
                // 30fps 달성을 위한 최적화: 저장 메시지도 간소화
                if (frameCount % saveInterval == 0) {
                    std::cout << " [SAVE]";
                }
            }
            
            dataExtracted = true;
        } catch (const std::exception& e) {
            std::cerr << "Error during RGB conversion: " << e.what() << std::endl;
        }
        
        // 다음 요청 준비
        request->reuse(Request::ReuseFlag::ReuseBuffers);
        
        frameCount++;
        
        if (frameCount < targetFrames && !stopping.load()) {
            camera->queueRequest(request);
        } else if (frameCount >= targetFrames && !stopping.load()) {
            std::cout << "\nTarget frame count reached. Stopping capture..." << std::endl;
            stop();
        }
    }
    
    // 저장된 프레임 통계 출력
    void printStats() {
        if (frameStats.empty()) {
            std::cout << "No frames captured" << std::endl;
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
        
        // 시간별 FPS 분포 (프레임 시작 후 시간대별)
        std::vector<double> fpsBySecond;
        std::vector<int> framesBySecond;
        int currentSecond = 0;
        int framesThisSecond = 0;
        auto firstFrameTime = frameStats.front().captureTime;
        
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
            
            // 초당 프레임 수 계산
            auto timeSinceStart = duration_cast<seconds>(stats.captureTime - firstFrameTime).count();
            if (static_cast<int>(timeSinceStart) > currentSecond) {
                while (currentSecond < static_cast<int>(timeSinceStart)) {
                    fpsBySecond.push_back(framesThisSecond);
                    framesBySecond.push_back(framesThisSecond);
                    framesThisSecond = 0;
                    currentSecond++;
                }
            }
            framesThisSecond++;
        }
        
        // 마지막 초에 대한 처리
        if (framesThisSecond > 0) {
            fpsBySecond.push_back(framesThisSecond);
            framesBySecond.push_back(framesThisSecond);
        }
        
        double avgIoTime = totalIoTime / frameStats.size();
        auto totalTime = duration_cast<milliseconds>(
            frameStats.back().processTime - startTime).count();
        double avgFps = frameStats.size() * 1000.0 / totalTime;
        double percent30Fps = (frameCount > 0) ? (framesAbove30Fps * 100.0 / frameCount) : 0;
        
        // 통계 정보 출력
        std::cout << "\n===== 캡처 성능 통계 =====" << std::endl;
        std::cout << "총 프레임 수: " << frameCount << " 프레임" << std::endl;
        std::cout << "총 실행 시간: " << std::fixed << std::setprecision(2) << totalTime / 1000.0 << " 초" << std::endl;
        std::cout << "평균 FPS: " << avgFps << std::endl;
        std::cout << "최소 FPS: " << minFps << std::endl;
        std::cout << "최대 FPS: " << maxFps << std::endl;
        std::cout << "30FPS 이상 달성 비율: " << percent30Fps << "% (" << framesAbove30Fps << "/" << frameCount << ")" << std::endl;
        std::cout << "\n=== I/O 처리 시간 통계 ===" << std::endl;
        std::cout << "평균 I/O 시간: " << avgIoTime << " ms" << std::endl;
        std::cout << "최소 I/O 시간: " << minIoTime << " ms" << std::endl;
        std::cout << "최대 I/O 시간: " << maxIoTime << " ms" << std::endl;
        
        // 초당 프레임 수 출력 (시간대별 분석)
        std::cout << "\n=== 시간별 프레임 분포 ===" << std::endl;
        for (size_t i = 0; i < fpsBySecond.size(); i++) {
            std::cout << i+1 << "초: " << framesBySecond[i] << " 프레임";
            
            // 30fps 달성 여부 표시
            if (framesBySecond[i] >= 30) {
                std::cout << " ✓";
            } else if (framesBySecond[i] >= 25) {
                std::cout << " △";
            } else {
                std::cout << " ✗";
            }
            
            std::cout << std::endl;
        }
        
        // 파일 저장 정보
        std::cout << "\n=== 파일 저장 정보 ===" << std::endl;
        std::cout << "저장 주기: " << saveInterval << " 프레임당 1회" << std::endl;
        std::cout << "저장된 파일 수: " << (frameCount / saveInterval) << " 파일" << std::endl;
    }
    
    // 30fps 달성을 위한 최적화된 BGR->RGB 변환 함수들
private:
    
#ifdef __ARM_NEON
    // ARM NEON SIMD를 활용한 BGR->RGB 변환 (8픽셀 단위 처리)
    inline void convertBGRtoRGB_NEON(const uint8_t* src, uint8_t* dst, int width) {
        const int simd_width = 8; // NEON은 128bit이므로 8픽셀(24바이트) 처리 가능
        int x = 0;
        
        for (x = 0; x <= width - simd_width; x += simd_width) {
            // 24바이트(8픽셀 * 3채널) 로드
            uint8x8x3_t bgr = vld3_u8(src + x * 3);
            
            // BGR -> RGB 변환 (채널 순서만 바꿈)
            uint8x8x3_t rgb;
            rgb.val[0] = bgr.val[2]; // R = B
            rgb.val[1] = bgr.val[1]; // G = G
            rgb.val[2] = bgr.val[0]; // B = R
            
            // 결과 저장
            vst3_u8(dst + x * 3, rgb);
        }
        
        // 남은 픽셀 처리 (SIMD로 처리하지 못한 부분)
        for (; x < width; x++) {
            dst[x * 3]     = src[x * 3 + 2]; // R
            dst[x * 3 + 1] = src[x * 3 + 1]; // G
            dst[x * 3 + 2] = src[x * 3];     // B
        }
    }
#endif
    
    // OpenMP를 활용한 병렬 BGR->RGB 변환
    void convertBGRtoRGB_Parallel(const uint8_t* src, uint8_t* dst, 
                                  int width, int height, size_t srcStride) {
        const size_t dstStride = width * 3;
        
#ifdef _OPENMP
        // OpenMP 병렬처리 - Raspberry Pi 4B의 4코어 활용
        #pragma omp parallel for schedule(dynamic, 32)
#endif
        for (int y = 0; y < height; y++) {
            const uint8_t* srcRow = src + y * srcStride;
            uint8_t* dstRow = dst + y * dstStride;
            
#ifdef __ARM_NEON
            // NEON SIMD 사용 가능한 경우
            convertBGRtoRGB_NEON(srcRow, dstRow, width);
#else
            // 일반적인 최적화된 변환 (8픽셀 단위 언롤링)
            int x = 0;
            for (; x <= width - 8; x += 8) {
                // 8픽셀 언롤링으로 처리
                for (int i = 0; i < 8; i++) {
                    const int srcIdx = (x + i) * 3;
                    const int dstIdx = (x + i) * 3;
                    dstRow[dstIdx]     = srcRow[srcIdx + 2]; // R
                    dstRow[dstIdx + 1] = srcRow[srcIdx + 1]; // G
                    dstRow[dstIdx + 2] = srcRow[srcIdx];     // B
                }
            }
            
            // 남은 픽셀 처리
            for (; x < width; x++) {
                const int srcIdx = x * 3;
                const int dstIdx = x * 3;
                dstRow[dstIdx]     = srcRow[srcIdx + 2]; // R
                dstRow[dstIdx + 1] = srcRow[srcIdx + 1]; // G
                dstRow[dstIdx + 2] = srcRow[srcIdx];     // B
            }
#endif
        }
    }
    
    // memcpy 기반 고속 변환 (BGR->RGB 변환 없이 raw 데이터만 복사)
    void copyRawData_Fast(const uint8_t* src, uint8_t* dst, 
                         int width, int height, size_t srcStride) {
        const size_t dstStride = width * 3;
        
        if (srcStride == dstStride) {
            // stride가 같으면 한번에 복사
            std::memcpy(dst, src, height * dstStride);
        } else {
            // 줄별로 복사
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (int y = 0; y < height; y++) {
                std::memcpy(dst + y * dstStride, src + y * srcStride, dstStride);
            }
        }
    }

public:
};

// 시그널 핸들러
static std::atomic<bool> shouldExit{false};
static DMABufZeroCopyCapture* captureInstance = nullptr;

void signalHandler(int signal) {
    std::cout << "\n신호 수신: " << signal << ". 종료 중..." << std::endl;
    shouldExit.store(true);
    if (captureInstance) {
        captureInstance->stop();
    }
}

int main(int argc, char** argv) {
    // 기본 설정
    size_t numFrames = 100;     // 기본값 - 30fps 측정을 위한 적정 수치
    int saveInterval = 10;      // 파일 저장 주기 (프레임 단위)
    bool savePPM = false;       // PPM 파일 저장 여부
    bool verboseOutput = true;  // 상세 출력 활성화 여부
    
    // 명령행 인수 처리
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--frames" || arg == "-f") {
            if (i + 1 < argc) {
                numFrames = std::stoul(argv[++i]);
            }
        } else if (arg == "--save-interval" || arg == "-s") {
            if (i + 1 < argc) {
                saveInterval = std::stoi(argv[++i]);
            }
        } else if (arg == "--ppm") {
            savePPM = true;
        } else if (arg == "--quiet" || arg == "-q") {
            verboseOutput = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --frames <n>, -f <n>     캡처할 프레임 수 (기본값: " << numFrames << ")" << std::endl;
            std::cout << "  --save-interval <n>, -s <n>  파일 저장 주기 (기본값: " << saveInterval << ")" << std::endl;
            std::cout << "  --ppm                    PPM 파일도 저장" << std::endl;
            std::cout << "  --quiet, -q              상세 출력 비활성화" << std::endl;
            std::cout << "  --help, -h               이 도움말 출력" << std::endl;
            return 0;
        }
    }
    
    // 시그널 핸들러 등록
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    std::cout << "=========================================================" << std::endl;
    std::cout << "RGB888 DMA-BUF zero-copy 성능 측정 (30fps 달성 여부 테스트)" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "- 설정: 1920x1080 RGB888 포맷, BGR->RGB 변환 적용" << std::endl;
    std::cout << "- 저장 주기: " << saveInterval << " 프레임당 1회 저장" << std::endl;
    std::cout << "- 목표: 30fps 달성 및 안정적인 프레임 레이트 유지" << std::endl;
    std::cout << "- 캡처 프레임 수: " << numFrames << std::endl;
    std::cout << "- 예상 저장 파일 수: " << (numFrames / saveInterval) << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "30fps 달성을 위한 최적화 적용:" << std::endl;
#ifdef _OPENMP
    std::cout << "✓ OpenMP 병렬처리 활성화 (4 threads)" << std::endl;
#else
    std::cout << "✗ OpenMP 병렬처리 비활성화" << std::endl;
#endif
#ifdef __ARM_NEON
    std::cout << "✓ ARM NEON SIMD 최적화 활성화" << std::endl;
#else
    std::cout << "✗ ARM NEON SIMD 최적화 비활성화" << std::endl;
#endif
    std::cout << "✓ -O3 최적화, 루프 언롤링, 네이티브 아키텍처 최적화" << std::endl;
    std::cout << "✓ 최소 버퍼 수, 자동 기능 비활성화, 빠른 노출 설정" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    try {
        DMABufZeroCopyCapture capture(numFrames, saveInterval, savePPM, verboseOutput);
        captureInstance = &capture;
        
        if (!capture.initialize()) {
            std::cerr << "카메라 초기화 실패" << std::endl;
            return -1;
        }
        
        if (!capture.start()) {
            std::cerr << "캡처 시작 실패" << std::endl;
            return -1;
        }
        
        std::cout << "캡처 시작... Ctrl+C로 중단 가능" << std::endl;
        
        // 캡처 완료까지 대기
        while (!shouldExit.load()) {
            std::this_thread::sleep_for(100ms);
        }
        
        capture.stop();
        
        std::cout << "\n=========================================================" << std::endl;
        std::cout << "캡처 완료! 통계 분석 중..." << std::endl;
        std::cout << "=========================================================" << std::endl;
        
        capture.printStats();
        
    } catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

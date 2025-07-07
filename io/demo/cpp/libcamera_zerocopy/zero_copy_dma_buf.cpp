#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>

using namespace libcamera;
using namespace std::chrono;

// 성능 측정 구조체
struct FrameStats {
    size_t frameIndex;
    high_resolution_clock::time_point captureTime;
    high_resolution_clock::time_point processTime;
    double ioTime;
};

// 파일 저장을 위한 구조체
struct FrameData {
    std::vector<uint8_t> data;
    size_t frameIndex;
    high_resolution_clock::time_point timestamp;
};

class DMABufZeroCopyCapture {
private:
    std::shared_ptr<Camera> camera;
    std::unique_ptr<CameraManager> cameraManager;
    std::unique_ptr<CameraConfiguration> config;
    Stream* stream;
    
    // DMA-BUF 관련
    std::shared_ptr<FrameBufferAllocator> allocator;
    std::vector<void*> mappedBuffers;
    std::vector<size_t> bufferSizes;
    
    // 성능 측정
    high_resolution_clock::time_point startTime;
    std::vector<FrameStats> frameStats;
    size_t frameCount;
    size_t targetFrames;
    
    // 파일 저장을 위한 백그라운드 처리
    std::queue<FrameData> saveQueue;
    std::mutex saveMutex;
    std::condition_variable saveCondition;
    std::thread saveThread;
    std::atomic<bool> shouldStop;
    
    // 종료 플래그
    std::atomic<bool> stopping;
    
public:
    DMABufZeroCopyCapture(size_t targetFrames = 100) 
        : frameCount(0), targetFrames(targetFrames), shouldStop(false), stopping(false) {
        
        // 백그라운드 저장 스레드 시작
        saveThread = std::thread(&DMABufZeroCopyCapture::saveWorker, this);
    }
    
    ~DMABufZeroCopyCapture() {
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
            std::cerr << "Failed to generate camera configuration" << std::endl;
            return false;
        }
        
        // 스트림 설정 - 1920x1080에서 30 FPS 달성을 위한 최적화
        StreamConfiguration& streamConfig = config->at(0);
        
        // 해상도를 1920x1080으로 설정
        streamConfig.size = Size(1920, 1080);
        streamConfig.pixelFormat = formats::YUV420;
        
        // 버퍼 수 증가로 처리량 향상
        streamConfig.bufferCount = 8;
        
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
            
            // 첫 번째 평면의 DMA-BUF fd를 사용
            const FrameBuffer::Plane& plane = buffer->planes()[0];
            int dmafd = plane.fd.get();
            size_t length = plane.length;
            
            // DMA-BUF를 메모리에 매핑 (zero-copy)
            void* mapped = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, dmafd, 0);
            if (mapped == MAP_FAILED) {
                std::cerr << "Failed to mmap DMA-BUF " << i << ": " << strerror(errno) << std::endl;
                return false;
            }
            
            mappedBuffers.push_back(mapped);
            bufferSizes.push_back(length);
            
            std::cout << "Buffer " << i << " mapped: " << mapped 
                      << ", size: " << length << " bytes" << std::endl;
        }
        
        std::cout << "Successfully set up " << buffers.size() << " DMA-BUF buffers" << std::endl;
        return true;
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
        void* data = mappedBuffers[bufferIndex];
        
        auto ioEndTime = high_resolution_clock::now();
        double ioTime = duration_cast<microseconds>(ioEndTime - ioStartTime).count() / 1000.0;
        
        // 프레임 통계 기록
        FrameStats stats;
        stats.frameIndex = frameCount;
        stats.captureTime = ioStartTime;
        stats.processTime = ioEndTime;
        stats.ioTime = ioTime;
        frameStats.push_back(stats);
        
        // FPS 계산
        double fps = 0;
        if (frameCount > 0) {
            auto elapsed = duration_cast<microseconds>(ioEndTime - startTime);
            fps = (frameCount + 1) * 1000000.0 / elapsed.count();
        }
        
        std::cout << "Frame " << frameCount 
                  << " | I/O Time: " << ioTime << "ms"
                  << " | FPS: " << fps
                  << " | Buffer: " << bufferIndex << std::endl;
        
        // YUV420 데이터를 표준 형식으로 파일 저장
        if (frameCount % 10 == 0) { // 10프레임마다 저장
            try {
                size_t actualBufferSize = bufferSizes[bufferIndex];
                const StreamConfiguration& streamConfig = config->at(0);
                size_t width = streamConfig.size.width;
                size_t height = streamConfig.size.height;
                
                std::cout << " [Buffer size: " << actualBufferSize << " bytes]";
                
                // libcamera에서 온 데이터를 표준 YUV420P로 변환
                size_t ySize = width * height;
                size_t uvSize = (width / 2) * (height / 2);
                size_t standardYuvSize = ySize + uvSize + uvSize;
                
                // 실제 버퍼에서 데이터 복사
                std::vector<uint8_t> buffer_copy(actualBufferSize);
                std::memcpy(buffer_copy.data(), data, actualBufferSize);
                
                // 표준 YUV420P 형식으로 재배열
                std::vector<uint8_t> standardYuv(standardYuvSize);
                
                if (actualBufferSize >= ySize) {
                    // Y 평면 복사
                    std::memcpy(standardYuv.data(), buffer_copy.data(), ySize);
                    
                    // U와 V 평면 처리 (libcamera 형식에 따라 조정 필요)
                    if (actualBufferSize >= ySize + uvSize * 2) {
                        // 표준 YUV420P 형식으로 가정하고 복사
                        std::memcpy(standardYuv.data() + ySize, 
                                   buffer_copy.data() + ySize, uvSize * 2);
                    } else {
                        // 부족한 UV 데이터는 0으로 채움 (그레이스케일)
                        std::memset(standardYuv.data() + ySize, 128, uvSize * 2);
                    }
                } else {
                    std::cerr << "Buffer too small for Y plane" << std::endl;
                    return;
                }
                
                // 표준 형식으로 파일 저장
                std::string filename = "frame_" + std::to_string(frameCount) + 
                                     "_" + std::to_string(width) + 
                                     "x" + std::to_string(height) + 
                                     "_yuv420p.yuv";
                
                std::ofstream file(filename, std::ios::binary);
                if (file.is_open()) {
                    file.write(reinterpret_cast<const char*>(standardYuv.data()), standardYuvSize);
                    file.close();
                    std::cout << " [Saved: " << filename << " (" << standardYuvSize << " bytes)]";
                    
                    // 메타데이터 파일 생성
                    std::string metaFilename = "frame_" + std::to_string(frameCount) + "_meta.txt";
                    std::ofstream metaFile(metaFilename);
                    if (metaFile.is_open()) {
                        metaFile << "Frame: " << frameCount << std::endl;
                        metaFile << "Format: YUV420P" << std::endl;
                        metaFile << "Width: " << width << std::endl;
                        metaFile << "Height: " << height << std::endl;
                        metaFile << "Original Buffer Size: " << actualBufferSize << " bytes" << std::endl;
                        metaFile << "Standard YUV Size: " << standardYuvSize << " bytes" << std::endl;
                        metaFile << "Y Size: " << ySize << " bytes" << std::endl;
                        metaFile << "U Size: " << uvSize << " bytes" << std::endl;
                        metaFile << "V Size: " << uvSize << " bytes" << std::endl;
                        metaFile << "Stride: " << streamConfig.stride << std::endl;
                        metaFile.close();
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during image processing: " << e.what() << std::endl;
            }
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
    
    bool start() {
        startTime = high_resolution_clock::now();
        
        // 요청 생성 및 큐잉
        const std::vector<std::unique_ptr<FrameBuffer>>& buffers = allocator->buffers(stream);
        std::vector<std::unique_ptr<Request>> requests;
        
        for (size_t i = 0; i < buffers.size(); ++i) {
            std::unique_ptr<Request> request = camera->createRequest();
            if (!request) {
                std::cerr << "Failed to create request " << i << std::endl;
                return false;
            }
            
            if (request->addBuffer(stream, buffers[i].get())) {
                std::cerr << "Failed to add buffer to request " << i << std::endl;
                return false;
            }
            
            requests.push_back(std::move(request));
        }
        
        // 콜백 설정
        camera->requestCompleted.connect(this, &DMABufZeroCopyCapture::onRequestCompleted);
        
        // 카메라 시작
        ControlList controls;
        // 30 FPS 목표로 프레임 지속시간 설정 (마이크로초 단위)
        controls.set(controls::FrameDurationLimits, 
                    {static_cast<int64_t>(1000000/30), static_cast<int64_t>(1000000/30)});
        
        if (camera->start(&controls)) {
            std::cerr << "Failed to start camera with controls, trying without controls..." << std::endl;
            // 컨트롤 없이 다시 시도
            if (camera->start()) {
                std::cerr << "Failed to start camera" << std::endl;
                return false;
            }
        } else {
            std::cout << "Camera started with 30 FPS target" << std::endl;
        }
        
        // 모든 요청을 큐에 추가
        for (auto& request : requests) {
            camera->queueRequest(request.release());
        }
        
        std::cout << "Capture started with " << buffers.size() << " buffers" << std::endl;
        return true;
    }
    
    void stop() {
        stopping.store(true);
        
        if (camera) {
            camera->stop();
            // 대기 중인 요청들이 완료될 때까지 잠시 대기
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            camera->release();
        }
        
        // 백그라운드 저장 스레드 종료
        shouldStop.store(true);
        saveCondition.notify_all();
        
        if (saveThread.joinable()) {
            saveThread.join();
        }
        
        printStatistics();
    }
    
    void cleanup() {
        stop();
        
        // 매핑된 메모리 해제
        for (size_t i = 0; i < mappedBuffers.size(); ++i) {
            if (mappedBuffers[i] != MAP_FAILED) {
                munmap(mappedBuffers[i], bufferSizes[i]);
            }
        }
        
        if (cameraManager) {
            cameraManager->stop();
        }
    }
    
    void waitForCompletion() {
        while (frameCount < targetFrames && !stopping.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
private:
    void saveWorker() {
        while (!shouldStop.load()) {
            std::unique_lock<std::mutex> lock(saveMutex);
            saveCondition.wait(lock, [this] { return !saveQueue.empty() || shouldStop.load(); });
            
            while (!saveQueue.empty()) {
                FrameData frameData = saveQueue.front();
                saveQueue.pop();
                lock.unlock();
                
                // 파일 저장 (I/O 성능에 영향 없이)
                std::string filename = "frame_" + std::to_string(frameData.frameIndex) + ".raw";
                std::ofstream file(filename, std::ios::binary);
                if (file.is_open()) {
                    file.write(reinterpret_cast<const char*>(frameData.data.data()), frameData.data.size());
                    file.close();
                }
                
                lock.lock();
            }
        }
    }
    
    void printStatistics() {
        if (frameStats.empty()) return;
        
        std::cout << "\n=== Performance Statistics ===" << std::endl;
        
        double totalIoTime = 0;
        double minIoTime = frameStats[0].ioTime;
        double maxIoTime = frameStats[0].ioTime;
        
        for (const auto& stats : frameStats) {
            totalIoTime += stats.ioTime;
            minIoTime = std::min(minIoTime, stats.ioTime);
            maxIoTime = std::max(maxIoTime, stats.ioTime);
        }
        
        double avgIoTime = totalIoTime / frameStats.size();
        
        auto totalTime = duration_cast<microseconds>(
            frameStats.back().processTime - frameStats.front().captureTime
        );
        double avgFps = frameStats.size() * 1000000.0 / totalTime.count();
        
        std::cout << "Total frames: " << frameStats.size() << std::endl;
        std::cout << "Average I/O time: " << avgIoTime << "ms" << std::endl;
        std::cout << "Min I/O time: " << minIoTime << "ms" << std::endl;
        std::cout << "Max I/O time: " << maxIoTime << "ms" << std::endl;
        std::cout << "Average FPS: " << avgFps << std::endl;
        std::cout << "Total capture time: " << totalTime.count() / 1000.0 << "ms" << std::endl;
    }
};

int main() {
    std::cout << "Starting DMA-BUF Zero-Copy Capture Demo" << std::endl;
    
    // 100프레임 캡처
    DMABufZeroCopyCapture capture(100);
    
    if (!capture.initialize()) {
        std::cerr << "Failed to initialize capture" << std::endl;
        return -1;
    }
    
    if (!capture.start()) {
        std::cerr << "Failed to start capture" << std::endl;
        return -1;
    }
    
    std::cout << "Capturing frames... Press Ctrl+C to stop early" << std::endl;
    
    // 캡처 완료까지 대기
    capture.waitForCompletion();
    
    std::cout << "Capture completed successfully!" << std::endl;
    return 0;
}
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
#include <sys/mman.h> // mmap, munmap
#include <algorithm>  // min, max
#include <libcamera/libcamera.h>

using namespace libcamera;
using namespace std::chrono;

// 프레임 통계를 위한 구조체
struct FrameStats {
    size_t frameIndex;
    high_resolution_clock::time_point captureTime;
    high_resolution_clock::time_point processTime;
    double ioTime;
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
        shouldStop.store(true);
        saveCondition.notify_all();
        if (saveThread.joinable()) {
            saveThread.join();
        }
        cleanup();
    }
    
    // 파일 저장 워커 스레드
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
            
            // 파일에 저장
            std::string filename = "frame_rgb_" + std::to_string(frameData.frameNumber) + 
                                "_" + std::to_string(frameData.width) + 
                                "x" + std::to_string(frameData.height) + 
                                ".rgb";
                                
            std::ofstream file(filename, std::ios::binary);
            if (file.is_open()) {
                file.write(reinterpret_cast<const char*>(frameData.data.data()), 
                          frameData.data.size());
                file.close();
                std::cout << "Saved RGB data: " << filename << 
                         " (" << frameData.data.size() << " bytes)" << std::endl;
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
        
        // 해상도를 1920x1080으로 설정
        streamConfig.size = Size(1920, 1080);
        
        // RGB888 포맷으로 직접 설정
        streamConfig.pixelFormat = libcamera::formats::RGB888;
        
        // 이미 설정됨 - 중복 제거
        
        std::cout << "Available formats:" << std::endl;
        // 사용 가능한 모든 포맷 출력
        for (const auto& format : streamConfig.formats().pixelformats()) {
            std::cout << "  - " << format.toString() << std::endl;
        }
        
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
        
        // RGB888 데이터를 직접 저장
        try {
            // 스트림 설정에서 해상도와 stride 정보 가져오기
            const StreamConfiguration& streamConfig = config->at(0);
            int width = streamConfig.size.width;
            int height = streamConfig.size.height;
            int stride = streamConfig.stride;
            
            std::cout << " [Format: " << streamConfig.pixelFormat.toString() 
                      << ", Size: " << width << "x" << height 
                      << ", Stride: " << stride << "]";
            
            std::cout << " [Planes: " << planeMappings.size() << "]";
            
            bool dataExtracted = false;
            std::vector<uint8_t> rgbData;
            size_t rgbSize = 0;
            
            // RGB888 형식 처리
            if (planeMappings.size() >= 1) {
                uint8_t* srcRGB = static_cast<uint8_t*>(planeMappings[0]);
                size_t dataSize = planeSizes[0];
                
                std::cout << " [RGB Data Size: " << dataSize << " bytes]";
                
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
                
                std::cout << " [RGB Byte Stride: " << effectiveStride << "]";
                
                // 줄별로 복사 (필요시 BGR -> RGB 변환)
                for (int y = 0; y < height; ++y) {
                    size_t srcOffset = y * effectiveStride;
                    size_t dstOffset = y * width * bytesPerPixel;
                    size_t lineSizeBytes = width * bytesPerPixel;
                    
                    // 범위를 벗어나지 않는지 확인
                    if (srcOffset + lineSizeBytes <= dataSize) {
                        // BGR -> RGB 순서로 변환하며 복사
                        for (int x = 0; x < width; ++x) {
                            size_t srcPixelOffset = srcOffset + x * bytesPerPixel;
                            size_t dstPixelOffset = dstOffset + x * bytesPerPixel;
                            
                            // libcamera의 "RGB888" 포맷은 실제로는 BGR 순서일 가능성이 높음
                            // BGR -> RGB 순서로 변환
                            rgbData[dstPixelOffset] = srcRGB[srcPixelOffset + 2];     // R <- B
                            rgbData[dstPixelOffset + 1] = srcRGB[srcPixelOffset + 1]; // G <- G
                            rgbData[dstPixelOffset + 2] = srcRGB[srcPixelOffset];     // B <- R
                        }
                    } else if (srcOffset < dataSize) {
                        // 일부만 복사 가능한 경우
                        int pixelsToProcess = (dataSize - srcOffset) / bytesPerPixel;
                        for (int x = 0; x < pixelsToProcess; ++x) {
                            size_t srcPixelOffset = srcOffset + x * bytesPerPixel;
                            size_t dstPixelOffset = dstOffset + x * bytesPerPixel;
                            
                            // BGR -> RGB 순서로 변환
                            rgbData[dstPixelOffset] = srcRGB[srcPixelOffset + 2];     // R <- B
                            rgbData[dstPixelOffset + 1] = srcRGB[srcPixelOffset + 1]; // G <- G
                            rgbData[dstPixelOffset + 2] = srcRGB[srcPixelOffset];     // B <- R
                        }
                        // 나머지는 0으로 채움
                        size_t processedBytes = pixelsToProcess * bytesPerPixel;
                        std::memset(rgbData.data() + dstOffset + processedBytes, 0, lineSizeBytes - processedBytes);
                    } else {
                        // 데이터 범위를 벗어난 경우 0으로 채움
                        std::memset(rgbData.data() + dstOffset, 0, lineSizeBytes);
                    }
                }
                
                // 첫 프레임의 RGB 데이터 샘플 출력 (채널 순서 확인용)
                if (frameCount == 0) {
                    std::cout << "\nRGB Data Sample - First 10 pixels (원본 순서로 출력):";
                    for (int i = 0; i < 10; i++) {
                        int index = i * 3;
                        std::cout << "\nPixel " << i << ": ("
                                  << (int)rgbData[index] << ","
                                  << (int)rgbData[index+1] << ","
                                  << (int)rgbData[index+2] << ")";
                    }
                    std::cout << std::endl;
                    
                    // RGB 조합으로 이미지가 파란색으로 보인다면, 채널 순서가 BGR일 가능성이 높음
                    // 첫 픽셀에 대해 가능한 모든 채널 조합을 출력해봄
                    std::cout << "\n첫 번째 픽셀의 가능한 채널 조합:";
                    std::cout << "\nRGB 순서: (" 
                              << (int)rgbData[0] << "," 
                              << (int)rgbData[1] << "," 
                              << (int)rgbData[2] << ")";
                    std::cout << "\nBGR 순서: (" 
                              << (int)rgbData[2] << "," 
                              << (int)rgbData[1] << "," 
                              << (int)rgbData[0] << ")";
                    std::cout << std::endl;
                }
                
                dataExtracted = true;
            }
            
            // RGB 데이터를 파일로 저장
            if (dataExtracted) {
                std::string filename = "frame_rgb_" + std::to_string(frameCount) + 
                                     "_" + std::to_string(width) + 
                                     "x" + std::to_string(height) + 
                                     ".rgb";
                
                // 동기적으로 파일에 저장
                std::ofstream file(filename, std::ios::binary);
                if (file.is_open()) {
                    file.write(reinterpret_cast<const char*>(rgbData.data()), rgbSize);
                    file.close();
                    std::cout << " [Saved RGB: " << filename << " (" << rgbSize << " bytes)]";
                    
                    // PNG로 변환을 위한 메타데이터 파일 생성
                    std::string metaFilename = "frame_rgb_" + std::to_string(frameCount) + "_meta.txt";
                    std::ofstream metaFile(metaFilename);
                    if (metaFile.is_open()) {
                        metaFile << "Frame: " << frameCount << std::endl;
                        metaFile << "Format: RGB888 (BGR->RGB 변환)" << std::endl;
                        metaFile << "Width: " << width << std::endl;
                        metaFile << "Height: " << height << std::endl;
                        metaFile << "RGB Size: " << rgbSize << " bytes" << std::endl;
                        metaFile << "Bytes per pixel: 3" << std::endl;
                        metaFile << "Channel order: R,G,B (카메라에서는 B,G,R로 제공됨)" << std::endl;
                        metaFile << "Data source: Camera BGR888 output converted to RGB888" << std::endl;
                        metaFile.close();
                    }
                }
                
                // PPM 파일로도 저장 (이미지 뷰어에서 바로 확인 가능)
                if (frameCount % 10 == 0) {  // 10프레임마다 PPM 파일 생성
                    std::string ppmFilename = "frame_rgb_" + std::to_string(frameCount) + 
                                           "_" + std::to_string(width) + 
                                           "x" + std::to_string(height) + 
                                           ".ppm";
                    
                    std::ofstream ppmFile(ppmFilename, std::ios::binary);
                    if (ppmFile.is_open()) {
                        // PPM 헤더 작성
                        ppmFile << "P6\n" << width << " " << height << "\n255\n";
                        // RGB 데이터 저장
                        ppmFile.write(reinterpret_cast<const char*>(rgbData.data()), rgbSize);
                        ppmFile.close();
                        std::cout << " [Saved PPM: " << ppmFilename << "]";
                    }
                }
            }
            
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
        
        double totalIoTime = 0;
        for (const auto& stats : frameStats) {
            totalIoTime += stats.ioTime;
        }
        
        double avgIoTime = totalIoTime / frameStats.size();
        auto totalTime = duration_cast<milliseconds>(
            frameStats.back().processTime - startTime).count();
        double avgFps = frameStats.size() * 1000.0 / totalTime;
        
        std::cout << "\nCapture Statistics:" << std::endl;
        std::cout << "Total frames: " << frameStats.size() << std::endl;
        std::cout << "Average FPS: " << avgFps << std::endl;
        std::cout << "Average I/O time: " << avgIoTime << " ms" << std::endl;
        std::cout << "Total time: " << totalTime / 1000.0 << " seconds" << std::endl;
    }
};

int main(int argc, char** argv) {
    size_t numFrames = 100; // 기본값
    
    // 명령행 인수가 있으면 프레임 수 조정
    if (argc > 1) {
        numFrames = std::stoul(argv[1]);
    }
    
    std::cout << "Starting DMA-BUF zero-copy capture with direct RGB888 format..." << std::endl;
    std::cout << "Target frames: " << numFrames << std::endl;
    
    DMABufZeroCopyCapture capture(numFrames);
    
    if (!capture.initialize()) {
        std::cerr << "Failed to initialize capture" << std::endl;
        return 1;
    }
    
    if (!capture.start()) {
        std::cerr << "Failed to start capture" << std::endl;
        return 1;
    }
    
    // 메인 스레드는 대기 - 이벤트 기반 처리
    std::cout << "Capturing... Press Ctrl+C to stop." << std::endl;
    
    // 사용자가 Ctrl+C를 누를 때까지 대기
    try {
        // 간단한 시그널 처리
        signal(SIGINT, [](int) { /* 무시 */ });
        
        // 모든 프레임이 처리될 때까지 대기
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } catch (...) {
        // 예외 발생 시 정리
    }
    
    // 종료 및 통계 출력
    capture.printStats();
    
    std::cout << "RGB 파일을 PNG로 변환하려면 다음 명령을 사용하세요:" << std::endl;
    std::cout << "for file in frame_rgb_*_1920x1080.rgb; do" << std::endl;
    std::cout << "  ffmpeg -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -i \"$file\" \"${file%.rgb}.png\"" << std::endl;
    std::cout << "done" << std::endl;
    
    return 0;
}

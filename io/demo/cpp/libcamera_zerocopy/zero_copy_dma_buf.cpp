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
        
        // 픽셀 포맷을 명시적으로 NV12로 설정
        // NV12는 Y 평면 뒤에 UV 인터리브 평면이 있는 형식
        streamConfig.pixelFormat = libcamera::formats::NV12;
        
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
            
            // YUV420은 보통 3개의 평면(Y, U, V)을 가짐
            std::vector<void*> planeMappings;
            std::vector<size_t> planeSizes;
            
            for (size_t planeIndex = 0; planeIndex < buffer->planes().size(); ++planeIndex) {
                const FrameBuffer::Plane& plane = buffer->planes()[planeIndex];
                int dmafd = plane.fd.get();
                size_t length = plane.length;
                
                // DMA-BUF를 메모리에 매핑 (zero-copy)
                void* mapped = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, dmafd, 0);
                if (mapped == MAP_FAILED) {
                    std::cerr << "Failed to mmap DMA-BUF " << i << " plane " << planeIndex 
                              << ": " << strerror(errno) << std::endl;
                    return false;
                }
                
                planeMappings.push_back(mapped);
                planeSizes.push_back(length);
                
                std::cout << "  Plane " << planeIndex << " mapped: " << mapped 
                          << ", size: " << length << " bytes"
                          << ", fd: " << dmafd 
                          << ", offset: " << plane.offset << std::endl;
                          
                // DMA-BUF 매핑 검증 - 첫 몇 바이트 출력
                if (planeIndex < 2) { // Y 또는 UV 평면인 경우만
                    std::cout << "    First 8 bytes: ";
                    uint8_t* data = static_cast<uint8_t*>(mapped);
                    for (int j = 0; j < 8 && j < length; ++j) {
                        std::cout << (int)data[j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
            
            // 각 버퍼의 모든 평면을 저장
            bufferPlaneMappings.push_back(planeMappings);
            bufferPlaneSizes.push_back(planeSizes);
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
        
        // YUV420 컬러 데이터를 표준 형식으로 파일 저장
        // 모든 프레임을 저장 (원래 if (frameCount % 10 == 0) 조건 제거)
        try {
            // 스트림 설정에서 해상도와 stride 정보 가져오기
            const StreamConfiguration& streamConfig = config->at(0);
            int width = streamConfig.size.width;
            int height = streamConfig.size.height;
            int stride = streamConfig.stride;
            
            std::cout << " [Format: " << streamConfig.pixelFormat.toString() 
                      << ", Size: " << width << "x" << height 
                      << ", Stride: " << stride << "]";
            
            // 표준 YUV420P 크기 계산
            size_t outputYSize = width * height;
            size_t outputUVSize = (width / 2) * (height / 2);
            size_t standardYuvSize = outputYSize + outputUVSize + outputUVSize;
            
            std::cout << " [Planes: " << planeMappings.size() << "]";
            
            // 표준 YUV420P 버퍼 생성
            std::vector<uint8_t> standardYuv(standardYuvSize);
            uint8_t* dstY = standardYuv.data();
            uint8_t* dstU = standardYuv.data() + outputYSize;
            uint8_t* dstV = standardYuv.data() + outputYSize + outputUVSize;
            
            bool colorExtracted = false;
            
            if (planeMappings.size() >= 3 && planeSizes.size() >= 3) {
                // 3개 평면 (Y, U, V)이 분리된 경우
                uint8_t* srcY = static_cast<uint8_t*>(planeMappings[0]);
                uint8_t* srcU = static_cast<uint8_t*>(planeMappings[1]);
                uint8_t* srcV = static_cast<uint8_t*>(planeMappings[2]);
                
                size_t ySizeAvailable = planeSizes[0];
                size_t uSizeAvailable = planeSizes[1];
                size_t vSizeAvailable = planeSizes[2];
                
                std::cout << " [Y:" << ySizeAvailable << ", U:" << uSizeAvailable << ", V:" << vSizeAvailable << " bytes]";                    // Y 평면 복사
                    if (stride == width) {
                        size_t bytesToCopy = ySizeAvailable < outputYSize ? ySizeAvailable : outputYSize;
                        std::memcpy(dstY, srcY, bytesToCopy);
                    } else {
                        for (int y = 0; y < height && y * stride < ySizeAvailable; ++y) {
                            size_t bytesToCopy = (ySizeAvailable - y * stride < width) ? 
                                               (ySizeAvailable - y * stride) : width;
                            std::memcpy(dstY + y * width, 
                                       srcY + y * stride, 
                                       bytesToCopy);
                        }
                    }
                
                // U, V 평면 복사
                size_t uvWidth = width / 2;
                size_t uvHeight = height / 2;
                size_t uvStride = uvWidth; // 대개 UV 평면의 stride도 절반임                    // U 채널 복사
                    if (uvStride == uvWidth) {
                        std::memcpy(dstU, srcU, std::min(uSizeAvailable, outputUVSize));
                    } else {
                        for (int y = 0; y < uvHeight && y * uvStride < uSizeAvailable; ++y) {
                            size_t bytesToCopy = (uSizeAvailable - y * uvStride < uvWidth) ? 
                                               (uSizeAvailable - y * uvStride) : uvWidth;
                            std::memcpy(dstU + y * uvWidth, 
                                       srcU + y * uvStride, 
                                       bytesToCopy);
                        }
                    }
                    
                    // V 채널 복사
                    if (uvStride == uvWidth) {
                        std::memcpy(dstV, srcV, std::min(vSizeAvailable, outputUVSize));
                    } else {
                        for (int y = 0; y < uvHeight && y * uvStride < vSizeAvailable; ++y) {
                            size_t bytesToCopy = (vSizeAvailable - y * uvStride < uvWidth) ? 
                                               (vSizeAvailable - y * uvStride) : uvWidth;
                            std::memcpy(dstV + y * uvWidth, 
                                       srcV + y * uvStride, 
                                       bytesToCopy);
                        }
                    }
                
                colorExtracted = true;                } 
                else if (planeMappings.size() >= 2 && planeSizes.size() >= 2) {
                    // 2개 평면 (Y, UV interleaved) - NV12 형식
                    uint8_t* srcY = static_cast<uint8_t*>(planeMappings[0]);
                    uint8_t* srcUV = static_cast<uint8_t*>(planeMappings[1]);
                    
                    size_t ySizeAvailable = planeSizes[0];
                    size_t uvSizeAvailable = planeSizes[1];
                    
                    std::cout << " [Y: " << ySizeAvailable << ", UV: " << uvSizeAvailable << " bytes]";
                    
                    // Y 평면 복사
                    if (stride == width) {
                        std::memcpy(dstY, srcY, std::min(ySizeAvailable, outputYSize));
                    } else {
                        for (int y = 0; y < height; ++y) {
                            if (static_cast<size_t>(y * stride) < ySizeAvailable) {
                                size_t bytesToCopy = (ySizeAvailable - y * stride) < static_cast<size_t>(width) ? 
                                                (ySizeAvailable - y * stride) : width;
                                std::memcpy(dstY + y * width, 
                                        srcY + y * stride, 
                                        bytesToCopy);
                            }
                        }
                    }
                    
                    // NV12 데이터 디버깅 - 처음 20바이트 출력
                    if (frameCount == 0 || frameCount == 1) {
                        std::cout << " [NV12 Debug: UV[0-19]={";
                        for (int i = 0; i < 20 && i < uvSizeAvailable; ++i) {
                            std::cout << (int)srcUV[i];
                            if (i < 19) std::cout << ",";
                        }
                        std::cout << "}]";
                    }
                    
                    // 라즈베리파이 카메라는 NV12 형식으로 UV가 인터리브됨 (UVUV...)
                    // NV12에서는 U,V 채널이 인터리브되어 있고, 각각 2x2 픽셀당 하나의 값을 가짐
                    size_t uvWidth = width / 2;  // U,V는 수평 방향으로 절반의 해상도
                    size_t uvHeight = height / 2; // U,V는 수직 방향으로 절반의 해상도
                    size_t uvSize = uvWidth * uvHeight;
                    
                    // NV12에서 UV 평면의 stride는 원본 폭과 같을 수 있음 (Y 평면의 절반이 아님)
                    size_t uvStride = stride; // 전체 stride
                    
                    // 디버그: UV 평면 크기와 처리할 UV 데이터 크기 출력
                    std::cout << " [UV Size: " << uvSize << ", UV Width: " << uvWidth 
                              << ", UV Height: " << uvHeight << ", UV Stride: " << uvStride << "]";
                              
                    // 순차적으로 U/V 분리 - NV12에서는 U, V가 인터리브됨 (UVUVUV...)
                    for (int y = 0; y < uvHeight; ++y) {
                        for (int x = 0; x < uvWidth; ++x) {
                            // NV12 형식에서 각 픽셀에 대한 UV 인덱스 계산
                            // 실제 stride가 가로 크기의 정확히 2배가 아닐 수 있으므로 stride 사용
                            size_t uvOffset = y * stride + x * 2; // NV12에서 U와 V는 인터리브됨
                            
                            if (uvOffset + 1 < uvSizeAvailable) {
                                dstU[y * uvWidth + x] = srcUV[uvOffset];     // U (짝수 인덱스)
                                dstV[y * uvWidth + x] = srcUV[uvOffset + 1]; // V (홀수 인덱스)
                            } else {
                                // 범위를 벗어나면 회색조(128)로 설정
                                dstU[y * uvWidth + x] = 128;
                                dstV[y * uvWidth + x] = 128;
                            }
                        }
                    }
                    
                    // 추가 디버깅 - 평면별 처리 후 샘플 데이터 출력
                    if (frameCount == 0) {
                        std::cout << "\nAfter processing - First 10 bytes of each plane:";
                        std::cout << "\nY: ";
                        for (int i = 0; i < 10 && i < outputYSize; i++) 
                            std::cout << (int)dstY[i] << " ";
                        
                        std::cout << "\nU: ";
                        for (int i = 0; i < 10 && i < outputUVSize; i++) 
                            std::cout << (int)dstU[i] << " ";
                            
                        std::cout << "\nV: ";
                        for (int i = 0; i < 10 && i < outputUVSize; i++) 
                            std::cout << (int)dstV[i] << " ";
                        std::cout << std::endl;
                        
                        // NV21 테스트 - U와 V 채널을 교환하여 별도 파일로 저장
                        // 실제 NV12와 NV21 구분이 애매할 경우 두 가지 방식 모두 시도
                        std::string filename_nv21 = "frame_" + std::to_string(frameCount) + 
                                           "_" + std::to_string(width) + 
                                           "x" + std::to_string(height) + 
                                           "_yuv420p_nv21.yuv";
                        
                        std::vector<uint8_t> nv21Data(standardYuvSize);
                        // Y 데이터는 그대로 복사
                        std::memcpy(nv21Data.data(), standardYuv.data(), outputYSize);
                        
                        // U와 V 데이터 위치를 서로 바꿔서 복사 (NV21 테스트)
                        std::memcpy(nv21Data.data() + outputYSize, 
                                   standardYuv.data() + outputYSize + outputUVSize, // V 위치
                                   outputUVSize);
                        std::memcpy(nv21Data.data() + outputYSize + outputUVSize, 
                                   standardYuv.data() + outputYSize, // U 위치
                                   outputUVSize);
                        
                        std::ofstream file_nv21(filename_nv21, std::ios::binary);
                        if (file_nv21.is_open()) {
                            file_nv21.write(reinterpret_cast<const char*>(nv21Data.data()), standardYuvSize);
                            file_nv21.close();
                            std::cout << " [Saved NV21 Test: " << filename_nv21 << "]";
                        }
                    }
                    
                    colorExtracted = true;
                
            } else if (planeMappings.size() >= 1) {
                // 1개 평면 - packed YUV 또는 Y-only
                uint8_t* srcData = static_cast<uint8_t*>(planeMappings[0]);
                size_t dataSize = planeSizes[0];
                
                std::cout << " [Single plane: " << dataSize << " bytes]";
                
                // Y 데이터만 추출하고 U, V는 중립값(128)으로 설정
                if (stride == width) {
                    size_t bytesToCopy = dataSize < outputYSize ? dataSize : outputYSize;
                    std::memcpy(dstY, srcData, bytesToCopy);
                } else {
                    for (int y = 0; y < height && y * stride < dataSize; ++y) {
                        size_t bytesToCopy = (dataSize - y * stride < width) ? 
                                           (dataSize - y * stride) : width;
                        std::memcpy(dstY + y * width, 
                                   srcData + y * stride, 
                                   bytesToCopy);
                    }
                }
                
                // U와 V를 중립값(128)으로 설정하여 그레이스케일로 표시
                std::memset(dstU, 128, outputUVSize);
                std::memset(dstV, 128, outputUVSize);
                
                colorExtracted = true;
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
                    metaFile << "Stride: " << stride << std::endl;
                    metaFile << "Planes: " << planeMappings.size() << std::endl;
                    metaFile << "Standard YUV Size: " << standardYuvSize << " bytes" << std::endl;
                    metaFile << "Y Size: " << outputYSize << " bytes (stride: " << width << ")" << std::endl;
                    metaFile << "U Size: " << outputUVSize << " bytes (stride: " << (width/2) << ")" << std::endl;
                    metaFile << "V Size: " << outputUVSize << " bytes (stride: " << (width/2) << ")" << std::endl;
                    for (size_t i = 0; i < planeSizes.size(); ++i) {
                        metaFile << "Plane " << i << " size: " << planeSizes[i] << " bytes" << std::endl;
                    }
                    metaFile << "Y plane offset: 0" << std::endl;
                    metaFile << "U plane offset: " << outputYSize << std::endl;
                    metaFile << "V plane offset: " << (outputYSize + outputUVSize) << std::endl;
                    metaFile << "Color extracted: " << (colorExtracted ? "Yes" : "No") << std::endl;
                    metaFile.close();
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during image processing: " << e.what() << std::endl;
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
        
        // 매핑된 메모리 해제 - 다중 평면 처리
        for (size_t i = 0; i < bufferPlaneMappings.size(); ++i) {
            for (size_t j = 0; j < bufferPlaneMappings[i].size(); ++j) {
                if (bufferPlaneMappings[i][j] != MAP_FAILED) {
                    munmap(bufferPlaneMappings[i][j], bufferPlaneSizes[i][j]);
                }
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
        try {
            while (!shouldStop.load()) {
                std::unique_lock<std::mutex> lock(saveMutex);
                saveCondition.wait(lock, [this] { return !saveQueue.empty() || shouldStop.load(); });
                
                // shouldStop이 true이고 큐가 비어있으면 루프 종료
                if (shouldStop.load() && saveQueue.empty()) {
                    break;
                }
                
                // 큐가 비어있지 않을 때만 처리
                if (!saveQueue.empty()) {
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
        } catch (const std::exception& e) {
            std::cerr << "saveWorker 예외 발생: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "saveWorker 알 수 없는 예외 발생" << std::endl;
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
    
    // 10프레임 캡처 (테스트용) - 모든 프레임이 저장됨
    DMABufZeroCopyCapture capture(10);
    
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
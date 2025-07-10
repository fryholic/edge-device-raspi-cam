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
#include <array>
#include <sys/mman.h>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <limits>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/control_ids.h>
#include <libcamera/stream.h>

// OpenCV 헤더
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/ocl.hpp> // OpenCL 지원을 위해 필수

using namespace libcamera;
using namespace std::chrono;
using namespace std::literals::chrono_literals;

// 프레임 통계를 위한 구조체
struct FrameStats {
    size_t frameIndex;
    high_resolution_clock::time_point captureTime;
    high_resolution_clock::time_point processEndTime;
    double totalProcessingTime; // 전체 처리 시간 (I/O + OpenCV)
    double imageProcessingTime; // 순수 이미지 처리 시간 (GPU/CPU)
    double instantFps;
    double avgFps;
};

class DMABufOpenCVDemo {
private:
    // 카메라 및 버퍼 관련
    std::shared_ptr<Camera> camera;
    std::unique_ptr<CameraManager> cameraManager;
    std::unique_ptr<CameraConfiguration> config;
    Stream* stream;
    ControlList cameraControls;
    std::shared_ptr<FrameBufferAllocator> allocator;
    std::vector<std::vector<void*>> bufferPlaneMappings;
    std::vector<std::vector<size_t>> bufferPlaneSizes;

    // 성능 측정
    high_resolution_clock::time_point startTime;
    std::vector<FrameStats> frameStats;
    size_t frameCount;

    // 기능 플래그
    bool verboseOutput;
    bool enableOpenCV;
    bool saveFrames;
    bool useGPU;

    // 종료 플래그
    std::atomic<bool> stopping;

public:
    DMABufOpenCVDemo(bool verbose = true, bool opencv = true, bool save = false)
        : stream(nullptr), frameCount(0), verboseOutput(verbose),
          enableOpenCV(opencv), saveFrames(save), useGPU(false), stopping(false) {

        // OpenCV 하드웨어 가속 설정
        if (enableOpenCV) {
            cv::setUseOptimized(true);
            
            if (cv::ocl::haveOpenCL()) {
                std::cout << "OpenCL 지원 확인: 사용 가능" << std::endl;
                cv::ocl::setUseOpenCL(true);
                if (!cv::ocl::useOpenCL() || cv::ocl::Context::getDefault().ptr() == NULL) {
                    std::cout << "경고: OpenCL을 사용할 수 있도록 설정했지만, 컨텍스트를 초기화할 수 없습니다. CPU로 대체합니다." << std::endl;
                    useGPU = false;
                } else {
                    useGPU = true;
                    cv::ocl::Device defaultDevice = cv::ocl::Device::getDefault();
                    std::cout << "  - 기본 OpenCL 장치: " << defaultDevice.name() << std::endl;
                }
            } else {
                std::cout << "OpenCL 지원 확인: 사용 불가. CPU 기반 최적화로 대체합니다." << std::endl;
                useGPU = false;
            }
        }
    }

    ~DMABufOpenCVDemo() {
        cleanup();
    }

    bool initialize() {
        cameraManager = std::make_unique<CameraManager>();
        int ret = cameraManager->start();
        if (ret) {
            std::cerr << "카메라 매니저 시작 실패: " << ret << std::endl;
            return false;
        }

        auto cameras = cameraManager->cameras();
        if (cameras.empty()) {
            std::cerr << "카메라를 찾을 수 없습니다." << std::endl;
            return false;
        }

        camera = cameras[0];
        if (camera->acquire()) {
            std::cerr << "카메라 획득 실패" << std::endl;
            return false;
        }
        std::cout << "사용 카메라: " << camera->id() << std::endl;

        config = camera->generateConfiguration({StreamRole::Viewfinder});
        StreamConfiguration& streamConfig = config->at(0);
        
        streamConfig.size = Size(1920, 1080);
        streamConfig.pixelFormat = libcamera::formats::BGR888;
        streamConfig.bufferCount = 4;

        config->validate();
        
        // ========================[ 수정된 부분 ]========================
        // 충돌을 유발하는 컨트롤 설정을 비활성화합니다.
        // 이 컨트롤들은 libcamera 버전이나 카메라 펌웨어에 따라 지원되지 않을 수 있습니다.
        // cameraControls.set(controls::FrameDurationLimits, {33333, 33333}); // 충돌의 주 원인
        // cameraControls.set(controls::AeEnable, false);
        // cameraControls.set(controls::AwbEnable, false);
        // cameraControls.set(controls::ExposureTime, 10000);
        // ===============================================================

        if (camera->configure(config.get())) {
            std::cerr << "카메라 설정 실패" << std::endl;
            return false;
        }

        stream = streamConfig.stream();
        std::cout << "스트림 설정 완료:" << std::endl;
        std::cout << "  - 해상도: " << streamConfig.size.width << "x" << streamConfig.size.height << std::endl;
        std::cout << "  - 픽셀 포맷: " << streamConfig.pixelFormat.toString() << std::endl;

        return setupBuffers();
    }

    bool setupBuffers() {
        allocator = std::make_shared<FrameBufferAllocator>(camera);
        if (allocator->allocate(stream) < 0) {
            std::cerr << "버퍼 할당 실패" << std::endl;
            return false;
        }

        const auto& buffers = allocator->buffers(stream);
        bufferPlaneMappings.resize(buffers.size());
        bufferPlaneSizes.resize(buffers.size());

        for (size_t i = 0; i < buffers.size(); ++i) {
            for (size_t j = 0; j < buffers[i]->planes().size(); ++j) {
                const auto& plane = buffers[i]->planes()[j];
                void* memory = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
                if (memory == MAP_FAILED) {
                    std::cerr << "버퍼 매핑 실패 (mmap)" << std::endl;
                    return false;
                }
                bufferPlaneMappings[i].push_back(memory);
                bufferPlaneSizes[i].push_back(plane.length);
            }
        }
        std::cout << buffers.size() << "개의 DMA-BUF 버퍼 설정 완료" << std::endl;
        return true;
    }

    void cleanup() {
        if (allocator) {
            const auto& buffers = allocator->buffers(stream);
            for (size_t i = 0; i < bufferPlaneMappings.size(); ++i) {
                for (size_t j = 0; j < bufferPlaneMappings[i].size(); ++j) {
                     if (i < buffers.size() && j < buffers[i]->planes().size()) {
                        munmap(bufferPlaneMappings[i][j], buffers[i]->planes()[j].length);
                    }
                }
            }
        }
        if (camera) {
            camera->release();
        }
    }

    bool start() {
        camera->requestCompleted.connect(this, &DMABufOpenCVDemo::onRequestCompleted);
        if (camera->start(&cameraControls)) {
            std::cerr << "카메라 시작 실패" << std::endl;
            return false;
        }

        for (const auto& buffer : allocator->buffers(stream)) {
            std::unique_ptr<Request> request = camera->createRequest();
            if (!request || request->addBuffer(stream, buffer.get())) {
                 std::cerr << "요청 생성 또는 버퍼 추가 실패" << std::endl;
                 return false;
            }
            camera->queueRequest(request.release());
        }
        
        startTime = high_resolution_clock::now();
        std::cout << "캡처 시작..." << std::endl;
        return true;
    }

    void stop() {
        stopping.store(true);
        if (camera) {
            camera->stop();
            camera->requestCompleted.disconnect(this, &DMABufOpenCVDemo::onRequestCompleted);
        }
    }

    cv::UMat processWithGPU(const cv::Mat& inputCpuMat) {
        cv::UMat imageGpu = inputCpuMat.getUMat(cv::ACCESS_READ);
        cv::UMat resultGpu;
        imageGpu.convertTo(resultGpu, -1, 1.1, 10);
        cv::UMat blurredGpu;
        cv::GaussianBlur(resultGpu, blurredGpu, cv::Size(0, 0), 3);
        cv::addWeighted(resultGpu, 1.5, blurredGpu, -0.5, 0, resultGpu);
        return resultGpu;
    }

    void onRequestCompleted(Request* request) {
        if (stopping.load() || request->status() != Request::RequestComplete) {
            if (!stopping.load()) std::cerr << "요청 실패: " << request->status() << std::endl;
            request->reuse(Request::ReuseFlag::ReuseBuffers);
            camera->queueRequest(request);
            return;
        }

        auto processStartTime = high_resolution_clock::now();
        FrameBuffer* buffer = request->buffers().begin()->second;

        const auto& buffers = allocator->buffers(stream);
        size_t bufferIndex = std::distance(buffers.begin(), 
            std::find_if(buffers.begin(), buffers.end(), 
                         [buffer](const auto& b){ return b.get() == buffer; }));

        void* data = bufferPlaneMappings[bufferIndex][0];
        const auto& streamConfig = config->at(0);
        cv::Mat frame(streamConfig.size.height, streamConfig.size.width, CV_8UC3, data, streamConfig.stride);
        
        double imageProcessingTimeMs = 0;

        if (enableOpenCV) {
            auto cvStart = high_resolution_clock::now();
            
            if (useGPU) {
                cv::UMat processedGpu = processWithGPU(frame);
                cv::ocl::finish(); 
                if (saveFrames && (frameCount % 30 == 0)) {
                    cv::Mat processedCpu = processedGpu.getMat(cv::ACCESS_READ);
                    cv::imwrite("enhanced_frame_" + std::to_string(frameCount) + ".jpg", processedCpu);
                }
            } else {
                // CPU 처리 (GPU 사용 불가 시)
            }

            auto cvEnd = high_resolution_clock::now();
            imageProcessingTimeMs = duration_cast<microseconds>(cvEnd - cvStart).count() / 1000.0;
        }

        auto processEndTime = high_resolution_clock::now();
        double totalProcessingTimeMs = duration_cast<microseconds>(processEndTime - processStartTime).count() / 1000.0;
        
        double avgFps = 0;
        if (frameCount > 0) {
            auto totalElapsed = duration_cast<microseconds>(processEndTime - startTime);
            avgFps = frameCount * 1000000.0 / totalElapsed.count();
        }

        frameStats.push_back({
            frameCount, processStartTime, processEndTime,
            totalProcessingTimeMs, imageProcessingTimeMs, 0.0, avgFps
        });

        if (verboseOutput && frameCount % 30 == 0) {
            std::cout << "\r" << std::fixed << std::setprecision(1)
                      << "Frame: " << std::setw(5) << frameCount
                      << " | Avg FPS: " << std::setw(5) << avgFps
                      << " | Total Time: " << std::setw(5) << totalProcessingTimeMs << "ms"
                      << " | ImgProc (" << (useGPU ? "GPU" : "CPU") << "): " << std::setw(5) << imageProcessingTimeMs << "ms  ";
            std::cout.flush();
        }

        frameCount++;
        request->reuse(Request::ReuseFlag::ReuseBuffers);
        if (!stopping.load()) {
            camera->queueRequest(request);
        }
    }

    void printFinalStats() {
        if (frameStats.empty()) return;

        double totalProcTimeSum = 0;
        double imgProcTimeSum = 0;
        double maxImgProcTime = 0;
        double minImgProcTime = std::numeric_limits<double>::max();

        for (const auto& s : frameStats) {
            totalProcTimeSum += s.totalProcessingTime;
            if (enableOpenCV) {
                imgProcTimeSum += s.imageProcessingTime;
                maxImgProcTime = std::max(maxImgProcTime, s.imageProcessingTime);
                minImgProcTime = std::min(minImgProcTime, s.imageProcessingTime);
            }
        }

        double avgTotalTime = totalProcTimeSum / frameStats.size();
        double avgImgProcTime = enableOpenCV ? (imgProcTimeSum / frameStats.size()) : 0;
        double finalAvgFps = frameStats.back().avgFps;

        std::cout << "\n\n================ 최종 성능 통계 ================" << std::endl;
        std::cout << "총 실행 시간: " << duration_cast<milliseconds>(frameStats.back().processEndTime - startTime).count() / 1000.0 << " 초" << std::endl;
        std::cout << "총 처리 프레임: " << frameStats.size() << " 프레임" << std::endl;
        std::cout << "최종 평균 FPS: " << std::fixed << std::setprecision(1) << finalAvgFps << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "평균 프레임 처리 시간 (Total): " << std::fixed << std::setprecision(2) << avgTotalTime << " ms" << std::endl;
        if (enableOpenCV) {
            std::cout << "이미지 처리 엔진: " << (useGPU ? "GPU (OpenCL)" : "CPU") << std::endl;
            std::cout << "  - 평균 처리 시간: " << avgImgProcTime << " ms" << std::endl;
            std::cout << "  - 최소 처리 시간: " << minImgProcTime << " ms" << std::endl;
            std::cout << "  - 최대 처리 시간: " << maxImgProcTime << " ms" << std::endl;
        }
        std::cout << "================================================" << std::endl;

        std::cout << "\n성능 평가:" << std::endl;
        if (enableOpenCV) {
            if (avgImgProcTime <= 20.0) {
                std::cout << "🎉 목표 달성! 이미지 처리 시간이 " << avgImgProcTime << "ms로 매우 빠릅니다." << std::endl;
            } else {
                std::cout << "⚠️ 목표 미달. 이미지 처리 시간이 " << avgImgProcTime << "ms입니다. 추가 최적화가 필요할 수 있습니다." << std::endl;
            }
        }
        if (finalAvgFps >= 29.5) {
            std::cout << "✅ 실시간 처리 성공! 평균 " << finalAvgFps << " FPS를 달성했습니다." << std::endl;
        } else {
            std::cout << "❌ 실시간 처리 실패. 평균 " << finalAvgFps << " FPS로, 30 FPS에 미치지 못했습니다." << std::endl;
        }
    }
};

static std::atomic<bool> shouldExit{false};
static DMABufOpenCVDemo* demoInstance = nullptr;

void signalHandler(int signal) {
    std::cout << "\n\n신호 " << signal << " 수신. 종료를 시작합니다..." << std::endl;
    shouldExit.store(true);
    if (demoInstance) {
        demoInstance->stop();
    }
}

int main(int argc, char** argv) {
    bool verbose = true;
    bool opencv = true;
    bool save = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-opencv") opencv = false;
        else if (arg == "--save-frames") save = true;
    }

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::cout << "=========================================================" << std::endl;
    std::cout << "GPU 가속 Zero-Copy OpenCV 데모 (최적화 버전)" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "- 이미지 처리: " << (opencv ? "활성화 (GPU 가속)" : "비활성화") << std::endl;
    std::cout << "- 프레임 저장: " << (save ? "활성화 (성능 저하 유발)" : "비활성화") << std::endl;
    std::cout << "=========================================================" << std::endl;

    try {
        DMABufOpenCVDemo demo(verbose, opencv, save);
        demoInstance = &demo;

        if (!demo.initialize()) return -1;
        if (!demo.start()) return -1;

        while (!shouldExit.load()) {
            std::this_thread::sleep_for(100ms);
        }

        demo.printFinalStats();

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV 오류: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "일반 오류: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "프로그램을 성공적으로 종료했습니다." << std::endl;
    return 0;
}

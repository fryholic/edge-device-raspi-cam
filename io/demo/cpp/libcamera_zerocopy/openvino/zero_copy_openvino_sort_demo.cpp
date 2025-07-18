#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <sys/mman.h>
#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/stream.h>
// 트래킹 관련 헤더 추가
#include "sort.hpp"
#include "object_tracker.hpp"

// --- OpenVINO YOLO 추론 클래스 ---
class OpenVINOYOLO {
public:
    OpenVINOYOLO(const std::string& model_xml, const std::string& device = "CPU") {
        core = std::make_shared<ov::Core>();
        model = core->read_model(model_xml);
        compiled_model = core->compile_model(model, device);
        infer_request = compiled_model.create_infer_request();
        input_port = compiled_model.input();
        output_port = compiled_model.outputs()[0];
        std::cout << "OpenVINO YOLO 모델 초기화 완료" << std::endl;
        std::cout << "입력 크기: " << input_port.get_shape()[3] << "x" << input_port.get_shape()[2] << std::endl;
    }
    struct Detection {
        cv::Rect bbox;
        int class_id;
        float confidence;
    };
    std::vector<Detection> detections;
    void infer(const cv::Mat& frame) {
        detections.clear();
        cv::Mat resized, rgb;
        cv::resize(frame, resized, cv::Size(input_port.get_shape()[3], input_port.get_shape()[2]));
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255);
        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), rgb.data);
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        auto output = infer_request.get_output_tensor(0);
        const float* data = output.data<float>();
        const ov::Shape& shape = output.get_shape();
        size_t numDet = shape[1];
        size_t numElem = shape[2];
        for (size_t i = 0; i < numDet; ++i) {
            const float* det = data + i * numElem;
            float conf = det[4];
            if (conf < 0.3) continue;
            float maxScore = 0;
            int classId = -1;
            for (int c = 5; c < numElem; ++c) {
                if (det[c] > maxScore) {
                    maxScore = det[c];
                    classId = c - 5;
                }
            }
            if (maxScore * conf < 0.3) continue;
            float cx = det[0], cy = det[1], w = det[2], h = det[3];
            int left = static_cast<int>((cx - w/2) * frame.cols / input_port.get_shape()[3]);
            int top = static_cast<int>((cy - h/2) * frame.rows / input_port.get_shape()[2]);
            int width = static_cast<int>(w * frame.cols / input_port.get_shape()[3]);
            int height = static_cast<int>(h * frame.rows / input_port.get_shape()[2]);
            detections.push_back({cv::Rect(left, top, width, height), classId, maxScore * conf});
        }
    }
    const std::vector<Detection>& getDetections() const { return detections; }
private:
    std::shared_ptr<ov::Core> core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Output<const ov::Node> input_port;
    ov::Output<const ov::Node> output_port;
};

class ZeroCopyOpenVINOYOLO {
    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<libcamera::CameraManager> cameraManager;
    std::unique_ptr<libcamera::CameraConfiguration> config;
    libcamera::Stream* stream;
    std::shared_ptr<libcamera::FrameBufferAllocator> allocator;
    std::vector<std::vector<void*>> bufferPlaneMappings;
    std::vector<std::vector<size_t>> bufferPlaneSizes;
    std::atomic<bool> stopping{false};
    std::unique_ptr<OpenVINOYOLO> yolo;
    // SORT 트래커 추가
    Sort sort_tracker;
    // FPS 측정
    std::chrono::steady_clock::time_point lastTime;
    int frameCounter = 0;
    double fps = 0.0;
public:
    ZeroCopyOpenVINOYOLO(const std::string& model_xml)
        : sort_tracker(5, 2, 0.3f) {
        yolo = std::make_unique<OpenVINOYOLO>(model_xml);
        lastTime = std::chrono::steady_clock::now();
    }
    bool initialize() {
        std::cout << "카메라 초기화 중..." << std::endl;
        cameraManager = std::make_unique<libcamera::CameraManager>();
        if (cameraManager->start()) {
            std::cout << "카메라 매니저 시작 실패" << std::endl;
            return false;
        }
        auto cameras = cameraManager->cameras();
        if (cameras.empty()) {
            std::cout << "사용 가능한 카메라가 없습니다" << std::endl;
            return false;
        }
        camera = cameras[0];
        if (camera->acquire()) {
            std::cout << "카메라 획득 실패" << std::endl;
            return false;
        }
        config = camera->generateConfiguration({libcamera::StreamRole::Viewfinder});
        auto& streamConfig = config->at(0);
        streamConfig.size = libcamera::Size(1920, 1080);
        streamConfig.pixelFormat = libcamera::formats::RGB888;
        streamConfig.bufferCount = 4;
        config->validate();
        if (camera->configure(config.get())) {
            std::cout << "카메라 설정 실패" << std::endl;
            return false;
        }
        stream = streamConfig.stream();
        std::cout << "카메라 설정 완료: " << streamConfig.size.width << "x" << streamConfig.size.height << std::endl;
        return setupBuffers();
    }
    bool setupBuffers() {
        std::cout << "버퍼 설정 중..." << std::endl;
        allocator = std::make_shared<libcamera::FrameBufferAllocator>(camera);
        if (allocator->allocate(stream) < 0) {
            std::cout << "버퍼 할당 실패" << std::endl;
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
                    std::cout << "메모리 맵핑 실패" << std::endl;
                    return false;
                }
                bufferPlaneMappings[i].push_back(memory);
                bufferPlaneSizes[i].push_back(plane.length);
            }
        }
        std::cout << "버퍼 설정 완료: " << buffers.size() << "개 버퍼" << std::endl;
        return true;
    }
    bool start() {
        std::cout << "카메라 시작 중..." << std::endl;
        camera->requestCompleted.connect(this, &ZeroCopyOpenVINOYOLO::onRequestCompleted);
        if (camera->start()) {
            std::cout << "카메라 시작 실패" << std::endl;
            return false;
        }
        for (const auto& buffer : allocator->buffers(stream)) {
            std::unique_ptr<libcamera::Request> request = camera->createRequest();
            if (!request || request->addBuffer(stream, buffer.get())) {
                std::cout << "요청 생성 실패" << std::endl;
                return false;
            }
            camera->queueRequest(request.release());
        }
        std::cout << "캡처 및 OpenVINO 추론 시작..." << std::endl;
        return true;
    }
    void stop() {
        std::cout << "중지 신호 받음..." << std::endl;
        stopping.store(true);
        if (camera) {
            camera->stop();
            camera->requestCompleted.disconnect(this, &ZeroCopyOpenVINOYOLO::onRequestCompleted);
        }
    }
    void onRequestCompleted(libcamera::Request* request) {
        if (stopping.load() || request->status() != libcamera::Request::RequestComplete) {
            request->reuse(libcamera::Request::ReuseFlag::ReuseBuffers);
            camera->queueRequest(request);
            return;
        }
        auto* buffer = request->buffers().begin()->second;
        const auto& buffers = allocator->buffers(stream);
        size_t bufferIndex = std::distance(buffers.begin(),
            std::find_if(buffers.begin(), buffers.end(),
                         [buffer](const auto& b){ return b.get() == buffer; }));
        void* data = bufferPlaneMappings[bufferIndex][0];
        const auto& streamConfig = config->at(0);
        cv::Mat frame(streamConfig.size.height, streamConfig.size.width, CV_8UC3, data, streamConfig.stride);
        yolo->infer(frame);
        // --- 트래킹 적용 ---
        const auto& dets = yolo->getDetections();
        std::vector<std::vector<float>> dets_for_sort;
        for (const auto& d : dets) {
            dets_for_sort.push_back({(float)d.bbox.x, (float)d.bbox.y, (float)(d.bbox.x+d.bbox.width), (float)(d.bbox.y+d.bbox.height), d.confidence, (float)d.class_id});
        }
        auto tracked = sort_tracker.update(dets_for_sort);
        // 트래킹 결과 로그 출력
        std::cout << "[트래킹 결과] 트랙 개수: " << tracked.size() << std::endl;
        for (const auto& t : tracked) {
            int id = (int)t[6];
            int class_id = (int)t[5];
            float conf = t[4];
            int x1 = (int)t[0], y1 = (int)t[1], x2 = (int)t[2], y2 = (int)t[3];
            std::cout << "  - ID: " << id << ", class: " << class_id << ", conf: " << conf
                      << ", bbox: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]" << std::endl;
        }
        // 시각화
        std::vector<Track> tracks;
        for (const auto& t : tracked) {
            int id = (int)t[6];
            int class_id = (int)t[5];
            float conf = t[4];
            cv::Rect bbox((int)t[0], (int)t[1], (int)(t[2]-t[0]), (int)(t[3]-t[1]));
            tracks.push_back({id, bbox, class_id, conf});
        }
        ObjectTracker::drawTracks(frame, tracks, true);
        // ...FPS, 성능 출력 등 기존 코드 유지...
        frameCounter++;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastTime).count() >= 1) {
            fps = frameCounter / std::chrono::duration_cast<std::chrono::duration<double>>(now - lastTime).count();
            frameCounter = 0;
            lastTime = now;
            std::cout << "[FPS: " << std::fixed << std::setprecision(1) << fps << "] 프레임 처리 중..." << std::endl;
        }
        request->reuse(libcamera::Request::ReuseFlag::ReuseBuffers);
        if (!stopping.load()) camera->queueRequest(request);
    }
};

static std::atomic<bool> shouldExit{false};
static ZeroCopyOpenVINOYOLO* demoInstance = nullptr;
void signalHandler(int signal) {
    std::cout << "\n종료 신호 받음 (Ctrl+C)" << std::endl;
    shouldExit.store(true);
    if (demoInstance) demoInstance->stop();
}
int main(int argc, char** argv) {
    std::cout << "=== Zero Copy OpenVINO YOLO+SORT Demo (Headless) ===" << std::endl;
    std::string model_xml = "/home/lee/Documents/server-raspicam/io/demo/cpp/libcamera_zerocopy/openvino/yolo5n_openvino_model/yolov5n.xml";
    std::cout << "모델 파일: " << model_xml << std::endl;
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    ZeroCopyOpenVINOYOLO demo(model_xml);
    demoInstance = &demo;
    if (!demo.initialize()) {
        std::cout << "초기화 실패" << std::endl;
        return -1;
    }
    if (!demo.start()) {
        std::cout << "시작 실패" << std::endl;
        return -1;
    }
    std::cout << "실행 중... (Ctrl+C로 종료)" << std::endl;
    while (!shouldExit.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "프로그램 종료" << std::endl;
    return 0;
}

#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> // OpenVINO 헤더 (경로는 환경에 맞게 수정)

#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/stream.h>

// --- OpenVINO YOLO 추론 클래스 (mainv11.cpp에서 복사/수정) ---
class OpenVINOYOLO {
public:
    OpenVINOYOLO(const std::string& model_xml, const std::string& device = "CPU") {
        core = std::make_shared<ov::Core>();
        model = core->read_model(model_xml);
        compiled_model = core->compile_model(model, device);
        infer_request = compiled_model.create_infer_request();
        input_port = compiled_model.input();
        output_port = compiled_model.output();
    }

    // YOLO 추론 (cv::Mat 입력)
    void infer(const cv::Mat& frame) {
        // 전처리: 모델에 맞게 수정
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(input_port.get_shape()[3], input_port.get_shape()[2]));
        resized.convertTo(resized, CV_32F, 1.0 / 255);

        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), resized.data);
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        // 후처리: mainv11.cpp의 후처리 코드 참고
        // auto output = infer_request.get_output_tensor();
        // ... (YOLO 결과 해석 및 시각화)
    }

private:
    std::shared_ptr<ov::Core> core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Output<const ov::Node> input_port;
    ov::Output<const ov::Node> output_port;
};
// -------------------------------------------------------------

class ZeroCopyOpenVINOYOLO {
    // libcamera 및 버퍼 관련 멤버
    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<libcamera::CameraManager> cameraManager;
    std::unique_ptr<libcamera::CameraConfiguration> config;
    libcamera::Stream* stream;
    std::shared_ptr<libcamera::FrameBufferAllocator> allocator;
    std::vector<std::vector<void*>> bufferPlaneMappings;
    std::vector<std::vector<size_t>> bufferPlaneSizes;
    std::atomic<bool> stopping{false};

    std::unique_ptr<OpenVINOYOLO> yolo;

public:
    ZeroCopyOpenVINOYOLO(const std::string& model_xml) {
        yolo = std::make_unique<OpenVINOYOLO>(model_xml);
    }

    bool initialize() {
        cameraManager = std::make_unique<libcamera::CameraManager>();
        if (cameraManager->start()) return false;
        auto cameras = cameraManager->cameras();
        if (cameras.empty()) return false;
        camera = cameras[0];
        if (camera->acquire()) return false;
        config = camera->generateConfiguration({libcamera::StreamRole::Viewfinder});
        auto& streamConfig = config->at(0);
        streamConfig.size = libcamera::Size(1920, 1080);
        streamConfig.pixelFormat = libcamera::formats::BGR888;
        streamConfig.bufferCount = 4;
        config->validate();
        if (camera->configure(config.get())) return false;
        stream = streamConfig.stream();
        return setupBuffers();
    }

    bool setupBuffers() {
        allocator = std::make_shared<libcamera::FrameBufferAllocator>(camera);
        if (allocator->allocate(stream) < 0) return false;
        const auto& buffers = allocator->buffers(stream);
        bufferPlaneMappings.resize(buffers.size());
        bufferPlaneSizes.resize(buffers.size());
        for (size_t i = 0; i < buffers.size(); ++i) {
            for (size_t j = 0; j < buffers[i]->planes().size(); ++j) {
                const auto& plane = buffers[i]->planes()[j];
                void* memory = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
                if (memory == MAP_FAILED) return false;
                bufferPlaneMappings[i].push_back(memory);
                bufferPlaneSizes[i].push_back(plane.length);
            }
        }
        return true;
    }

    bool start() {
        camera->requestCompleted.connect(this, &ZeroCopyOpenVINOYOLO::onRequestCompleted);
        if (camera->start()) return false;
        for (const auto& buffer : allocator->buffers(stream)) {
            std::unique_ptr<libcamera::Request> request = camera->createRequest();
            if (!request || request->addBuffer(stream, buffer.get())) return false;
            camera->queueRequest(request.release());
        }
        std::cout << "캡처 및 OpenVINO 추론 시작..." << std::endl;
        return true;
    }

    void stop() {
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

        // OpenVINO YOLO 추론
        yolo->infer(frame);

        request->reuse(libcamera::Request::ReuseFlag::ReuseBuffers);
        if (!stopping.load()) camera->queueRequest(request);
    }
};

static std::atomic<bool> shouldExit{false};
static ZeroCopyOpenVINOYOLO* demoInstance = nullptr;

void signalHandler(int signal) {
    shouldExit.store(true);
    if (demoInstance) demoInstance->stop();
}

int main(int argc, char** argv) {
    std::string model_xml = "/home/lee/Documents/server-raspicam/openvino-test/yolo11n_openvino_model/yolo11n.xml";
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    ZeroCopyOpenVINOYOLO demo(model_xml);
    demoInstance = &demo;

    if (!demo.initialize()) return -1;
    if (!demo.start()) return -1;

    while (!shouldExit.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "프로그램 종료" << std::endl;
    return 0;
}

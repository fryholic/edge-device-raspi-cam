#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> // OpenVINO 헤더 (경로는 환경에 맞게 수정)
#include <sys/mman.h>

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
        output_port = compiled_model.outputs()[0]; // 첫 번째 출력 사용
    }

    // YOLO 추론 (cv::Mat 입력)
    struct Detection {
        cv::Rect bbox;
        int class_id;
        float confidence;
    };
    std::vector<Detection> detections;

    void infer(const cv::Mat& frame) {
        // 전처리: 모델에 맞게 수정 (BGR → RGB)
        cv::Mat resized, rgb;
        cv::resize(frame, resized, cv::Size(input_port.get_shape()[3], input_port.get_shape()[2]));
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255);

        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), rgb.data);
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        // YOLOv5 후처리 (가장 일반적인 [N, 85] 구조, N: 감지 개수, 0~3: x,y,w,h, 4: conf, 5~: class score)
        detections.clear();
        auto output = infer_request.get_output_tensor(0); // 첫 번째 출력 사용
        const float* data = output.data<float>();
        const ov::Shape& shape = output.get_shape();
        std::cout << "output shape: ";
        for (auto s : shape) std::cout << s << " ";
        std::cout << std::endl;
        size_t numBatch = shape[0]; // 1
        size_t numDet = shape[1]; // 6300
        size_t numElem = shape[2]; // 85
        int detected = 0;
        for (size_t i = 0; i < numDet; ++i) {
            const float* det = data + i * numElem; // batch=0만 사용
            float conf = det[4];
            if (conf < 0.01) continue; // threshold를 0.01로 낮춤
            float maxScore = 0;
            int classId = -1;
            for (int c = 5; c < numElem; ++c) {
                if (det[c] > maxScore) {
                    maxScore = det[c];
                    classId = c - 5;
                }
            }
            if (maxScore * conf < 0.01) continue; // threshold를 0.01로 낮춤
            float cx = det[0], cy = det[1], w = det[2], h = det[3];
            // input shape: [N, 3, H, W], 여기서 H=input_port.get_shape()[2], W=input_port.get_shape()[3]
            int left = static_cast<int>((cx - w/2) * frame.cols / input_port.get_shape()[3]);
            int top = static_cast<int>((cy - h/2) * frame.rows / input_port.get_shape()[2]);
            int width = static_cast<int>(w * frame.cols / input_port.get_shape()[3]);
            int height = static_cast<int>(h * frame.rows / input_port.get_shape()[2]);
            detections.push_back({cv::Rect(left, top, width, height), classId, maxScore * conf});
            std::cout << "det: class=" << classId << ", conf=" << maxScore * conf << ", box=" << left << "," << top << "," << width << "," << height << std::endl;
            detected++;
        }
        std::cout << "Detected boxes: " << detected << std::endl;
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

        // FPS 계산
        static auto lastTime = std::chrono::steady_clock::now();
        static int frameCounter = 0;
        static double fps = 0.0;
        frameCounter++;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastTime).count() >= 1) {
            fps = frameCounter / std::chrono::duration_cast<std::chrono::duration<double>>(now - lastTime).count();
            frameCounter = 0;
            lastTime = now;
        }
        cv::putText(frame, cv::format("FPS: %.1f", fps), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 2);

        // Bounding Box (사람만, class_id==0)
        for (const auto& det : yolo->getDetections()) {
            if (det.class_id == 0 && det.confidence > 0.01) { // 0.5 → 0.01로 변경
                cv::rectangle(frame, det.bbox, cv::Scalar(0,0,255), 2);
                cv::putText(frame, "person", det.bbox.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
            }
        }

        // 결과 이미지 표시 (BGR888 → RGB 변환 후 출력)
        cv::Mat rgbFrame;
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
        cv::imshow("Camera", rgbFrame);
        cv::waitKey(1);

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
    std::string model_xml = "/home/park/vino_ws/yolo5n_openvino_model/yolov5n.xml";
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    ZeroCopyOpenVINOYOLO demo(model_xml);
    demoInstance = &demo;

    if (!demo.initialize()) return -1;
    if (!demo.start()) return -1;

    while (!shouldExit.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "프로그램 종료" << std::endl;
    cv::destroyAllWindows();
    return 0;
}

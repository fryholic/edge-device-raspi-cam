#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> // OpenVINO 헤더 (경로는 환경에 맞게 수정)
#include <sys/mman.h>
#include <algorithm>

// sigmoid_clip 함수 추가 (더 보수적 클리핑 적용)
float sigmoid_clip(float x, float min_val = 1e-7f, float max_val = 0.999f) {
    // 매우 큰 값은 미리 클리핑 (raw 값이 10 이상이면 의심스러움)
    if (x > 10.0f) x = 10.0f;
    if (x < -10.0f) x = -10.0f;
    
    float sigmoid_val = 1.0f / (1.0f + expf(-x));
    return std::max(min_val, std::min(max_val, sigmoid_val));
}

#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/stream.h>

// Detection 구조체를 전역으로 이동
struct Detection {
    cv::Rect bbox;
    int class_id;
    float confidence;
};

// NMS 함수 추가 (YOLOv5 코드에서 가져옴)
float iou(const cv::Rect& a, const cv::Rect& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return inter / uni;
}

std::vector<Detection> nms(const std::vector<Detection>& dets, float iou_threshold = 0.45f) {
    std::vector<Detection> res;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        res.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[i].class_id == dets[j].class_id &&
                iou(dets[i].bbox, dets[j].bbox) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return res;
}

// --- OpenVINO YOLO 추론 클래스 (mainv11.cpp에서 복사/수정) ---
class OpenVINOYOLO {
public:
    OpenVINOYOLO(const std::string& model_xml, const std::string& device = "CPU") {
        core = std::make_shared<ov::Core>();
        model = core->read_model(model_xml);
        ov::AnyMap compile_options;
        compile_options["INFERENCE_PRECISION_HINT"] = "f16";
        compiled_model = core->compile_model(model, device, compile_options);
        infer_request = compiled_model.create_infer_request();
        input_port = compiled_model.input();
        output_port = compiled_model.outputs()[0]; // 첫 번째 출력 사용
        
        std::cout << "OpenVINO YOLO 모델 초기화 완료" << std::endl;
        std::cout << "입력 크기: " << input_port.get_shape()[3] << "x" << input_port.get_shape()[2] << std::endl;
        // === FP16 최적화 디버그 출력 ===
        std::cout << "[디버그] 입력 tensor 타입: " << input_port.get_element_type() << std::endl;
        std::cout << "[디버그] 출력 tensor 타입: " << output_port.get_element_type() << std::endl;
        try {
            std::cout << "[디버그] 사용 디바이스: " << compiled_model.get_property(ov::device::full_name) << std::endl;
        } catch (...) {
            std::cout << "[디버그] 사용 디바이스 정보 조회 실패" << std::endl;
        }
        std::cout << "[디버그] 레이어별 출력 타입 정보:" << std::endl;
        for (const auto& op : model->get_ops()) {
            std::cout << "  레이어: " << op->get_friendly_name()
                      << ", 타입: " << op->get_type_name()
                      << ", 출력 타입: " << op->get_output_element_type(0) << std::endl;
        }
    }

    // YOLO 추론 (cv::Mat 입력)
    std::vector<Detection> detections;

    void infer(const cv::Mat& frame) {
        static bool first_infer = true;
        auto start_time = std::chrono::steady_clock::now();
        
        // 입력 크기 자동 추출
        int input_w = input_port.get_shape()[3];
        int input_h = input_port.get_shape()[2];
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(input_w, input_h));
        resized.convertTo(resized, CV_32F, 1.0 / 255);

        auto preprocess_time = std::chrono::steady_clock::now();

        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), resized.data);
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        auto inference_time = std::chrono::steady_clock::now();

        auto output = infer_request.get_output_tensor(0); // 첫 번째 출력 사용
        const float* data = output.data<float>();
        const ov::Shape& shape = output.get_shape();
        // YOLOv5 후처리 (output shape: [1, 84, 2100], 0~3: bbox, 4: objectness, 5~: class score)
        detections.clear();
        // output shape: [1, 84, 2100]
        size_t numBatch = shape[0]; // 1
        size_t numFeature = shape[1]; // 84
        size_t numDet = shape[2]; // 2100
        int detected = 0;
        
        // 최초 1회 bbox raw 값 출력 + 디버그: 최초 10개 detection raw 값 출력
        static bool first_bbox_output = true;
        if (first_bbox_output) {
            std::cout << "[bbox raw 값 예시 (첫 3개 detection)]:" << std::endl;
            for (int i = 0; i < 3; ++i) {
                float cx = data[numFeature * i + 0];
                float cy = data[numFeature * i + 1];
                float w  = data[numFeature * i + 2];
                float h  = data[numFeature * i + 3];
                float obj_conf_raw = data[numFeature * i + 4];
                float cls_score_raw = data[numFeature * i + 5];
                std::cout << "  Detection " << i << ": cx=" << cx << ", cy=" << cy 
                         << ", w=" << w << ", h=" << h 
                         << ", obj_conf_raw=" << obj_conf_raw 
                         << ", cls_score_raw=" << cls_score_raw << std::endl;
            }
            
            // 객체 신뢰도 분포 분석
            std::cout << "[디버그] 객체 신뢰도 분포 분석:" << std::endl;
            std::vector<float> conf_values;
            std::vector<float> sigmoid_conf_values;
            for (size_t i = 0; i < std::min((size_t)100, numDet); ++i) {
                float raw_conf = data[numFeature * i + 4];
                float sigmoid_conf = sigmoid_clip(raw_conf);
                conf_values.push_back(raw_conf);
                sigmoid_conf_values.push_back(sigmoid_conf);
            }
            std::sort(conf_values.begin(), conf_values.end(), std::greater<float>());
            std::sort(sigmoid_conf_values.begin(), sigmoid_conf_values.end(), std::greater<float>());
            
            std::cout << "  상위 10개 raw 신뢰도: ";
            for (int i = 0; i < std::min(10, (int)conf_values.size()); ++i) {
                std::cout << std::fixed << std::setprecision(3) << conf_values[i] << " ";
            }
            std::cout << std::endl;
            
            std::cout << "  상위 10개 sigmoid 신뢰도: ";
            for (int i = 0; i < std::min(10, (int)sigmoid_conf_values.size()); ++i) {
                std::cout << std::fixed << std::setprecision(3) << sigmoid_conf_values[i] << " ";
            }
            std::cout << std::endl;
            
            first_bbox_output = false;
        }
        
        for (size_t i = 0; i < numDet; ++i) {
            float cx = data[numFeature * i + 0];
            float cy = data[numFeature * i + 1];
            float w  = data[numFeature * i + 2];
            float h  = data[numFeature * i + 3];
            
            // Raw 값 체크 - 극단적인 값만 필터링 (조건 완화)
            float obj_conf_raw = data[numFeature * i + 4];
            if (obj_conf_raw > 100.0f) continue; // 100 이상만 필터링 (50에서 완화)
            
            float obj_conf = sigmoid_clip(obj_conf_raw);
            if (obj_conf < 0.3f) continue; // threshold를 0.3으로 완화 (0.5에서 하향)
            
            // class score
            float max_cls = 0.0f;
            int class_id = -1;
            for (int c = 5; c < (int)numFeature; ++c) {
                float cls_score_raw = data[numFeature * i + c];
                if (cls_score_raw > 100.0f) continue; // raw 값 체크도 완화
                
                float cls_score = sigmoid_clip(cls_score_raw);
                if (cls_score > max_cls) {
                    max_cls = cls_score;
                    class_id = c - 5;
                }
            }
            
            // max_cls 검증 조건 완화
            if (max_cls > 0.999f) continue; // 0.95에서 0.999로 완화
            
            float conf = obj_conf * max_cls;
            if (conf < 0.3f) continue; // confidence threshold를 0.3으로 완화
            
            // 특정 클래스만 허용 (person = class_id 0만 허용)
            if (class_id != 0) continue; // person 클래스가 아니면 스킵
            
            // bbox 변환 (0~input_w 입력 해상도 → frame 해상도)
            int left = static_cast<int>((cx - w/2) * frame.cols / input_w);
            int top = static_cast<int>((cy - h/2) * frame.rows / input_h);
            int width = static_cast<int>(w * frame.cols / input_w);
            int height = static_cast<int>(h * frame.rows / input_h);
            
            // 유효한 바운딩 박스인지 확인
            if (width > 0 && height > 0 && left >= 0 && top >= 0 && 
                left + width <= frame.cols && top + height <= frame.rows) {
                detections.push_back({cv::Rect(left, top, width, height), class_id, conf});
                detected++;
            }
        }
        
        // NMS 적용 (더 엄격한 IoU threshold)
        auto nms_results = nms(detections, 0.4f); // 0.3에서 0.4로 조정
        detections = nms_results;
        detected = detections.size();

        auto end_time = std::chrono::steady_clock::now();

        // 시간 측정 결과
        auto preprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(preprocess_time - start_time).count();
        auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_time - preprocess_time).count();
        auto postprocess_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - inference_time).count();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // 매 프레임마다가 아닌 감지된 객체가 있을 때만 상세 정보 출력
        if (detected > 0) {
            std::cout << "=== 객체 감지됨 ===" << std::endl;
            std::cout << "감지된 객체 수: " << detected << std::endl;
            std::cout << "추론 시간 - 전처리: " << preprocess_ms << "ms, 추론: " << inference_ms << "ms, 후처리: " << postprocess_ms << "ms, 총: " << total_ms << "ms" << std::endl;
            
            // 클래스 이름 (예시: COCO 80 클래스, 실제 모델에 맞게 수정)
            const char* class_names[] = {
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
            };
            for (const auto& det : detections) {
                const char* class_name = (det.class_id < 80) ? class_names[det.class_id] : "unknown";
                std::cout << "  - " << class_name << " (신뢰도: " << std::fixed << std::setprecision(2) << det.confidence 
                         << ", 위치: " << det.bbox.x << "," << det.bbox.y << "," << det.bbox.width << "," << det.bbox.height << ")" << std::endl;
            }
            std::cout << "===================" << std::endl;
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

    // FPS 측정을 위한 변수들
    std::chrono::steady_clock::time_point lastTime;
    int frameCounter = 0;
    double fps = 0.0;

public:
    ZeroCopyOpenVINOYOLO(const std::string& model_xml) {
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
        // RGB888로 설정 시도
        streamConfig.pixelFormat = libcamera::formats::RGB888;
        streamConfig.bufferCount = 4;
        config->validate();
        if (camera->configure(config.get())) {
            std::cout << "카메라 설정 실패 (RGB888 미지원일 수 있음)" << std::endl;
            return false;
        }
        std::cout << "카메라 설정 완료: " << streamConfig.size.width << "x" << streamConfig.size.height
                  << ", 포맷: " << streamConfig.pixelFormat.toString() << std::endl;
        stream = streamConfig.stream();
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

        // 추론 시간 측정 시작
        auto inference_start = std::chrono::steady_clock::now();
        yolo->infer(frame);
        auto inference_end = std::chrono::steady_clock::now();
        
        // 추론 시간 계산
        auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start).count();

        // FPS 계산
        frameCounter++;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastTime).count() >= 1) {
            fps = frameCounter / std::chrono::duration_cast<std::chrono::duration<double>>(now - lastTime).count();
            frameCounter = 0;
            lastTime = now;
            
            // 상세한 성능 정보 출력
            std::cout << "[FPS: " << std::fixed << std::setprecision(1) << fps 
                      << ", 추론시간: " << inference_ms << "ms] 프레임 처리 중..." << std::endl;
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
    std::cout << "=== Zero Copy OpenVINO YOLO Demo (Headless) ===" << std::endl;
    
    std::string model_xml = "/home/lee/Documents/server-raspicam/io/demo/cpp/libcamera_zerocopy/openvino/yolo11n_openvino_model/yolo11n_FP16_true.xml";
    
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

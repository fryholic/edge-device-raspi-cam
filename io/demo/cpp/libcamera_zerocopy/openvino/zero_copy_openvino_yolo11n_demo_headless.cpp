#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> // OpenVINO 헤더 (경로는 환경에 맞게 수정)
#include <sys/mman.h>

#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/stream.h>

#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>

// --- OpenVINO YOLO Detector 클래스 ---
// 효율적인 전처리를 위해 OpenVINO Preprocessing API를 사용하고,
// 정확한 후처리를 통해 신뢰도 높은 객체 탐지를 수행합니다.
class OpenVINOYOLODetector {
public:
    // 탐지 결과 구조체
    struct Detection {
        cv::Rect bbox;
        int class_id;
        float confidence;
        std::string class_name; // 클래스 이름 추가
    };

    // 생성자: 모델 로드 및 전/후처리 설정
    OpenVINOYOLODetector(const std::string& model_xml, const std::string& labels_path = "", const std::string& device = "CPU") {
        if (!labels_path.empty()) {
            load_class_labels(labels_path);
        }

        core = std::make_shared<ov::Core>();
        auto model = core->read_model(model_xml);

        // --- OpenVINO Preprocessing 설정 ---
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input()
            .tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC") 
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        ppp.input().model().set_layout("NCHW");
        ppp.input().preprocess()
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
            .convert_element_type(ov::element::f32)
            .scale(255.0f);
        
        model = ppp.build();
        // --- Preprocessing 설정 완료 ---

        compiled_model = core->compile_model(model, device);
        infer_request = compiled_model.create_infer_request();

        input_port = compiled_model.input();
        output_port = compiled_model.output();

        const ov::Shape& input_shape = input_port.get_shape();
        input_height = input_shape[2];
        input_width = input_shape[3];

        std::cout << "OpenVINO YOLO 모델 초기화 완료" << std::endl;
        std::cout << "모델 입력 크기 (HxW): " << input_height << "x" << input_width << std::endl;
        std::cout << "모델 출력 형태: " << output_port.get_shape() << std::endl;
    }

    // 추론 함수: 카메라 버퍼를 받아 객체 탐지 수행
    std::vector<Detection> infer(void* buffer, int frame_width, int frame_height, 
                                 float conf_threshold = 0.25f, float nms_threshold = 0.45f) {
        
        ov::Tensor input_tensor(ov::element::u8, {1, (size_t)frame_height, (size_t)frame_width, 3}, buffer);
        infer_request.set_input_tensor(input_tensor);
        
        auto start_infer = std::chrono::steady_clock::now();
        infer_request.infer();
        auto end_infer = std::chrono::steady_clock::now();
        inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer).count();

        const ov::Tensor& output = infer_request.get_output_tensor();
        const float* data = output.data<float>();
        
        const ov::Shape& shape = output.get_shape();
        bool is_transposed = (shape.size() == 3 && shape[1] == 84 && shape[2] == 8400);
        
        if (!is_transposed && !(shape.size() == 3 && shape[1] == 8400 && shape[2] == 84)) {
            throw std::runtime_error("오류: 지원되지 않는 모델 출력 형태입니다. [1, 84, 8400] 또는 [1, 8400, 84]가 필요합니다.");
        }

        size_t num_detections = is_transposed ? shape[2] : shape[1];
        int num_classes = (is_transposed ? shape[1] : shape[2]) - 4;

        std::vector<cv::Rect> bboxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        float x_factor = static_cast<float>(frame_width) / input_width;
        float y_factor = static_cast<float>(frame_height) / input_height;

        if (is_transposed) {
             // Transposed output: [1, 84, 8400]
            std::vector<float> class_scores(num_classes);
            for (size_t i = 0; i < num_detections; ++i) {
                for (int j = 0; j < num_classes; ++j) {
                    class_scores[j] = data[(4 + j) * num_detections + i];
                }
                
                softmax(class_scores);

                float max_score = 0.0f;
                int class_id = -1;
                for(int j=0; j<num_classes; ++j) {
                    if(class_scores[j] > max_score) {
                        max_score = class_scores[j];
                        class_id = j;
                    }
                }

                if (max_score > conf_threshold) {
                    float cx = data[0 * num_detections + i];
                    float cy = data[1 * num_detections + i];
                    float w  = data[2 * num_detections + i];
                    float h  = data[3 * num_detections + i];
                    bboxes.push_back(build_bbox(cx, cy, w, h, x_factor, y_factor));
                    confidences.push_back(max_score);
                    class_ids.push_back(class_id);
                }
            }
        } else { // Standard output: [1, 8400, 84]
            for (size_t i = 0; i < num_detections; ++i) {
                const float* det_data = data + i * (num_classes + 4);
                
                std::vector<float> class_scores(det_data + 4, det_data + 4 + num_classes);
                softmax(class_scores);

                float max_score = 0.0f;
                int class_id = -1;
                for(int j=0; j<num_classes; ++j) {
                    if(class_scores[j] > max_score) {
                        max_score = class_scores[j];
                        class_id = j;
                    }
                }

                if (max_score > conf_threshold) {
                    float cx = det_data[0], cy = det_data[1], w = det_data[2], h = det_data[3];
                    bboxes.push_back(build_bbox(cx, cy, w, h, x_factor, y_factor));
                    confidences.push_back(max_score);
                    class_ids.push_back(class_id);
                }
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, confidences, conf_threshold, nms_threshold, indices);

        std::vector<Detection> detections;
        for (int idx : indices) {
            std::string class_name = (class_ids[idx] < class_labels.size()) ? class_labels[class_ids[idx]] : "unknown";
            detections.push_back({bboxes[idx], class_ids[idx], confidences[idx], class_name});
        }
        return detections;
    }
    
    long get_inference_time_ms() const { return inference_time_ms; }

private:
    std::shared_ptr<ov::Core> core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Output<const ov::Node> input_port;
    ov::Output<const ov::Node> output_port;
    
    long input_height;
    long input_width;
    long inference_time_ms = 0;
    std::vector<std::string> class_labels;

    void load_class_labels(const std::string& path) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                class_labels.push_back(line);
            }
            file.close();
            std::cout << class_labels.size() << "개의 클래스 레이블 로드 완료." << std::endl;
        } else {
            std::cerr << "경고: 클래스 레이블 파일을 열 수 없습니다: " << path << std::endl;
        }
    }

    void softmax(std::vector<float>& scores) {
        if (scores.empty()) return;
        float max_score = scores[0];
        for (size_t i = 1; i < scores.size(); ++i) {
            if (scores[i] > max_score) {
                max_score = scores[i];
            }
        }
        float sum = 0.0f;
        for (size_t i = 0; i < scores.size(); ++i) {
            scores[i] = std::exp(scores[i] - max_score);
            sum += scores[i];
        }
        for (size_t i = 0; i < scores.size(); ++i) {
            scores[i] /= sum;
        }
    }

    cv::Rect build_bbox(float cx, float cy, float w, float h, float x_factor, float y_factor) {
        int left = static_cast<int>((cx - w / 2) * x_factor);
        int top = static_cast<int>((cy - h / 2) * y_factor);
        int box_width = static_cast<int>(w * x_factor);
        int box_height = static_cast<int>(h * y_factor);
        return cv::Rect(left, top, box_width, box_height);
    }
};
// -------------------------------------------------------------

class ZeroCopyOpenVINOYOLO11n {
    // libcamera 및 버퍼 관련 멤버
    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<libcamera::CameraManager> cameraManager;
    std::unique_ptr<libcamera::CameraConfiguration> config;
    libcamera::Stream* stream;
    std::shared_ptr<libcamera::FrameBufferAllocator> allocator;
    std::vector<std::vector<void*>> bufferPlaneMappings;
    std::vector<std::vector<size_t>> bufferPlaneSizes;
    std::atomic<bool> stopping{false};

    std::unique_ptr<OpenVINOYOLODetector> yolo;

    // FPS 측정을 위한 변수들
    std::chrono::steady_clock::time_point last_fps_time;
    int frame_counter = 0;
    double current_fps = 0.0;

    void requestComplete(libcamera::Request *request);

public:
    ZeroCopyOpenVINOYOLO11n(const std::string& model_xml, const std::string& labels_path) {
        yolo = std::make_unique<OpenVINOYOLODetector>(model_xml, labels_path);
        last_fps_time = std::chrono::steady_clock::now();
    }

    bool initialize() {
        std::cout << "카메라 초기화 중..." << std::endl;
        cameraManager = std::make_unique<libcamera::CameraManager>();
        if (cameraManager->start()) {
            std::cerr << "카메라 매니저 시작 실패" << std::endl;
            return false;
        }
        auto cameras = cameraManager->cameras();
        if (cameras.empty()) {
            std::cerr << "사용 가능한 카메라가 없습니다" << std::endl;
            return false;
        }
        camera = cameras[0];
        if (camera->acquire()) {
            std::cerr << "카메라 획득 실패" << std::endl;
            return false;
        }
        config = camera->generateConfiguration({libcamera::StreamRole::Viewfinder});
        auto& streamConfig = config->at(0);
        streamConfig.size = libcamera::Size(1920, 1080);
        streamConfig.pixelFormat = libcamera::formats::BGR888;
        streamConfig.bufferCount = 4; // 버퍼 수
        config->validate();
        if (camera->configure(config.get())) {
            std::cerr << "카메라 설정 실패" << std::endl;
            return false;
        }
        stream = streamConfig.stream();
        std::cout << "카메라 설정 완료: " << streamConfig.size.width << "x" << streamConfig.size.height << std::endl;

        allocator = std::make_shared<libcamera::FrameBufferAllocator>(camera);
        if (allocator->allocate(stream) < 0) {
            std::cerr << "버퍼 할당 실패" << std::endl;
            return false;
        }
        
        const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& buffers = allocator->buffers(stream);
        bufferPlaneMappings.resize(buffers.size());
        bufferPlaneSizes.resize(buffers.size());

        for (size_t i = 0; i < buffers.size(); ++i) {
            const auto& planes = buffers[i]->planes();
            bufferPlaneMappings[i].resize(planes.size());
            bufferPlaneSizes[i].resize(planes.size());
            for (size_t j = 0; j < planes.size(); ++j) {
                const libcamera::FrameBuffer::Plane& plane = planes[j];
                bufferPlaneMappings[i][j] = mmap(NULL, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
                if (bufferPlaneMappings[i][j] == MAP_FAILED) {
                    std::cerr << "mmap 실패" << std::endl;
                    return false;
                }
                bufferPlaneSizes[i][j] = plane.length;
            }
        }
        std::cout << "버퍼 설정 완료: " << buffers.size() << "개 버퍼" << std::endl;
        return true;
    }

    void run() {
        camera->requestCompleted.connect(this, &ZeroCopyOpenVINOYOLO11n::requestComplete);

        if (camera->start()) {
            std::cerr << "카메라 시작 실패" << std::endl;
            return;
        }

        const auto& buffers = allocator->buffers(stream);
        for (const auto& buffer : buffers) {
            auto request = camera->createRequest();
            if (!request) {
                std::cerr << "요청 생성 실패" << std::endl;
                camera->stop();
                return;
            }
            request->addBuffer(stream, buffer.get());
            camera->queueRequest(request.release());
        }

        std::cout << "캡처 및 OpenVINO 추론 시작..." << std::endl;
        std::cout << "실행 중... (Ctrl+C로 종료)" << std::endl;
        
        while (!stopping) {
            if (camera->processRequests() < 0) {
                std::cerr << "카메라 요청 처리 오류" << std::endl;
                break;
            }
        }
    }

    void stop() {
        stopping = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        camera->stop();
        camera->release();
        cameraManager->stop();
        std::cout << "
카메라 정지됨" << std::endl;

        for (size_t i = 0; i < bufferPlaneMappings.size(); ++i) {
            for (size_t j = 0; j < bufferPlaneMappings[i].size(); ++j) {
                if (bufferPlaneMappings[i][j] != MAP_FAILED) {
                    munmap(bufferPlaneMappings[i][j], bufferPlaneSizes[i][j]);
                }
            }
        }
    }
};

void ZeroCopyOpenVINOYOLO11n::requestComplete(libcamera::Request *request) {
    if (request->status() == libcamera::Request::RequestCancelled) return;
    if (stopping) {
        request->reuse(libcamera::Request::ReuseBuffers);
        camera->queueRequest(request);
        return;
    }

    libcamera::FrameBuffer* buffer = request->findBuffer(stream);
    unsigned int buffer_idx = 0;
    const auto& buffers = allocator->buffers(stream);
    for(unsigned int i=0; i<buffers.size(); ++i) {
        if(buffers[i].get() == buffer) {
            buffer_idx = i;
            break;
        }
    }

    void* data = bufferPlaneMappings[buffer_idx][0];
    auto detections = yolo->infer(data, stream->configuration().size.width, stream->configuration().size.height);

    // 결과 출력
    if (!detections.empty()) {
        std::cout << "=== 객체 감지됨 === (" << detections.size() << "개)" << std::endl;
        for (const auto& d : detections) {
            std::cout << "  - " << d.class_name << " (신뢰도: " << std::fixed << std::setprecision(2) << d.confidence
                      << ", 위치: " << d.bbox.x << "," << d.bbox.y << "," << d.bbox.width << "," << d.bbox.height << ")" << std::endl;
        }
        std::cout << "===================" << std::endl;
    }

    // FPS 계산 및 출력 (1초마다)
    frame_counter++;
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_fps_time);
    if (duration.count() >= 1) {
        current_fps = static_cast<double>(frame_counter) / duration.count();
        std::cout << "[FPS: " << std::fixed << std::setprecision(1) << current_fps 
                  << ", 추론시간: " << yolo->get_inference_time_ms() << "ms] 프레임 처리 중..." << std::endl;
        frame_counter = 0;
        last_fps_time = now;
    }

    request->reuse(libcamera::Request::ReuseBuffers);
    camera->queueRequest(request);
}


// --- main 함수 ---
std::atomic<bool> quit(false);
void signal_handler(int signum) {
    if (signum == SIGINT) {
        quit = true;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "사용법: " << argv[0] << " <모델.xml 경로> [클래스 레이블 파일 경로]" << std::endl;
        return -1;
    }
    signal(SIGINT, signal_handler);

    try {
        std::string model_path = argv[1];
        std::string labels_path = (argc > 2) ? argv[2] : "";

        ZeroCopyOpenVINOYOLO11n demo(model_path, labels_path);
        if (!demo.initialize()) {
            return -1;
        }

        std::thread run_thread([&demo]() { demo.run(); });

        while (!quit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        demo.stop();
        run_thread.join();

    } catch (const std::exception& e) {
        std::cerr << "오류: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

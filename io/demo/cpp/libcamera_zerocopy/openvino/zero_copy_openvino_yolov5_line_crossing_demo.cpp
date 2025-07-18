#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <sys/mman.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/stream.h>

// 트래킹 관련 헤더 추가
#include "sort.hpp"
#include "object_tracker.hpp"

// YOLOv5 constants
const int input_width = 320;
const int input_height = 320;
const float conf_threshold = 0.3;
const float iou_threshold = 0.45;
const int target_class = 0; // class 0: person

// Line crossing 관련 구조체
struct Point {
    float x;
    float y;
    
    Point() : x(0), y(0) {}
    Point(float x_, float y_) : x(x_), y(y_) {}
};

struct LineCrossingZone {
    Point start;
    Point end;
    std::string name;
    
    LineCrossingZone(const Point& s, const Point& e, const std::string& n) 
        : start(s), end(e), name(n) {}
};

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

struct CrossingEvent {
    int track_id;
    std::string zone_name;
    std::chrono::steady_clock::time_point timestamp;
    Point crossing_point;
    
    CrossingEvent(int id, const std::string& zone, const Point& point)
        : track_id(id), zone_name(zone), crossing_point(point), 
          timestamp(std::chrono::steady_clock::now()) {}
};

// Line crossing 계산 함수들
class LineCrossingDetector {
private:
    std::vector<LineCrossingZone> zones;
    std::unordered_map<int, std::unordered_map<std::string, float>> prev_positions;
    std::vector<CrossingEvent> recent_crossings;
    
public:
    void addZone(const LineCrossingZone& zone) {
        zones.push_back(zone);
    }
    
    // 선의 일반식을 사용한 위치 계산 (Ax + By + C = 0)
    float getPositionRelativeToLine(const Point& pt, const Point& lineStart, const Point& lineEnd) {
        float A = lineEnd.y - lineStart.y;
        float B = lineStart.x - lineEnd.x;
        float C = (lineEnd.x * lineStart.y) - (lineStart.x * lineEnd.y);
        
        return A * pt.x + B * pt.y + C;
    }
    
    // bounding box 중심점 계산
    Point getBboxCenter(const cv::Rect& bbox) {
        return Point((bbox.x + bbox.x + bbox.width) / 2.0f,
                    (bbox.y + bbox.y + bbox.height) / 2.0f);
    }
    
    // 트래킹된 객체들에 대해 line crossing 검사
    std::vector<CrossingEvent> checkCrossings(const std::vector<Track>& tracks) {
        std::vector<CrossingEvent> new_crossings;
        
        for (const auto& track : tracks) {
            Point center = getBboxCenter(track.bbox);
            
            for (const auto& zone : zones) {
                float position = getPositionRelativeToLine(center, zone.start, zone.end);
                
                // 이전 위치와 비교하여 선 통과 여부 확인
                if (prev_positions[track.id].find(zone.name) != prev_positions[track.id].end()) {
                    float prev_pos = prev_positions[track.id][zone.name];
                    
                    // 부호가 바뀌면 선을 통과한 것
                    if (prev_pos * position < 0) {
                        CrossingEvent event(track.id, zone.name, center);
                        new_crossings.push_back(event);
                        recent_crossings.push_back(event);
                        
                        std::cout << "🚨 LINE CROSSING DETECTED! 🚨" << std::endl;
                        std::cout << "  - Object ID: " << track.id << std::endl;
                        std::cout << "  - Zone: " << zone.name << std::endl;
                        std::cout << "  - Position: (" << center.x << ", " << center.y << ")" << std::endl;
                        std::cout << "  - Direction: " << (prev_pos > 0 ? "상→하" : "하→상") << std::endl;
                    }
                }
                
                // 현재 위치 저장
                prev_positions[track.id][zone.name] = position;
            }
        }
        
        // 오래된 crossing 이벤트 정리 (10초 이상 된 것들)
        auto now = std::chrono::steady_clock::now();
        recent_crossings.erase(
            std::remove_if(recent_crossings.begin(), recent_crossings.end(),
                [now](const CrossingEvent& event) {
                    return std::chrono::duration_cast<std::chrono::seconds>(now - event.timestamp).count() > 10;
                }),
            recent_crossings.end()
        );
        
        return new_crossings;
    }
    
    const std::vector<CrossingEvent>& getRecentCrossings() const {
        return recent_crossings;
    }
    
    void clearOldTracks(const std::vector<Track>& active_tracks) {
        // 활성 트랙이 아닌 것들의 이전 위치 정보 삭제
        std::unordered_set<int> active_ids;
        for (const auto& track : active_tracks) {
            active_ids.insert(track.id);
        }
        
        for (auto it = prev_positions.begin(); it != prev_positions.end();) {
            if (active_ids.find(it->first) == active_ids.end()) {
                it = prev_positions.erase(it);
            } else {
                ++it;
            }
        }
    }
};

// IoU calculation
float iou(const cv::Rect& a, const cv::Rect& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return inter / uni;
}

// NMS implementation
std::vector<Detection> nms(const std::vector<Detection>& dets) {
    std::vector<Detection> res;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        res.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[i].class_id == dets[j].class_id &&
                iou(dets[i].box, dets[j].box) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    return res;
}

// Letterbox preprocessing
cv::Mat letterbox(const cv::Mat& src, cv::Mat& out, float& scale, int& pad_x, int& pad_y) {
    int w = src.cols, h = src.rows;
    scale = std::min((float)input_width / w, (float)input_height / h);
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);
    pad_x = (input_width - new_w) / 2;
    pad_y = (input_height - new_h) / 2;
    cv::resize(src, out, cv::Size(new_w, new_h));
    cv::copyMakeBorder(out, out, pad_y, input_height - new_h - pad_y,
                              pad_x, input_width - new_w - pad_x,
                              cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

// OpenVINO YOLOv5 inference class with tracking and line crossing
class OpenVINOYOLOv5TrackerWithLineCrossing {
public:
    OpenVINOYOLOv5TrackerWithLineCrossing(const std::string& model_xml, const std::string& device = "CPU") 
        : sort_tracker(5, 2, 0.3f) {
        core = std::make_shared<ov::Core>();
        model = core->read_model(model_xml);
        compiled_model = core->compile_model(model, device);
        infer_request = compiled_model.create_infer_request();
        
        // 기본 line crossing zone 설정 (화면 중앙 가로선)
        setupDefaultLineCrossingZones();
        
        std::cout << "OpenVINO YOLOv5 + SORT + Line Crossing 초기화 완료" << std::endl;
        std::cout << "입력 크기: " << input_width << "x" << input_height << std::endl;
    }

    std::vector<Detection> detections;
    std::vector<Track> tracks;
    LineCrossingDetector line_detector;

    void inferTrackAndCheckCrossings(const cv::Mat& frame) {
        auto start_time = std::chrono::steady_clock::now();
        
        // 1. YOLOv5 추론
        performInference(frame);
        
        auto inference_time = std::chrono::steady_clock::now();
        
        // 2. SORT 트래킹 수행
        performTracking();
        
        auto tracking_time = std::chrono::steady_clock::now();
        
        // 3. Line crossing 검사
        auto crossings = line_detector.checkCrossings(tracks);
        line_detector.clearOldTracks(tracks);
        
        auto crossing_time = std::chrono::steady_clock::now();

        // 시간 측정 결과
        auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_time - start_time).count();
        auto tracking_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_time - inference_time).count();
        auto crossing_ms = std::chrono::duration_cast<std::chrono::milliseconds>(crossing_time - tracking_time).count();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(crossing_time - start_time).count();

        // 결과 출력 (트래킹 또는 crossing이 있을 때만)
        if (!tracks.empty() || !crossings.empty()) {
            std::cout << "=== YOLOv5 + SORT + Line Crossing 결과 ===" << std::endl;
            std::cout << "감지: " << detections.size() << ", 추적: " << tracks.size() 
                      << ", 신규 crossing: " << crossings.size() << std::endl;
            std::cout << "처리 시간 - 추론: " << inference_ms << "ms, 트래킹: " << tracking_ms 
                      << "ms, crossing: " << crossing_ms << "ms, 총: " << total_ms << "ms" << std::endl;
            
            for (const auto& track : tracks) {
                std::cout << "  - ID: " << track.id << ", person (신뢰도: " << std::fixed << std::setprecision(2) << track.confidence 
                         << ", 위치: " << track.bbox.x << "," << track.bbox.y << "," << track.bbox.width << "," << track.bbox.height << ")" << std::endl;
            }
            
            if (!crossings.empty()) {
                std::cout << "🚨 새로운 Line Crossing 이벤트: " << crossings.size() << "개" << std::endl;
            }
            std::cout << "==========================================" << std::endl;
        }
    }

    const std::vector<Detection>& getDetections() const { return detections; }
    const std::vector<Track>& getTracks() const { return tracks; }
    const LineCrossingDetector& getLineCrossingDetector() const { return line_detector; }

private:
    std::shared_ptr<ov::Core> core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    Sort sort_tracker;

    void setupDefaultLineCrossingZones() {
        // 1920x1080 해상도 기준으로 line crossing zone 설정
        // 화면 중앙 가로선
        line_detector.addZone(LineCrossingZone(
            Point(0, 540),      // 화면 왼쪽 중앙
            Point(1920, 540),   // 화면 오른쪽 중앙
            "center_horizontal"
        ));
        
        // 화면 1/3 지점 가로선
        line_detector.addZone(LineCrossingZone(
            Point(0, 360),      // 화면 왼쪽 1/3
            Point(1920, 360),   // 화면 오른쪽 1/3
            "upper_third"
        ));
        
        // 화면 2/3 지점 가로선
        line_detector.addZone(LineCrossingZone(
            Point(0, 720),      // 화면 왼쪽 2/3
            Point(1920, 720),   // 화면 오른쪽 2/3
            "lower_third"
        ));
        
        std::cout << "Line Crossing Zones 설정 완료:" << std::endl;
        std::cout << "  - center_horizontal: (0,540) - (1920,540)" << std::endl;
        std::cout << "  - upper_third: (0,360) - (1920,360)" << std::endl;
        std::cout << "  - lower_third: (0,720) - (1920,720)" << std::endl;
    }

    void performInference(const cv::Mat& frame) {
        // Letterbox preprocessing
        cv::Mat input_img;
        float scale;
        int pad_x, pad_y;
        letterbox(frame, input_img, scale, pad_x, pad_y);

        input_img.convertTo(input_img, CV_32F, 1.0 / 255.0);
        cv::Mat blob = cv::dnn::blobFromImage(input_img);

        ov::Tensor input_tensor = ov::Tensor(ov::element::f32,
                                             {1, 3, input_height, input_width},
                                             blob.ptr<float>());
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        ov::Tensor output = infer_request.get_output_tensor();
        const float* data = output.data<float>();
        auto shape = output.get_shape();  // [1, 25200, 85]

        detections.clear();
        
        // YOLOv5 postprocessing
        for (size_t i = 0; i < shape[1]; ++i) {
            const float* row = data + i * 85;
            float obj_conf = row[4];
            if (obj_conf < 0.01f) continue;

            float max_cls_score = 0.0f;
            int class_id = -1;
            for (int c = 0; c < 80; ++c) {
                if (row[5 + c] > max_cls_score) {
                    max_cls_score = row[5 + c];
                    class_id = c;
                }
            }

            float conf = obj_conf * max_cls_score;
            if (conf < conf_threshold || class_id != target_class) continue;

            float cx = row[0], cy = row[1], w = row[2], h = row[3];
            float x0 = (cx - w / 2 - pad_x) / scale;
            float y0 = (cy - h / 2 - pad_y) / scale;
            float x1 = (cx + w / 2 - pad_x) / scale;
            float y1 = (cy + h / 2 - pad_y) / scale;

            int x = std::clamp((int)x0, 0, frame.cols - 1);
            int y = std::clamp((int)y0, 0, frame.rows - 1);
            int box_w = std::min((int)(x1 - x0), frame.cols - x);
            int box_h = std::min((int)(y1 - y0), frame.rows - y);

            if (box_w > 0 && box_h > 0) {
                detections.push_back({class_id, conf, cv::Rect(x, y, box_w, box_h)});
            }
        }

        // Apply NMS
        auto results = nms(detections);
        detections = results;
    }

    void performTracking() {
        // SORT 알고리즘을 위한 detection 형식 변환
        std::vector<std::vector<float>> dets_for_sort;
        for (const auto& d : detections) {
            // [x1, y1, x2, y2, confidence, class_id]
            dets_for_sort.push_back({
                (float)d.box.x, 
                (float)d.box.y, 
                (float)(d.box.x + d.box.width), 
                (float)(d.box.y + d.box.height), 
                d.confidence, 
                (float)d.class_id
            });
        }

        // SORT 트래킹 수행
        auto tracked = sort_tracker.update(dets_for_sort);

        // 트래킹 결과를 Track 구조체로 변환
        tracks.clear();
        for (const auto& t : tracked) {
            int id = (int)t[6];           // track ID
            int class_id = (int)t[5];     // class ID
            float conf = t[4];            // confidence
            cv::Rect bbox((int)t[0], (int)t[1], (int)(t[2]-t[0]), (int)(t[3]-t[1]));
            
            Track track;
            track.id = id;
            track.bbox = bbox;
            track.class_id = class_id;
            track.confidence = conf;
            tracks.push_back(track);
        }
    }
};

class ZeroCopyOpenVINOYOLOv5LineCrossingTracker {
    // libcamera 및 버퍼 관련 멤버
    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<libcamera::CameraManager> cameraManager;
    std::unique_ptr<libcamera::CameraConfiguration> config;
    libcamera::Stream* stream;
    std::shared_ptr<libcamera::FrameBufferAllocator> allocator;
    std::vector<std::vector<void*>> bufferPlaneMappings;
    std::vector<std::vector<size_t>> bufferPlaneSizes;
    std::atomic<bool> stopping{false};

    std::unique_ptr<OpenVINOYOLOv5TrackerWithLineCrossing> yolo_tracker;

    // FPS 측정을 위한 변수들
    std::chrono::steady_clock::time_point lastTime;
    int frameCounter = 0;
    double fps = 0.0;

public:
    ZeroCopyOpenVINOYOLOv5LineCrossingTracker(const std::string& model_xml) {
        yolo_tracker = std::make_unique<OpenVINOYOLOv5TrackerWithLineCrossing>(model_xml);
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
        camera->requestCompleted.connect(this, &ZeroCopyOpenVINOYOLOv5LineCrossingTracker::onRequestCompleted);
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
        std::cout << "캡처 및 OpenVINO YOLOv5 + SORT + Line Crossing 시작..." << std::endl;
        return true;
    }

    void stop() {
        std::cout << "중지 신호 받음..." << std::endl;
        stopping.store(true);
        if (camera) {
            camera->stop();
            camera->requestCompleted.disconnect(this, &ZeroCopyOpenVINOYOLOv5LineCrossingTracker::onRequestCompleted);
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

        // 추론, 트래킹, line crossing 시간 측정 시작
        auto process_start = std::chrono::steady_clock::now();
        yolo_tracker->inferTrackAndCheckCrossings(frame);
        auto process_end = std::chrono::steady_clock::now();
        
        // 처리 시간 계산
        auto process_ms = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start).count();

        // FPS 계산
        frameCounter++;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastTime).count() >= 1) {
            fps = frameCounter / std::chrono::duration_cast<std::chrono::duration<double>>(now - lastTime).count();
            frameCounter = 0;
            lastTime = now;
            
            // 상세한 성능 정보 출력
            auto recent_crossings = yolo_tracker->getLineCrossingDetector().getRecentCrossings();
            std::cout << "[YOLOv5+SORT+LineCrossing FPS: " << std::fixed << std::setprecision(1) << fps 
                      << ", 처리시간: " << process_ms << "ms, 최근 crossing: " << recent_crossings.size() << "개] 프레임 처리 중..." << std::endl;
        }

        request->reuse(libcamera::Request::ReuseFlag::ReuseBuffers);
        if (!stopping.load()) camera->queueRequest(request);
    }
};

static std::atomic<bool> shouldExit{false};
static ZeroCopyOpenVINOYOLOv5LineCrossingTracker* demoInstance = nullptr;

void signalHandler(int signal) {
    std::cout << "\n종료 신호 받음 (Ctrl+C)" << std::endl;
    shouldExit.store(true);
    if (demoInstance) demoInstance->stop();
}

int main(int argc, char** argv) {
    std::cout << "=== Zero Copy OpenVINO YOLOv5 + SORT + Line Crossing Demo (Headless) ===" << std::endl;
    
    // YOLOv5 모델 경로
    std::string model_xml = "yolo5n_openvino_model/yolov5n.xml";
    
    std::cout << "YOLOv5 모델 파일: " << model_xml << std::endl;
    std::cout << "SORT 트래킹 활성화 (max_age=5, min_hits=2, iou_threshold=0.3)" << std::endl;
    std::cout << "Line Crossing 감지 활성화 (3개 가로선 영역)" << std::endl;
    
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    ZeroCopyOpenVINOYOLOv5LineCrossingTracker demo(model_xml);
    demoInstance = &demo;

    if (!demo.initialize()) {
        std::cout << "초기화 실패" << std::endl;
        return -1;
    }
    if (!demo.start()) {
        std::cout << "시작 실패" << std::endl;
        return -1;
    }

    std::cout << "YOLOv5 + SORT + Line Crossing 실행 중... (Ctrl+C로 종료)" << std::endl;
    while (!shouldExit.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "프로그램 종료" << std::endl;
    return 0;
}

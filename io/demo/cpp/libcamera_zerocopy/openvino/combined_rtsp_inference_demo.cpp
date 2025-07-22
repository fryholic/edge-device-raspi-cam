// combined_rtsp_inference_demo.cpp
//
// libcamera를 사용한 Zero-Copy 프레임 캡처 후,
// 1. GStreamer를 이용한 RTSP 스트리밍 (30fps)
// 2. OpenVINO YOLOv5 + SORT를 이용한 Line Crossing 추론 (최대 10fps)
// 두 작업을 동시에 수행하는 코드입니다.
//
// 주요 아키텍처:
// - 단일 카메라 소스 (BGR888, 1920x1080, 30fps)
// - 메인 스레드: 초기화 및 종료 처리
// - RTSP 전송 스레드: 카메라 콜백에서 받은 모든 프레임을 큐를 통해 RTSP 서버로 전송
// - AI 추론 스레드: 최신 프레임의 복사본을 가져와 BGR->RGB 변환 후 추론 수행
//
// 빌드 방법 (의존성: libcamera, gstreamer, opencv, openvino):
// make combined-demo

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <csignal>
#include <sys/mman.h>
#include <memory>
#include <functional>
#include <iomanip>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

// Libcamera
#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/stream.h>

// GStreamer
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/app/gstappsrc.h>

// OpenVINO & OpenCV
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

// SORT & Line Crossing (실제 프로젝트에 헤더 파일이 필요합니다)
#include "sort.hpp"
#include "object_tracker.hpp"

using namespace libcamera;
using namespace std::chrono;
using namespace std::literals::chrono_literals;

// ====================================================================
//                          공통 설정 및 구조체
// ====================================================================

// 카메라 및 RTSP 설정
constexpr int CAPTURE_WIDTH = 1920;
constexpr int CAPTURE_HEIGHT = 1080;
constexpr int CAPTURE_FPS = 30;
constexpr int RTSP_PORT = 8554;
constexpr const char* RTSP_MOUNT_POINT = "/stream";

// YOLOv5 설정
const int YoloInputWidth = 320;
const int YoloInputHeight = 320;
const float ConfThreshold = 0.3;
const float IouThreshold = 0.45;
const int TargetClass = 0; // person

// 이름 충돌 방지를 위한 네임스페이스
namespace App {
    // Line Crossing 관련 구조체
    struct Point {
        float x, y;
        Point(float x_ = 0, float y_ = 0) : x(x_), y(y_) {}
    };

    struct LineCrossingZone {
        Point start, end;
        std::string name;
        LineCrossingZone(const Point& s, const Point& e, const std::string& n) : start(s), end(e), name(n) {}
    };

    struct Detection {
        int class_id;
        float confidence;
        cv::Rect box;
    };

    struct CrossingEvent {
        int track_id;
        std::string zone_name;
        steady_clock::time_point timestamp;
        Point crossing_point;

        CrossingEvent(int id, const std::string& zone, const Point& point)
            : track_id(id), zone_name(zone), crossing_point(point),
              timestamp(steady_clock::now()) {}
    };
}

// ====================================================================
//                         RTSP 서버 클래스
// ====================================================================
class RTSPServer {
private:
    GMainLoop *loop;
    GstRTSPServer *server;
    GstRTSPMountPoints *mounts;
    GstRTSPMediaFactory *factory;
    GstAppSrc *appsrc;
    std::thread server_thread;
    std::atomic<bool> is_running;
    int width, height, fps;

public:
    RTSPServer(int port, const std::string& mount_point, int w, int h, int f)
        : loop(nullptr), server(nullptr), mounts(nullptr), factory(nullptr), appsrc(nullptr),
          is_running(false), width(w), height(h), fps(f) {
        gst_init(nullptr, nullptr);
    }

    ~RTSPServer() { stop(); gst_deinit(); }

    bool start() {
        loop = g_main_loop_new(NULL, FALSE);
        server = gst_rtsp_server_new();
        g_object_set(server, "service", std::to_string(RTSP_PORT).c_str(), NULL);
        mounts = gst_rtsp_server_get_mount_points(server);
        factory = gst_rtsp_media_factory_new();

        // ** 수정: videoconvert를 v4l2convert로 변경하여 하드웨어 호환성 및 안정성 향상 **
        std::string pipeline_str = "appsrc name=mysrc ! queue ! v4l2convert ! video/x-raw,format=NV12 ! v4l2h264enc ! rtph264pay name=pay0 pt=96";
        
        gst_rtsp_media_factory_set_launch(factory, pipeline_str.c_str());
        gst_rtsp_media_factory_set_shared(factory, TRUE);
        g_signal_connect(factory, "media-configure", (GCallback)media_configure_callback, this);
        
        gst_rtsp_mount_points_add_factory(mounts, RTSP_MOUNT_POINT, factory);
        g_object_unref(mounts);

        if (gst_rtsp_server_attach(server, NULL) == 0) return false;

        is_running.store(true);
        server_thread = std::thread([this]() {
            std::cout << "[INFO] RTSP stream ready at: rtsp://<your-ip-address>:" << RTSP_PORT << RTSP_MOUNT_POINT << std::endl;
            g_main_loop_run(loop);
        });
        return true;
    }

    void stop() {
        if (is_running.exchange(false)) {
            if (loop) g_main_loop_quit(loop);
            if (server_thread.joinable()) server_thread.join();
            if (server) g_object_unref(server);
            if (loop) g_main_loop_unref(loop);
        }
    }

    void pushFrame(const std::vector<uint8_t>& frame_data) {
        if (!is_running.load() || !appsrc || GST_STATE(appsrc) != GST_STATE_PLAYING) return;
        
        GstBuffer* buffer = gst_buffer_new_allocate(nullptr, frame_data.size(), nullptr);
        gst_buffer_fill(buffer, 0, frame_data.data(), frame_data.size());
        
        GstFlowReturn ret;
        g_signal_emit_by_name(appsrc, "push-buffer", buffer, &ret);
        gst_buffer_unref(buffer);
        if (ret != GST_FLOW_OK) {
            // This warning can be frequent if client is not connected, so we can comment it out
            // std::cerr << "[WARN] Error pushing buffer to appsrc, flow return: " << gst_flow_get_name(ret) << std::endl;
        }
    }

private:
    static void media_configure_callback(GstRTSPMediaFactory *f, GstRTSPMedia *m, gpointer u) {
        static_cast<RTSPServer*>(u)->on_media_configure(m);
    }

    void on_media_configure(GstRTSPMedia *media) {
        GstElement *pipeline = gst_rtsp_media_get_element(media);
        GstElement *appsrc_element = gst_bin_get_by_name(GST_BIN(pipeline), "mysrc");
        
        GstCaps* caps = gst_caps_new_simple("video/x-raw",
                                            "format", G_TYPE_STRING, "BGR",
                                            "width", G_TYPE_INT, width,
                                            "height", G_TYPE_INT, height,
                                            "framerate", GST_TYPE_FRACTION, fps, 1,
                                            NULL);
        
        g_object_set(G_OBJECT(appsrc_element), "caps", caps, "format", GST_FORMAT_TIME, "is-live", TRUE, "do-timestamp", TRUE, NULL);
        gst_caps_unref(caps);
        this->appsrc = GST_APP_SRC(appsrc_element);
        g_object_unref(appsrc_element);
    }
};

// ====================================================================
//                  AI 추론 및 Line Crossing 클래스
// ====================================================================

// Line Crossing Detector 클래스
class LineCrossingDetector {
private:
    std::vector<App::LineCrossingZone> zones;
    std::unordered_map<int, std::unordered_map<std::string, float>> prev_positions;
    std::vector<App::CrossingEvent> recent_crossings;
    
public:
    void addZone(const App::LineCrossingZone& zone) {
        zones.push_back(zone);
    }
    
    float getPositionRelativeToLine(const App::Point& pt, const App::Point& lineStart, const App::Point& lineEnd) {
        float A = lineEnd.y - lineStart.y;
        float B = lineStart.x - lineEnd.x;
        float C = (lineEnd.x * lineStart.y) - (lineStart.x * lineEnd.y);
        return A * pt.x + B * pt.y + C;
    }
    
    App::Point getBboxCenter(const cv::Rect& bbox) {
        return App::Point((bbox.x + bbox.x + bbox.width) / 2.0f,
                         (bbox.y + bbox.y + bbox.height) / 2.0f);
    }
    
    std::vector<App::CrossingEvent> checkCrossings(const std::vector<Track>& tracks) {
        std::vector<App::CrossingEvent> new_crossings;
        for (const auto& track : tracks) {
            App::Point center = getBboxCenter(track.bbox);
            for (const auto& zone : zones) {
                float position = getPositionRelativeToLine(center, zone.start, zone.end);
                if (prev_positions[track.id].count(zone.name)) {
                    float prev_pos = prev_positions[track.id][zone.name];
                    if (prev_pos * position < 0) {
                        App::CrossingEvent event(track.id, zone.name, center);
                        new_crossings.push_back(event);
                        recent_crossings.push_back(event);
                        std::cout << "🚨 LINE CROSSING: ID " << track.id << " crossed " << zone.name << std::endl;
                    }
                }
                prev_positions[track.id][zone.name] = position;
            }
        }
        
        auto now = steady_clock::now();
        recent_crossings.erase(
            std::remove_if(recent_crossings.begin(), recent_crossings.end(),
                [now](const App::CrossingEvent& event) {
                    return duration_cast<seconds>(now - event.timestamp).count() > 10;
                }),
            recent_crossings.end()
        );
        return new_crossings;
    }
    
    void clearOldTracks(const std::vector<Track>& active_tracks) {
        std::unordered_set<int> active_ids;
        for (const auto& track : active_tracks) active_ids.insert(track.id);
        
        for (auto it = prev_positions.begin(); it != prev_positions.end();) {
            if (active_ids.find(it->first) == active_ids.end()) it = prev_positions.erase(it);
            else ++it;
        }
    }
};

// 추론 관련 헬퍼 함수
namespace {
    float iou(const cv::Rect& a, const cv::Rect& b) {
        float inter = (a & b).area();
        float uni = a.area() + b.area() - inter;
        return (uni > 0) ? (inter / uni) : 0;
    }

    std::vector<App::Detection> nms(std::vector<App::Detection>& dets) {
        std::vector<App::Detection> res;
        if (dets.empty()) return res;
        std::sort(dets.begin(), dets.end(), [](const App::Detection& a, const App::Detection& b) {
            return a.confidence > b.confidence;
        });
        std::vector<bool> suppressed(dets.size(), false);
        for (size_t i = 0; i < dets.size(); ++i) {
            if (suppressed[i]) continue;
            res.push_back(dets[i]);
            for (size_t j = i + 1; j < dets.size(); ++j) {
                if (suppressed[j]) continue;
                if (dets[i].class_id == dets[j].class_id && iou(dets[i].box, dets[j].box) > IouThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        return res;
    }

    cv::Mat letterbox(const cv::Mat& src, float& scale, int& pad_x, int& pad_y) {
        cv::Mat out;
        int w = src.cols, h = src.rows;
        scale = std::min((float)YoloInputWidth / w, (float)YoloInputHeight / h);
        int new_w = std::round(w * scale);
        int new_h = std::round(h * scale);
        pad_x = (YoloInputWidth - new_w) / 2;
        pad_y = (YoloInputHeight - new_h) / 2;
        cv::resize(src, out, cv::Size(new_w, new_h));
        cv::copyMakeBorder(out, out, pad_y, YoloInputHeight - new_h - pad_y,
                                  pad_x, YoloInputWidth - new_w - pad_x,
                                  cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return out;
    }
}

class InferenceEngine {
public:
    InferenceEngine(const std::string& model_xml, const std::string& device = "CPU")
        : sort_tracker(5, 2, 0.3f) {
        core = std::make_shared<ov::Core>();
        model = core->read_model(model_xml);
        compiled_model = core->compile_model(model, device);
        infer_request = compiled_model.create_infer_request();
        setupDefaultLineCrossingZones();
        std::cout << "[INFO] Inference engine initialized (OpenVINO YOLOv5 + SORT)." << std::endl;
    }

    void processFrame(const cv::Mat& frame_bgr) {
        auto start_time = steady_clock::now();
        cv::Mat frame_rgb;
        cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);

        performInference(frame_rgb);
        performTracking();
        line_detector.checkCrossings(tracks);
        line_detector.clearOldTracks(tracks);

        auto total_ms = duration_cast<milliseconds>(steady_clock::now() - start_time).count();
        updateFPS(total_ms);
    }

private:
    std::shared_ptr<ov::Core> core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    Sort sort_tracker;
    LineCrossingDetector line_detector;
    std::vector<App::Detection> detections;
    std::vector<Track> tracks;
    std::atomic<int> frame_count_{0};
    double current_fps_ = 0.0;
    steady_clock::time_point last_fps_time_ = steady_clock::now();

    void performInference(const cv::Mat& frame) {
        float scale;
        int pad_x, pad_y;
        cv::Mat input_img = letterbox(frame, scale, pad_x, pad_y);
        
        cv::Mat blob = cv::dnn::blobFromImage(input_img, 1.0/255.0, cv::Size(YoloInputWidth, YoloInputHeight), cv::Scalar(), true);
        ov::Tensor input_tensor(ov::element::f32, {1, 3, (size_t)YoloInputHeight, (size_t)YoloInputWidth}, blob.ptr<float>());
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        
        ov::Tensor output = infer_request.get_output_tensor();
        const float* data = output.data<float>();
        auto shape = output.get_shape();

        std::vector<App::Detection> temp_detections;
        for (size_t i = 0; i < shape[1]; ++i) {
            const float* row = data + i * shape[2];
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
            if (conf < ConfThreshold || class_id != TargetClass) continue;

            float cx = row[0], cy = row[1], w = row[2], h = row[3];
            float x0 = (cx - w / 2 - pad_x) / scale;
            float y0 = (cy - h / 2 - pad_y) / scale;
            int x = std::clamp((int)x0, 0, (int)CAPTURE_WIDTH - 1);
            int y = std::clamp((int)y0, 0, (int)CAPTURE_HEIGHT - 1);
            int box_w = std::min((int)((cx + w / 2 - pad_x) / scale - x0), (int)CAPTURE_WIDTH - x);
            int box_h = std::min((int)((cy + h / 2 - pad_y) / scale - y0), (int)CAPTURE_HEIGHT - y);

            if (box_w > 0 && box_h > 0) {
                temp_detections.push_back({class_id, conf, cv::Rect(x, y, box_w, box_h)});
            }
        }
        detections = nms(temp_detections);
    }

    void performTracking() {
        std::vector<std::vector<float>> dets_for_sort;
        for (const auto& d : detections) {
            dets_for_sort.push_back({(float)d.box.x, (float)d.box.y, (float)(d.box.x + d.box.width), (float)(d.box.y + d.box.height), d.confidence, (float)d.class_id});
        }
        auto tracked = sort_tracker.update(dets_for_sort);
        tracks.clear();
        for (const auto& t : tracked) {
            tracks.push_back({(int)t[6], cv::Rect((int)t[0], (int)t[1], (int)(t[2]-t[0]), (int)(t[3]-t[1])), (int)t[5], t[4]});
        }
    }
    
    void setupDefaultLineCrossingZones() {
        line_detector.addZone(App::LineCrossingZone(App::Point(0, 540), App::Point(1920, 540), "center_horizontal"));
        line_detector.addZone(App::LineCrossingZone(App::Point(0, 360), App::Point(1920, 360), "upper_third"));
        line_detector.addZone(App::LineCrossingZone(App::Point(0, 720), App::Point(1920, 720), "lower_third"));
    }

    void updateFPS(long long total_ms) {
        frame_count_++;
        auto now = steady_clock::now();
        auto elapsed_ms = duration_cast<milliseconds>(now - last_fps_time_).count();
        if (elapsed_ms >= 1000) {
            current_fps_ = frame_count_ * 1000.0 / elapsed_ms;
            frame_count_ = 0;
            last_fps_time_ = now;
            std::cout << "[INFERENCE] FPS: " << std::fixed << std::setprecision(1) << current_fps_
                      << ", Time: " << total_ms << "ms"
                      << ", Dets: " << detections.size()
                      << ", Tracks: " << tracks.size() << std::endl;
        }
    }
};

// ====================================================================
//                  메인 컨트롤러 클래스
// ====================================================================
class CameraController {
private:
    std::shared_ptr<Camera> camera;
    std::unique_ptr<CameraManager> cameraManager;
    std::unique_ptr<CameraConfiguration> config;
    Stream* stream;
    std::shared_ptr<FrameBufferAllocator> allocator;
    std::vector<std::vector<void*>> bufferPlaneMappings;
    std::atomic<bool> stopping;
    std::thread rtsp_thread;
    std::thread inference_thread;
    std::unique_ptr<RTSPServer> rtsp_server;
    std::unique_ptr<InferenceEngine> inference_engine;
    std::mutex frame_mutex;
    std::vector<uint8_t> latest_frame_buffer;
    std::atomic<bool> new_frame_for_inference;
    std::queue<std::vector<uint8_t>> rtsp_frame_queue;
    std::mutex rtsp_queue_mutex;
    std::condition_variable rtsp_queue_cv;

public:
    CameraController(const std::string& model_path) : stream(nullptr), stopping(false), new_frame_for_inference(false) {
        inference_engine = std::make_unique<InferenceEngine>(model_path);
    }

    ~CameraController() { cleanup(); }

    bool initialize() {
        std::cout << "[INFO] Initializing CameraController..." << std::endl;
        cameraManager = std::make_unique<CameraManager>();
        if (cameraManager->start()) return false;
        if (cameraManager->cameras().empty()) return false;
        
        camera = cameraManager->get(cameraManager->cameras()[0]->id());
        if (camera->acquire()) return false;

        config = camera->generateConfiguration({StreamRole::Viewfinder});
        StreamConfiguration& streamConfig = config->at(0);
        streamConfig.size = Size(CAPTURE_WIDTH, CAPTURE_HEIGHT);
        streamConfig.pixelFormat = formats::BGR888;
        streamConfig.bufferCount = 8;
        config->validate();
        
        if (camera->configure(config.get()) < 0) return false;
        stream = streamConfig.stream();
        std::cout << "[INFO] Camera configured: " << streamConfig.size.width << "x" << streamConfig.size.height
                  << " " << streamConfig.pixelFormat.toString() << std::endl;

        rtsp_server = std::make_unique<RTSPServer>(RTSP_PORT, RTSP_MOUNT_POINT, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS);
        return setupBuffers();
    }

    bool setupBuffers() {
        allocator = std::make_shared<FrameBufferAllocator>(camera);
        if (allocator->allocate(stream) < 0) return false;
        
        for (const auto& buffer : allocator->buffers(stream)) {
            std::vector<void*> planeMappings;
            for (const auto& plane : buffer->planes()) {
                void* memory = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
                if (memory == MAP_FAILED) return false;
                planeMappings.push_back(memory);
            }
            bufferPlaneMappings.push_back(planeMappings);
        }
        latest_frame_buffer.resize(CAPTURE_WIDTH * CAPTURE_HEIGHT * 3);
        std::cout << "[INFO] " << bufferPlaneMappings.size() << " DMA buffers mapped." << std::endl;
        return true;
    }

    void cleanup() {
        stop();
        if (allocator) {
            const auto& buffers = allocator->buffers(stream);
            for (size_t i = 0; i < bufferPlaneMappings.size(); ++i) {
                for (size_t j = 0; j < bufferPlaneMappings[i].size(); ++j) {
                    if (bufferPlaneMappings[i][j] != MAP_FAILED) {
                        munmap(bufferPlaneMappings[i][j], buffers[i]->planes()[j].length);
                    }
                }
            }
        }
        if (camera) {
            camera->release();
        }
        std::cout << "[INFO] Cleanup complete." << std::endl;
    }

    bool start() {
        if (!camera || !rtsp_server || !inference_engine) return false;
        if (!rtsp_server->start()) return false;
        
        rtsp_thread = std::thread(&CameraController::rtspSendLoop, this);
        inference_thread = std::thread(&CameraController::inferenceLoop, this);
        camera->requestCompleted.connect(this, &CameraController::onRequestCompleted);

        if (camera->start()) return false;
        for (const auto& buffer : allocator->buffers(stream)) {
            auto request = camera->createRequest();
            request->addBuffer(stream, buffer.get());
            camera->queueRequest(request.release());
        }
        std::cout << "[INFO] Camera started. Streaming and inference begins." << std::endl;
        return true;
    }

    void stop() {
        if (stopping.exchange(true)) return;
        
        std::cout << "[INFO] Stopping CameraController..." << std::endl;
        rtsp_queue_cv.notify_all();
        
        if (inference_thread.joinable()) inference_thread.join();
        if (rtsp_thread.joinable()) rtsp_thread.join();
        
        if (rtsp_server) rtsp_server->stop();
        if (camera) {
            camera->stop();
            camera->requestCompleted.disconnect(this, &CameraController::onRequestCompleted);
        }
    }

    void onRequestCompleted(Request* request) {
        if (stopping.load() || request->status() != Request::RequestComplete) {
            if (request->status() != Request::RequestCancelled)
                request->reuse(Request::ReuseBuffers);
            camera->queueRequest(request);
            return;
        }

        FrameBuffer* buffer = request->buffers().begin()->second;
        
        const auto& buffers = allocator->buffers(stream);
        size_t bufferIndex = 0;
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (buffers[i].get() == buffer) {
                bufferIndex = i;
                break;
            }
        }

        void* data = bufferPlaneMappings[bufferIndex][0];
        size_t size = buffer->planes()[0].length;

        // ** 개선: RTSP 큐가 너무 길어지는 것을 방지하여 지연시간 누적을 막음 **
        {
            std::lock_guard<std::mutex> lock(rtsp_queue_mutex);
            if (rtsp_frame_queue.size() < 5) {
                std::vector<uint8_t> frame_copy(size);
                memcpy(frame_copy.data(), data, size);
                rtsp_frame_queue.push(std::move(frame_copy));
                rtsp_queue_cv.notify_one();
            }
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            memcpy(latest_frame_buffer.data(), data, size);
            new_frame_for_inference.store(true);
        }
        
        request->reuse(Request::ReuseBuffers);
        camera->queueRequest(request);
    }
    
    void rtspSendLoop() {
        while (!stopping.load()) {
            std::vector<uint8_t> frame_to_send;
            {
                std::unique_lock<std::mutex> lock(rtsp_queue_mutex);
                rtsp_queue_cv.wait(lock, [this]{ return stopping.load() || !rtsp_frame_queue.empty(); });
                if (stopping.load()) break;
                
                frame_to_send = std::move(rtsp_frame_queue.front());
                rtsp_frame_queue.pop();
            }
            if (!frame_to_send.empty()) {
                rtsp_server->pushFrame(frame_to_send);
            }
        }
        std::cout << "[INFO] RTSP sending thread finished." << std::endl;
    }

    void inferenceLoop() {
        while (!stopping.load()) {
            if (new_frame_for_inference.load()) {
                std::vector<uint8_t> frame_copy;
                {
                    std::lock_guard<std::mutex> lock(frame_mutex);
                    if (!new_frame_for_inference.exchange(false)) continue;
                    frame_copy = latest_frame_buffer;
                }

                if (!frame_copy.empty()) {
                    cv::Mat frame(CAPTURE_HEIGHT, CAPTURE_WIDTH, CV_8UC3, frame_copy.data());
                    inference_engine->processFrame(frame);
                }
            } else {
                std::this_thread::sleep_for(10ms);
            }
        }
        std::cout << "[INFO] Inference thread finished." << std::endl;
    }
};

static std::atomic<bool> shouldExit{false};
static CameraController* controllerInstance = nullptr;

void signalHandler(int signal) {
    std::cout << "\n[INFO] Signal " << signal << " received. Exiting gracefully..." << std::endl;
    shouldExit.store(true);
    if (controllerInstance) {
        controllerInstance->stop();
    }
}

int main(int argc, char** argv) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::string model_path = "yolov5n.xml";
    if (argc > 1) model_path = argv[1];

    std::cout << "======================================================" << std::endl;
    std::cout << "  Combined RTSP Streamer & YOLOv5 Inference Demo" << std::endl;
    std::cout << "  - YOLO Model: " << model_path << std::endl;
    std::cout << "======================================================" << std::endl;

    try {
        CameraController controller(model_path);
        controllerInstance = &controller;

        if (!controller.initialize()) {
            std::cerr << "[FATAL] Initialization failed." << std::endl;
            return -1;
        }

        if (!controller.start()) {
            std::cerr << "[FATAL] Controller start failed." << std::endl;
            return -1;
        }

        while (!shouldExit.load()) {
            std::this_thread::sleep_for(100ms);
        }

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] An unhandled exception occurred: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "[INFO] Application finished." << std::endl;
    return 0;
}

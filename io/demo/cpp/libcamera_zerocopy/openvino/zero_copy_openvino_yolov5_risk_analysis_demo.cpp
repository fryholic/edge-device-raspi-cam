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
#include <regex>
#include <cmath>
#include <mutex>
#include <deque>
#include <limits>

#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/stream.h>
#include <SQLiteCpp/SQLiteCpp.h>

// 트래킹 관련 헤더 추가
#include "sort.hpp"
#include "object_tracker.hpp"

// YOLOv5 constants
const int input_width = 320;
const int input_height = 320;
const float conf_threshold = 0.3;
const float iou_threshold = 0.45;
const int target_class = 0; // class 0: person

// 서버 통신 설정
const std::string SERVER_URL = "http://192.168.0.137:3000"; // 서버 IP 및 포트
const std::string DB_FILE = "../server_log.db"; // 로컬 DB 경로 (서버와 동일한 DB 사용)

// Point 구조체 정의 (main_control.cpp와 동일)
struct Point {
    float x;
    float y;
    
    Point() : x(0), y(0) {}
    Point(float x_, float y_) : x(x_), y(y_) {}
};

// Line 구조체 정의 (main_control.cpp와 동일)  
struct Line {
    Point start;
    Point end;
    std::string mode;
    std::string name;
};

// LineCrossingZone 구조체 (기존)
struct LineCrossingZone {
    Point start;
    Point end;
    std::string name;
    std::string mode;
    int leftMatrixNum;
    int rightMatrixNum;
    
    LineCrossingZone(const Point& s, const Point& e, const std::string& n, 
                     const std::string& m = "BothDirections", int left = 0, int right = 0) 
        : start(s), end(e), name(n), mode(m), leftMatrixNum(left), rightMatrixNum(right) {}
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

// ObjectState 구조체 (main_control.cpp와 동일)
struct ObjectState {
    std::deque<Point> history;
};

// 전역 변수들 (main_control.cpp에서 가져온 것들)
std::recursive_mutex data_mutex;
std::vector<std::tuple<int, Point, int, Point>> base_line_pairs;
Point dot_center = {0, 0};
std::unordered_map<std::string, Line> rule_lines;
std::unordered_map<int, ObjectState> vehicle_trajectory_history;

// 설정값 (main_control.cpp와 동일)
constexpr float dist_threshold = 10.0f;         
constexpr float parrallelism_threshold = 0.75f; 
constexpr int HISTORY_SIZE = 10; 

// 서버에서 라인 정보 가져오기 (간단한 HTTP 클라이언트 또는 DB 직접 접근)
bool loadLineConfigsFromServer() {
    try {
        SQLite::Database db(DB_FILE, SQLite::OPEN_READONLY);
        
        std::lock_guard<std::recursive_mutex> lock(data_mutex);
        rule_lines.clear();
        
        // lines 테이블에서 데이터 로드 (서버와 동일한 방식)
        const float scale_x = 3840.0f / 960.0f;
        const float scale_y = 2160.0f / 540.0f;
        
        SQLite::Statement query(db, "SELECT x1, y1, x2, y2, name, mode FROM lines LIMIT 8");

        while (query.executeStep()) {
            Line line;
            line.start = { query.getColumn(0).getInt() * scale_x, query.getColumn(1).getInt() * scale_y };
            line.end   = { query.getColumn(2).getInt() * scale_x, query.getColumn(3).getInt() * scale_y };
            line.name = query.getColumn(4).getString();
            line.mode = query.getColumn(5).getString();

            rule_lines[line.name] = line;

            std::cout << "[DEBUG] Loaded line from server: " << line.name
                      << " [(" << line.start.x << "," << line.start.y << ") -> ("
                      << line.end.x << "," << line.end.y << ")] Mode: " << line.mode << std::endl;
        }
        
        std::cout << "[INFO] Successfully loaded " << rule_lines.size() << " lines from server." << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load line configs from server: " << e.what() << std::endl;
        return false;
    }
}

// 서버에서 base line pairs 가져오기
bool loadBaseLinePairsFromServer() {
    try {
        SQLite::Database db(DB_FILE, SQLite::OPEN_READONLY);
        
        std::lock_guard<std::recursive_mutex> lock(data_mutex);
        base_line_pairs.clear();
        
        const float scale_x = 3840.0f / 960.0f;
        const float scale_y = 2160.0f / 540.0f;
        
        SQLite::Statement query(db, "SELECT matrixNum1, x1, y1, matrixNum2, x2, y2 FROM baseLines");

        while (query.executeStep()) {
            int id1 = query.getColumn(0).getInt();
            Point p1 = {query.getColumn(1).getInt() * scale_x, query.getColumn(2).getInt() * scale_y};
            int id2 = query.getColumn(3).getInt();
            Point p2 = {query.getColumn(4).getInt() * scale_x, query.getColumn(5).getInt() * scale_y};

            base_line_pairs.emplace_back(id1, p1, id2, p2);

            std::cout << "[DEBUG] Loaded base line pair: " << id1 << "<->" << id2
                      << " (" << p1.x << "," << p1.y << ") <-> (" << p2.x << "," << p2.y << ")" << std::endl;
        }
        
        // dot_center 계산 (main_control.cpp와 동일한 로직)
        if (base_line_pairs.size() >= 2) {
            // 교차점 계산 로직 (간단히 중점으로 계산)
            dot_center = {
                (std::get<1>(base_line_pairs[0]).x + std::get<3>(base_line_pairs[0]).x + 
                 std::get<1>(base_line_pairs[1]).x + std::get<3>(base_line_pairs[1]).x) / 4,
                (std::get<1>(base_line_pairs[0]).y + std::get<3>(base_line_pairs[0]).y + 
                 std::get<1>(base_line_pairs[1]).y + std::get<3>(base_line_pairs[1]).y) / 4
            };
        } else if (base_line_pairs.size() == 1) {
            dot_center = {
                (std::get<1>(base_line_pairs[0]).x + std::get<3>(base_line_pairs[0]).x) / 2,
                (std::get<1>(base_line_pairs[0]).y + std::get<3>(base_line_pairs[0]).y) / 2
            };
        }
        
        std::cout << "[INFO] Calculated dot_center: (" << dot_center.x << ", " << dot_center.y << ")" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load base line pairs from server: " << e.what() << std::endl;
        return false;
    }
}

// main_control.cpp의 코사인 유사도 계산 함수
float compute_cosine_similarity(const Point& a, const Point& b) {
    float dot = a.x * b.x + a.y * b.y;
    float mag_a = sqrt(a.x * a.x + a.y * a.y);
    float mag_b = sqrt(b.x * b.x + b.y * b.y);
    if (mag_a == 0 || mag_b == 0) return -2.0f;
    return dot / (mag_a * mag_b);
}

// main_control.cpp의 analyze_risk_and_alert 로직을 이식한 함수
void analyze_risk_and_alert_edge(int human_id, const std::string& rule_name) {
    std::lock_guard<std::recursive_mutex> lock(data_mutex);

    std::cout << "[DEBUG] Analyzing risk for human_id: " << human_id << " crossing line: " << rule_name << std::endl;

    // 1. 이벤트 라인 정보 확인
    if (rule_lines.find(rule_name) == rule_lines.end()) {
        std::cout << "[DEBUG] Step Failed: RuleName '" << rule_name << "' not found in predefined lines." << std::endl;
        return;
    }

    Line crossed_line = rule_lines.at(rule_name);
    Point line_vector = {
        crossed_line.end.x - crossed_line.start.x,
        crossed_line.end.y - crossed_line.start.y
    };

    // 2. 차량 이력 존재 확인
    std::cout << "[DEBUG] Vehicle history size: " << vehicle_trajectory_history.size() << std::endl;
    if (vehicle_trajectory_history.empty()) {
        std::cout << "[DEBUG] Step Failed: No vehicles detected to analyze." << std::endl;
        return;
    }

    // 3. 각 차량 반복
    for (const auto& [vehicle_id, vehicle_state] : vehicle_trajectory_history) {
        std::cout << "[DEBUG] Vehicle " << vehicle_id << " history size: " << vehicle_state.history.size() << std::endl;

        if (vehicle_state.history.size() < 2) {
            std::cout << "[DEBUG] Vehicle " << vehicle_id << ": Insufficient history. Skipping." << std::endl;
            continue;
        }

        // 가장 오래된 위치와 최근 위치 추출
        const Point& oldest_pos = vehicle_state.history.front();
        const Point& newest_pos = vehicle_state.history.back();

        // 4. 가장 가까운 dot 쌍 탐색
        Point closest_dot;
        int matched_id = -1;
        int board_id = -1;
        float min_dist_sq = std::numeric_limits<float>::max();

        for (const auto& [id1, p1, id2, p2] : base_line_pairs) {
            float d1 = pow(oldest_pos.x - p1.x, 2) + pow(oldest_pos.y - p1.y, 2);
            float d2 = pow(oldest_pos.x - p2.x, 2) + pow(oldest_pos.y - p2.y, 2);

            if (d1 < min_dist_sq) {
                min_dist_sq = d1;
                closest_dot = p1;
                matched_id = id1;
                board_id = id2;
            }

            if (d2 < min_dist_sq) {
                min_dist_sq = d2;
                closest_dot = p2;
                matched_id = id2;
                board_id = id1;
            }
        }

        std::cout << "[DEBUG] Vehicle " << vehicle_id << ": Closest dot = (" << closest_dot.x << "," << closest_dot.y << ")" << std::endl;

        // 5. dot_center 접근 여부 확인
        float dist_old = hypot(oldest_pos.x - dot_center.x, oldest_pos.y - dot_center.y);
        float dist_new = hypot(newest_pos.x - dot_center.x, newest_pos.y - dot_center.y);

        std::cout << "[DEBUG] Vehicle " << vehicle_id << ": Old dist to dot_center = " << dist_old << ", New dist = " << dist_new << std::endl;

        if (dist_new > dist_old - dist_threshold) {
            std::cout << "[DEBUG] Step Failed (Vehicle " << vehicle_id << "): Not approaching dot_center enough." << std::endl;
            continue;
        }

        // 6. 벡터 유사도 분석
        Point vehicle_vector = {dot_center.x - closest_dot.x, dot_center.y - closest_dot.y};
        float similarity = compute_cosine_similarity(vehicle_vector, line_vector);

        std::cout << "[DEBUG] Vehicle " << vehicle_id << ": Cosine similarity = " << similarity << ", Threshold = " << parrallelism_threshold << std::endl;

        if (abs(similarity) >= parrallelism_threshold) {
            std::cout << "\n[ALERT] 🚨 위험 감지! 🚨" << std::endl;
            std::cout << "차량 " << vehicle_id << "이 사람 " << human_id << "을 향해 측면에서 접근 중입니다." << std::endl;
            std::cout << "Matrix " << board_id << "를 가동해야 합니다." << std::endl;
            std::cout << "(코사인 유사도: " << similarity << ")" << std::endl;
            
            // TODO: 실제 환경에서는 여기서 LED Matrix 제어 신호를 보내거나 경고음을 울릴 수 있음
            
        } else {
            std::cout << "[DEBUG] Step Failed (Vehicle " << vehicle_id << "): Cosine similarity not high enough." << std::endl;
        }
    }
}

// Line crossing 계산 및 위험 분석이 통합된 LineCrossingDetector
class EnhancedLineCrossingDetector {
private:
    std::vector<LineCrossingZone> zones;
    std::unordered_map<int, std::unordered_map<std::string, float>> prev_positions;
    std::vector<CrossingEvent> recent_crossings;
    std::mutex zones_mutex;
    std::chrono::steady_clock::time_point last_config_load;
    
public:
    EnhancedLineCrossingDetector() {
        last_config_load = std::chrono::steady_clock::now();
        loadConfigurations();
    }
    
    // 서버에서 설정 로드
    void loadConfigurations() {
        std::cout << "[EnhancedLineCrossingDetector] Loading configurations from server..." << std::endl;
        
        // 서버에서 라인 설정과 base line pairs 로드
        bool lines_ok = loadLineConfigsFromServer();
        bool base_ok = loadBaseLinePairsFromServer();
        
        if (lines_ok && base_ok) {
            std::cout << "[EnhancedLineCrossingDetector] Server configurations loaded successfully." << std::endl;
            
            // rule_lines를 zones로 변환
            std::lock_guard<std::mutex> lock(zones_mutex);
            zones.clear();
            
            for (const auto& [name, line] : rule_lines) {
                LineCrossingZone zone(line.start, line.end, name, line.mode);
                zones.push_back(zone);
                std::cout << "[DEBUG] Added zone from server: " << zone.name << std::endl;
            }
        } else {
            std::cout << "[EnhancedLineCrossingDetector] Failed to load from server, using fallback zones..." << std::endl;
            initializeFallbackZones();
        }
    }
    
    // 폴백 영역 초기화
    void initializeFallbackZones() {
        std::lock_guard<std::mutex> lock(zones_mutex);
        zones.clear();
        
        zones.emplace_back(Point(100, 150), Point(540, 150), "Zone1");
        zones.emplace_back(Point(100, 240), Point(540, 240), "Zone2");
        zones.emplace_back(Point(100, 330), Point(540, 330), "Zone3");
        
        std::cout << "[EnhancedLineCrossingDetector] Initialized " << zones.size() << " fallback zones." << std::endl;
    }
    
    // 주기적으로 서버 설정 업데이트 확인
    void checkForUpdates() {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_config_load).count() >= 30) {
            loadConfigurations();
            last_config_load = now;
        }
    }
    
    Point getBboxCenter(const cv::Rect& bbox) {
        return Point((bbox.x + bbox.x + bbox.width) / 2.0f,
                    (bbox.y + bbox.y + bbox.height) / 2.0f);
    }
    
    float getPositionRelativeToLine(const Point& pt, const Point& lineStart, const Point& lineEnd) {
        float A = lineEnd.y - lineStart.y;
        float B = lineStart.x - lineEnd.x;
        float C = (lineEnd.x * lineStart.y) - (lineStart.x * lineEnd.y);
        
        return A * pt.x + B * pt.y + C;
    }
    
    // 차량 위치 업데이트 (main_control.cpp 로직 적용)
    void updateVehiclePositions(const std::vector<Track>& tracks) {
        std::lock_guard<std::recursive_mutex> lock(data_mutex);
        
        std::unordered_map<int, bool> seen_vehicles;
        
        for (const auto& track : tracks) {
            // class 0이 person이므로 vehicle tracking은 별도 구현 필요
            // 여기서는 예시로 person을 vehicle로 간주
            Point center = getBboxCenter(track.bbox);
            
            seen_vehicles[track.id] = true;
            auto& state = vehicle_trajectory_history[track.id];
            state.history.push_back(center);
            if (state.history.size() > HISTORY_SIZE) {
                state.history.pop_front();
            }
            
            std::cout << "[DEBUG] Tracking Object " << track.id << " at (" << center.x << ", " << center.y << ")" << std::endl;
        }
        
        // 사라진 객체 정리
        for (auto it = vehicle_trajectory_history.begin(); it != vehicle_trajectory_history.end(); ) {
            if (seen_vehicles.find(it->first) == seen_vehicles.end()) {
                std::cout << "[DEBUG] Object " << it->first << " disappeared. Erasing." << std::endl;
                it = vehicle_trajectory_history.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    // Line crossing 검사 및 위험 분석 통합
    std::vector<CrossingEvent> checkCrossingsWithRiskAnalysis(const std::vector<Track>& tracks) {
        std::lock_guard<std::mutex> lock(zones_mutex);
        
        // 주기적으로 서버 설정 업데이트 확인
        checkForUpdates();
        
        // 차량 위치 업데이트
        updateVehiclePositions(tracks);
        
        std::vector<CrossingEvent> new_crossings;
        
        for (const auto& track : tracks) {
            Point center = getBboxCenter(track.bbox);
            
            for (const auto& zone : zones) {
                float position = getPositionRelativeToLine(center, zone.start, zone.end);
                
                if (prev_positions[track.id].find(zone.name) != prev_positions[track.id].end()) {
                    float prev_pos = prev_positions[track.id][zone.name];
                    
                    if (prev_pos * position < 0) {
                        CrossingEvent event(track.id, zone.name, center);
                        new_crossings.push_back(event);
                        recent_crossings.push_back(event);
                        
                        std::cout << "🚨 LINE CROSSING DETECTED! 🚨" << std::endl;
                        std::cout << "  - Object ID: " << track.id << std::endl;
                        std::cout << "  - Zone: " << zone.name << std::endl;
                        std::cout << "  - Position: (" << center.x << ", " << center.y << ")" << std::endl;
                        
                        // main_control.cpp의 위험 분석 로직 적용
                        analyze_risk_and_alert_edge(track.id, zone.name);
                    }
                }
                
                prev_positions[track.id][zone.name] = position;
            }
        }
        
        // 오래된 crossing 이벤트 정리
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

// 향상된 OpenVINO YOLOv5 inference class
class EnhancedOpenVINOYOLOv5TrackerWithRiskAnalysis {
public:
    EnhancedOpenVINOYOLOv5TrackerWithRiskAnalysis(const std::string& model_xml, const std::string& device = "CPU") 
        : sort_tracker(5, 2, 0.3f) {
        core = std::make_shared<ov::Core>();
        model = core->read_model(model_xml);
        compiled_model = core->compile_model(model, device);
        infer_request = compiled_model.create_infer_request();
        
        std::cout << "Enhanced OpenVINO YOLOv5 + SORT + Risk Analysis 초기화 완료" << std::endl;
        std::cout << "입력 크기: " << input_width << "x" << input_height << std::endl;
    }

    std::vector<Detection> detections;
    std::vector<Track> tracks;
    EnhancedLineCrossingDetector enhanced_detector;

    void inferTrackAndAnalyze(const cv::Mat& frame) {
        auto start_time = std::chrono::steady_clock::now();
        
        // 1. YOLOv5 추론
        performInference(frame);
        
        auto inference_time = std::chrono::steady_clock::now();
        
        // 2. SORT 트래킹 수행
        performTracking();
        
        auto tracking_time = std::chrono::steady_clock::now();
        
        // 3. Line crossing 검사 및 위험 분석 (main_control.cpp 로직 포함)
        auto crossings = enhanced_detector.checkCrossingsWithRiskAnalysis(tracks);
        enhanced_detector.clearOldTracks(tracks);
        
        auto analysis_time = std::chrono::steady_clock::now();

        // 시간 측정 결과
        auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_time - start_time).count();
        auto tracking_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_time - inference_time).count();
        auto analysis_ms = std::chrono::duration_cast<std::chrono::milliseconds>(analysis_time - tracking_time).count();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(analysis_time - start_time).count();

        // 결과 출력
        if (!tracks.empty() || !crossings.empty()) {
            std::cout << "=== Enhanced YOLOv5 + SORT + Risk Analysis 결과 ===" << std::endl;
            std::cout << "감지: " << detections.size() << ", 추적: " << tracks.size() 
                      << ", 신규 crossing: " << crossings.size() << std::endl;
            std::cout << "처리 시간 - 추론: " << inference_ms << "ms, 트래킹: " << tracking_ms 
                      << "ms, 위험분석: " << analysis_ms << "ms, 총: " << total_ms << "ms" << std::endl;
            
            for (const auto& track : tracks) {
                std::cout << "  - ID: " << track.id << ", person (신뢰도: " << std::fixed << std::setprecision(2) << track.confidence 
                         << ", 위치: " << track.bbox.x << "," << track.bbox.y << "," << track.bbox.width << "," << track.bbox.height << ")" << std::endl;
            }
            
            if (!crossings.empty()) {
                std::cout << "🚨 새로운 Line Crossing 이벤트 및 위험 분석 완료: " << crossings.size() << "개" << std::endl;
            }
            std::cout << "=========================================" << std::endl;
        }
    }

    const std::vector<Detection>& getDetections() const { return detections; }
    const std::vector<Track>& getTracks() const { return tracks; }
    const EnhancedLineCrossingDetector& getEnhancedDetector() const { return enhanced_detector; }

private:
    std::shared_ptr<ov::Core> core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    Sort sort_tracker;

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
        auto shape = output.get_shape();

        detections.clear();
        
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

        auto results = nms(detections);
        detections = results;
    }

    void performTracking() {
        std::vector<std::vector<float>> dets_for_sort;
        for (const auto& d : detections) {
            dets_for_sort.push_back({
                (float)d.box.x, 
                (float)d.box.y, 
                (float)(d.box.x + d.box.width), 
                (float)(d.box.y + d.box.height), 
                d.confidence, 
                (float)d.class_id
            });
        }

        auto tracked = sort_tracker.update(dets_for_sort);

        tracks.clear();
        for (const auto& t : tracked) {
            int id = (int)t[6];
            int class_id = (int)t[5];
            float conf = t[4];
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

class EnhancedZeroCopyOpenVINOTracker {
    // libcamera 및 버퍼 관련 멤버
    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<libcamera::CameraManager> cameraManager;
    std::unique_ptr<libcamera::CameraConfiguration> config;
    libcamera::Stream* stream;
    std::shared_ptr<libcamera::FrameBufferAllocator> allocator;
    std::vector<std::vector<void*>> bufferPlaneMappings;
    std::vector<std::vector<size_t>> bufferPlaneSizes;
    std::atomic<bool> stopping{false};

    std::unique_ptr<EnhancedOpenVINOYOLOv5TrackerWithRiskAnalysis> enhanced_tracker;

    // FPS 측정을 위한 변수들
    std::chrono::steady_clock::time_point lastTime;
    int frameCounter = 0;
    double fps = 0.0;

public:
    EnhancedZeroCopyOpenVINOTracker(const std::string& model_xml) {
        enhanced_tracker = std::make_unique<EnhancedOpenVINOYOLOv5TrackerWithRiskAnalysis>(model_xml);
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
        camera->requestCompleted.connect(this, &EnhancedZeroCopyOpenVINOTracker::onRequestCompleted);
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
        std::cout << "캡처 및 Enhanced YOLOv5 + SORT + Risk Analysis 시작..." << std::endl;
        return true;
    }

    void stop() {
        std::cout << "중지 신호 받음..." << std::endl;
        stopping.store(true);
        if (camera) {
            camera->stop();
            camera->requestCompleted.disconnect(this, &EnhancedZeroCopyOpenVINOTracker::onRequestCompleted);
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

        // 추론, 트래킹, line crossing, 위험분석 시간 측정 시작
        auto process_start = std::chrono::steady_clock::now();
        enhanced_tracker->inferTrackAndAnalyze(frame);
        auto process_end = std::chrono::steady_clock::now();
        
        auto process_ms = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start).count();

        // FPS 계산
        frameCounter++;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastTime).count() >= 1) {
            fps = frameCounter / std::chrono::duration_cast<std::chrono::duration<double>>(now - lastTime).count();
            frameCounter = 0;
            lastTime = now;
            
            auto recent_crossings = enhanced_tracker->getEnhancedDetector().getRecentCrossings();
            std::cout << "[Enhanced YOLOv5+Risk Analysis FPS: " << std::fixed << std::setprecision(1) << fps 
                      << ", 처리시간: " << process_ms << "ms, 최근 crossing: " << recent_crossings.size() << "개] 프레임 처리 중..." << std::endl;
        }

        request->reuse(libcamera::Request::ReuseFlag::ReuseBuffers);
        if (!stopping.load()) camera->queueRequest(request);
    }
};

static std::atomic<bool> shouldExit{false};
static EnhancedZeroCopyOpenVINOTracker* demoInstance = nullptr;

void signalHandler(int signal) {
    std::cout << "\n종료 신호 받음 (Ctrl+C)" << std::endl;
    shouldExit.store(true);
    if (demoInstance) demoInstance->stop();
}

int main(int argc, char** argv) {
    std::cout << "=== Enhanced Zero Copy OpenVINO YOLOv5 + Risk Analysis Demo ===" << std::endl;
    std::cout << "main_control.cpp의 analyze_risk_and_alert 로직이 통합된 버전" << std::endl;
    
    // YOLOv5 모델 경로
    std::string model_xml = "yolo5n_openvino_model/yolov5n.xml";
    
    std::cout << "YOLOv5 모델 파일: " << model_xml << std::endl;
    std::cout << "SORT 트래킹 활성화 (max_age=5, min_hits=2, iou_threshold=0.3)" << std::endl;
    std::cout << "Line Crossing 감지 + Risk Analysis 활성화" << std::endl;
    std::cout << "서버 DB 경로: " << DB_FILE << std::endl;
    
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    EnhancedZeroCopyOpenVINOTracker demo(model_xml);
    demoInstance = &demo;

    if (!demo.initialize()) {
        std::cout << "초기화 실패" << std::endl;
        return -1;
    }
    if (!demo.start()) {
        std::cout << "시작 실패" << std::endl;
        return -1;
    }

    std::cout << "Enhanced YOLOv5 + Risk Analysis 실행 중... (Ctrl+C로 종료)" << std::endl;
    while (!shouldExit.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "프로그램 종료" << std::endl;
    return 0;
}

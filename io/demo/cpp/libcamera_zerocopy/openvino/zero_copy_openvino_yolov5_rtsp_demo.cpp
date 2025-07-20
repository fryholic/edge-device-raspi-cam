#include <iostream>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <future>
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
#include <condition_variable>
#include <queue>
#include <sstream>

#include <libcamera/libcamera.h>
#include <libcamera/framebuffer.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/stream.h>
#include <SQLiteCpp/SQLiteCpp.h>

// GStreamer for RTSP streaming
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>

// 트래킹 관련 헤더 추가
#include "sort.hpp"
#include "object_tracker.hpp"

// YOLOv5 constants
const int input_width = 320;
const int input_height = 320;
const float conf_threshold = 0.3;
const float iou_threshold = 0.45;
const int target_class = 0; // class 0: person

// RTSP 설정
const int RTSP_PORT = 9554;
const std::string RTSP_PATH = "/stream";
const int STREAM_WIDTH = 1920;
const int STREAM_HEIGHT = 1080;
const int TARGET_FPS = 30; // 30fps로 증가

// 서버 통신 설정
const std::string SERVER_URL = "http://192.168.0.137:3000"; // 서버 IP 및 포트
const std::string DB_FILE = "./server_log.db"; // 로컬 DB 경로 (상대 경로 수정)

// YOLOv5 모델 파일 경로 (하드코딩)
const std::string YOLO_MODEL_PATH = "yolo5n_openvino_model/yolov5n.xml";

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
        : track_id(id), zone_name(zone), timestamp(std::chrono::steady_clock::now()), 
          crossing_point(point) {}
};

// RTSP 메타데이터 구조체
struct RTSPMetadata {
    int frame_number;
    std::chrono::steady_clock::time_point timestamp;
    int detection_count;
    int tracking_count;
    std::vector<Detection> detections;
    std::vector<Track> tracks;
    std::vector<CrossingEvent> crossings;
    std::string risk_alerts;
    double fps;
    
    // JSON 형태로 직렬화
    std::string toJson() const {
        std::stringstream ss;
        ss << "{";
        ss << "\"frame_number\":" << frame_number << ",";
        ss << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count() << ",";
        ss << "\"detection_count\":" << detection_count << ",";
        ss << "\"tracking_count\":" << tracking_count << ",";
        ss << "\"fps\":" << std::fixed << std::setprecision(2) << fps << ",";
        
        // Detections
        ss << "\"detections\":[";
        for (size_t i = 0; i < detections.size(); ++i) {
            if (i > 0) ss << ",";
            ss << "{\"class_id\":" << detections[i].class_id 
               << ",\"confidence\":" << detections[i].confidence
               << ",\"box\":{\"x\":" << detections[i].box.x 
               << ",\"y\":" << detections[i].box.y 
               << ",\"width\":" << detections[i].box.width 
               << ",\"height\":" << detections[i].box.height << "}}";
        }
        ss << "],";
        
        // Tracks
        ss << "\"tracks\":[";
        for (size_t i = 0; i < tracks.size(); ++i) {
            if (i > 0) ss << ",";
            ss << "{\"id\":" << tracks[i].id 
               << ",\"class_id\":" << tracks[i].class_id
               << ",\"confidence\":" << tracks[i].confidence
               << ",\"box\":{\"x\":" << tracks[i].bbox.x 
               << ",\"y\":" << tracks[i].bbox.y 
               << ",\"width\":" << tracks[i].bbox.width 
               << ",\"height\":" << tracks[i].bbox.height << "}}";
        }
        ss << "],";
        
        // Crossings
        ss << "\"crossings\":[";
        for (size_t i = 0; i < crossings.size(); ++i) {
            if (i > 0) ss << ",";
            ss << "{\"track_id\":" << crossings[i].track_id 
               << ",\"zone_name\":\"" << crossings[i].zone_name << "\""
               << ",\"point\":{\"x\":" << crossings[i].crossing_point.x 
               << ",\"y\":" << crossings[i].crossing_point.y << "}}";
        }
        ss << "],";
        
        // Risk alerts
        ss << "\"risk_alerts\":\"" << risk_alerts << "\"";
        ss << "}";
        
        return ss.str();
    }
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

// RTSP 스트리밍 관련 전역 변수
GstRTSPServer* rtsp_server = nullptr;
GstRTSPMountPoints* mounts = nullptr;
GMainLoop* main_loop = nullptr;
std::thread rtsp_thread;
std::atomic<bool> rtsp_running{false};

// 메타데이터 큐 (스레드 안전)
std::queue<RTSPMetadata> metadata_queue;
std::mutex metadata_mutex;
std::condition_variable metadata_cv;
std::atomic<bool> metadata_thread_running{false};

// DB 초기화 함수
bool initializeDatabase() {
    try {
        SQLite::Database db(DB_FILE, SQLite::OPEN_READWRITE | SQLite::OPEN_CREATE);
        
        // lines 테이블 생성
        db.exec("CREATE TABLE IF NOT EXISTS lines ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "x1 INTEGER, "
                "y1 INTEGER, "
                "x2 INTEGER, "
                "y2 INTEGER, "
                "name TEXT, "
                "mode TEXT"
                ")");
        
        // baseLines 테이블 생성
        db.exec("CREATE TABLE IF NOT EXISTS baseLines ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "matrixNum1 INTEGER, "
                "x1 INTEGER, "
                "y1 INTEGER, "
                "matrixNum2 INTEGER, "
                "x2 INTEGER, "
                "y2 INTEGER"
                ")");
        
        // 기존 데이터 확인
        SQLite::Statement count_lines(db, "SELECT COUNT(*) FROM lines");
        SQLite::Statement count_baselines(db, "SELECT COUNT(*) FROM baseLines");
        
        bool has_lines = false, has_baselines = false;
        
        if (count_lines.executeStep()) {
            has_lines = count_lines.getColumn(0).getInt() > 0;
        }
        
        if (count_baselines.executeStep()) {
            has_baselines = count_baselines.getColumn(0).getInt() > 0;
        }
        
        // 샘플 데이터 삽입 (데이터가 없을 경우에만)
        if (!has_lines) {
            std::cout << "[DB] 기본 라인 데이터 삽입 중..." << std::endl;
            db.exec("INSERT INTO lines (x1, y1, x2, y2, name, mode) VALUES "
                   "(100, 150, 540, 150, 'Zone1', 'BothDirections'), "
                   "(100, 240, 540, 240, 'Zone2', 'BothDirections'), "
                   "(100, 330, 540, 330, 'Zone3', 'BothDirections')");
        }
        
        if (!has_baselines) {
            std::cout << "[DB] 기본 베이스라인 데이터 삽입 중..." << std::endl;
            db.exec("INSERT INTO baseLines (matrixNum1, x1, y1, matrixNum2, x2, y2) VALUES "
                   "(1, 100, 200, 2, 500, 200), "
                   "(3, 100, 400, 4, 500, 400)");
        }
        
        std::cout << "[DB] 데이터베이스 초기화 완료: " << DB_FILE << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] 데이터베이스 초기화 실패: " << e.what() << std::endl;
        return false;
    }
}

    // 서버에서 라인 정보 가져오기 (간단한 HTTP 클라이언트 또는 DB 직접 접근)
bool loadLineConfigsFromServer() {
    try {
        std::cout << "[loadLineConfigsFromServer] Starting to load line configs..." << std::endl;
        
        // DB 파일 존재 확인
        if (!std::filesystem::exists(DB_FILE)) {
            std::cerr << "[ERROR] Database file does not exist: " << DB_FILE << std::endl;
            return false;
        }
        
        SQLite::Database db(DB_FILE, SQLite::OPEN_READONLY);
        
        std::lock_guard<std::recursive_mutex> lock(data_mutex);
        rule_lines.clear();
        
        // lines 테이블에서 데이터 로드 (서버와 동일한 방식)
        const float scale_x = 3840.0f / 960.0f;
        const float scale_y = 2160.0f / 540.0f;
        
        SQLite::Statement query(db, "SELECT x1, y1, x2, y2, name, mode FROM lines LIMIT 8");

        int line_count = 0;
        while (query.executeStep()) {
            try {
                Line line;
                line.start = { query.getColumn(0).getInt() * scale_x, query.getColumn(1).getInt() * scale_y };
                line.end   = { query.getColumn(2).getInt() * scale_x, query.getColumn(3).getInt() * scale_y };
                line.name = query.getColumn(4).getString();
                line.mode = query.getColumn(5).getString();

                rule_lines[line.name] = line;
                line_count++;

                if (line_count <= 3) { // 처음 3개만 로그 출력
                    std::cout << "[DEBUG] Loaded line from server: " << line.name
                              << " [(" << line.start.x << "," << line.start.y << ") -> ("
                              << line.end.x << "," << line.end.y << ")] Mode: " << line.mode << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Failed to process line row: " << e.what() << std::endl;
                continue; // 개별 라인 처리 실패 시 계속 진행
            }
        }
        
        std::cout << "[INFO] Successfully loaded " << rule_lines.size() << " lines from server." << std::endl;
        return rule_lines.size() > 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load line configs from server: " << e.what() << std::endl;
        return false;
    }
}

// 서버에서 base line pairs 가져오기
bool loadBaseLinePairsFromServer() {
    try {
        std::cout << "[loadBaseLinePairsFromServer] Starting to load base line pairs..." << std::endl;
        
        // DB 파일 존재 확인
        if (!std::filesystem::exists(DB_FILE)) {
            std::cerr << "[ERROR] Database file does not exist: " << DB_FILE << std::endl;
            return false;
        }
        
        SQLite::Database db(DB_FILE, SQLite::OPEN_READONLY);
        
        std::lock_guard<std::recursive_mutex> lock(data_mutex);
        base_line_pairs.clear();
        
        const float scale_x = 3840.0f / 960.0f;
        const float scale_y = 2160.0f / 540.0f;
        
        SQLite::Statement query(db, "SELECT matrixNum1, x1, y1, matrixNum2, x2, y2 FROM baseLines");

        int pair_count = 0;
        while (query.executeStep()) {
            try {
                int id1 = query.getColumn(0).getInt();
                Point p1 = {query.getColumn(1).getInt() * scale_x, query.getColumn(2).getInt() * scale_y};
                int id2 = query.getColumn(3).getInt();
                Point p2 = {query.getColumn(4).getInt() * scale_x, query.getColumn(5).getInt() * scale_y};

                base_line_pairs.emplace_back(id1, p1, id2, p2);
                pair_count++;

                if (pair_count <= 2) { // 처음 2개만 로그 출력
                    std::cout << "[DEBUG] Loaded base line pair: " << id1 << "<->" << id2
                              << " (" << p1.x << "," << p1.y << ") <-> (" << p2.x << "," << p2.y << ")" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Failed to process base line pair row: " << e.what() << std::endl;
                continue;
            }
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
        return base_line_pairs.size() > 0;
        
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

// main_control.cpp의 analyze_risk_and_alert 로직을 이식한 함수 (RTSP 메타데이터용 수정)
std::string analyze_risk_and_alert_edge(int human_id, const std::string& rule_name) {
    std::lock_guard<std::recursive_mutex> lock(data_mutex);
    std::stringstream alert_ss;

    std::cout << "[DEBUG] Analyzing risk for human_id: " << human_id << " crossing line: " << rule_name << std::endl;

    // 1. 이벤트 라인 정보 확인
    if (rule_lines.find(rule_name) == rule_lines.end()) {
        std::cout << "[DEBUG] Step Failed: RuleName '" << rule_name << "' not found in predefined lines." << std::endl;
        return "";
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
        return "";
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
        int board_id = -1;
        float min_dist_sq = std::numeric_limits<float>::max();

        for (const auto& [id1, p1, id2, p2] : base_line_pairs) {
            float d1 = pow(oldest_pos.x - p1.x, 2) + pow(oldest_pos.y - p1.y, 2);
            float d2 = pow(oldest_pos.x - p2.x, 2) + pow(oldest_pos.y - p2.y, 2);

            if (d1 < min_dist_sq) {
                min_dist_sq = d1;
                closest_dot = p1;
                board_id = id2;
            }

            if (d2 < min_dist_sq) {
                min_dist_sq = d2;
                closest_dot = p2;
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
            
            alert_ss << "ALERT: Vehicle " << vehicle_id << " approaching human " << human_id 
                     << " from side. Matrix " << board_id << " should be activated. "
                     << "Cosine similarity: " << similarity << "; ";
            
        } else {
            std::cout << "[DEBUG] Step Failed (Vehicle " << vehicle_id << "): Cosine similarity not high enough." << std::endl;
        }
    }
    
    return alert_ss.str();
}

// RTSP 서버 설정
static GstAppSrc *current_appsrc = nullptr;
static std::mutex appsrc_mutex;
static std::atomic<bool> appsrc_available{false};

static void need_data(GstElement* appsrc, guint unused, gpointer user_data) {
    // appsrc가 더 많은 데이터를 요구할 때 호출
    // std::cout << "[RTSP] Need data callback" << std::endl;
}

static void enough_data(GstElement* appsrc, gpointer user_data) {
    // appsrc 버퍼가 가득 찼을 때 호출
    // std::cout << "[RTSP] Enough data callback" << std::endl;
}

static void media_prepare(GstRTSPMedia *media, gpointer user_data) {
    std::cout << "[RTSP] Media prepare callback" << std::endl;
}

static void media_unprepared(GstRTSPMedia *media, gpointer user_data) {
    std::cout << "[RTSP] Media unprepared callback" << std::endl;
    std::lock_guard<std::mutex> lock(appsrc_mutex);
    current_appsrc = nullptr;
    appsrc_available = false;
}

// 미디어 준비 콜백 함수 - appsrc 찾기
static void media_prepared_callback(GstRTSPMedia *media, gpointer user_data) {
    std::cout << "[RTSP] Media prepared - finding appsrc element" << std::endl;
    
    // appsrc 찾기 (prepared 상태에서만 가능)
    GstElement *pipeline = gst_rtsp_media_get_element(media);
    if (pipeline) {
        // 먼저 직접 이름으로 찾기
        GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysrc");
        
        // 찾지 못한 경우 더 다양한 방법으로 시도
        if (!appsrc) {
            std::cout << "[RTSP] Could not find appsrc by name 'mysrc', trying alternative methods..." << std::endl;
            
            // 재귀적으로 검색
            appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(pipeline), "mysrc");
            
            // 여전히 못 찾았다면 타입으로 검색 (모든 appsrc 타입의 요소 찾기)
            if (!appsrc) {
                std::cout << "[RTSP] Still not found, searching by element type..." << std::endl;
                
                // 파이프라인에서 모든 요소 반복하며 검색
                GstIterator *it = gst_bin_iterate_recurse(GST_BIN(pipeline));
                GValue item = G_VALUE_INIT;
                gboolean done = FALSE;
                
                while (!done) {
                    switch (gst_iterator_next(it, &item)) {
                        case GST_ITERATOR_OK: {
                            GstElement *element = GST_ELEMENT(g_value_get_object(&item));
                            // 요소 타입 검사
                            if (GST_IS_APP_SRC(element)) {
                                std::cout << "[RTSP] Found element of type GstAppSrc: " 
                                          << GST_ELEMENT_NAME(element) << std::endl;
                                if (!appsrc) {
                                    appsrc = GST_ELEMENT_CAST(gst_object_ref(element));
                                }
                            }
                            g_value_reset(&item);
                            break;
                        }
                        case GST_ITERATOR_RESYNC:
                            gst_iterator_resync(it);
                            break;
                        case GST_ITERATOR_ERROR:
                        case GST_ITERATOR_DONE:
                            done = TRUE;
                            break;
                    }
                }
                g_value_unset(&item);
                gst_iterator_free(it);
            }
        }
            
        if (appsrc) {
            {
                std::lock_guard<std::mutex> lock(appsrc_mutex);
                current_appsrc = GST_APP_SRC(appsrc);
                appsrc_available = true;
                std::cout << "[RTSP] Found and configured appsrc: " << appsrc << std::endl;
            }
            
            // appsrc 속성 설정 (최적화)
            g_object_set(G_OBJECT(appsrc),
                        "is-live", TRUE,
                        "format", GST_FORMAT_TIME,
                        "do-timestamp", TRUE,
                        "min-latency", 0,
                        "max-latency", 100000000,  // 100ms
                        "block", FALSE,
                        "max-bytes", 1000000,      // 1MB 버퍼
                        "emit-signals", TRUE,
                        NULL);
            
            // appsrc 콜백 연결
            g_signal_connect(appsrc, "need-data", G_CALLBACK(need_data), NULL);
            g_signal_connect(appsrc, "enough-data", G_CALLBACK(enough_data), NULL);
            
            // 중요: appsrc를 해제하지 않음 - 우리가 참조 중인 객체이므로
            // gst_object_unref(appsrc);
        } else {
            std::cerr << "[ERROR] Could not find any appsrc element in the pipeline" << std::endl;
        }
        
        gst_object_unref(pipeline);
    } else {
        std::cerr << "[ERROR] Could not get media pipeline" << std::endl;
    }
}

// 미디어 구성 콜백
static void media_configure(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data) {
    std::cout << "[RTSP] Media configured" << std::endl;
    
    // 미디어 언프리페어 콜백 연결
    g_signal_connect(media, "unprepared", G_CALLBACK(media_unprepared), NULL);
    
    // prepared 상태에서만 appsrc를 찾을 수 있으므로, prepared 신호 연결 (미디어 준비됨 콜백)
    g_signal_connect(media, "prepared", G_CALLBACK(media_prepared_callback), NULL);
}

// RTSP 메타데이터 전송 스레드
void rtsp_metadata_thread() {
    std::cout << "[RTSP] Metadata thread started" << std::endl;
    
    int consecutive_errors = 0;
    const int max_consecutive_errors = 5;
    
    while (metadata_thread_running) {
        try {
            std::unique_lock<std::mutex> lock(metadata_mutex);
            
            // 타임아웃과 함께 대기 (최대 2초)
            bool has_data = metadata_cv.wait_for(lock, std::chrono::seconds(2), 
                [] { return !metadata_queue.empty() || !metadata_thread_running; });
            
            if (!metadata_thread_running) {
                std::cout << "[RTSP] Metadata thread shutdown signal received" << std::endl;
                break;
            }
            
            if (has_data && !metadata_queue.empty()) {
                RTSPMetadata metadata = std::move(metadata_queue.front());
                metadata_queue.pop();
                lock.unlock();
                
                try {
                    // 메타데이터를 JSON으로 직렬화하여 로그 출력
                    std::string json_metadata = metadata.toJson();
                    std::cout << "[RTSP_METADATA] " << json_metadata << std::endl;
                    
                    // 연속 오류 카운터 리셋
                    consecutive_errors = 0;
                    
                } catch (const std::exception& e) {
                    std::cerr << "[ERROR] Failed to serialize metadata: " << e.what() << std::endl;
                    consecutive_errors++;
                }
                
                lock.lock();
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception in metadata thread: " << e.what() << std::endl;
            consecutive_errors++;
            
            // 연속 오류가 너무 많으면 잠시 대기
            if (consecutive_errors >= max_consecutive_errors) {
                std::cerr << "[WARNING] Too many consecutive errors, sleeping..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                consecutive_errors = 0;
            }
        }
    }
    
    std::cout << "[RTSP] Metadata thread stopped" << std::endl;
}

// RTSP 서버 초기화
bool init_rtsp_server() {
    std::cout << "[RTSP] Initializing RTSP server on port " << RTSP_PORT << std::endl;
    
    // GStreamer 초기화 - argc, argv를 nullptr로 설정
    int argc = 0;
    char **argv = nullptr;
    GError *error = nullptr;
    
    if (!gst_init_check(&argc, &argv, &error)) {
        std::cerr << "[ERROR] Failed to initialize GStreamer: " 
                  << (error ? error->message : "Unknown error") << std::endl;
        if (error) g_error_free(error);
        return false;
    }
    
    std::cout << "[RTSP] GStreamer initialized successfully" << std::endl;
    
    rtsp_server = gst_rtsp_server_new();
    if (!rtsp_server) {
        std::cerr << "[ERROR] Failed to create RTSP server" << std::endl;
        return false;
    }
    
    // 서버 포트 설정
    std::cout << "[RTSP] Setting server port to: " << RTSP_PORT << std::endl;
    gst_rtsp_server_set_service(rtsp_server, std::to_string(RTSP_PORT).c_str());
    
    // 서버 바인드 주소 설정 (모든 인터페이스에서 수신)
    gst_rtsp_server_set_address(rtsp_server, "0.0.0.0");
    
    mounts = gst_rtsp_server_get_mount_points(rtsp_server);
    if (!mounts) {
        std::cerr << "[ERROR] Failed to get mount points" << std::endl;
        return false;
    }
    
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();
    if (!factory) {
        std::cerr << "[ERROR] Failed to create media factory" << std::endl;
        return false;
    }        // 최적화된 H.264 하드웨어 인코더 파이프라인
    std::string pipeline_hw = 
        "( appsrc name=mysrc is-live=true format=time do-timestamp=true "
        "caps=\"video/x-raw,format=RGB,width=" + std::to_string(STREAM_WIDTH) + 
        ",height=" + std::to_string(STREAM_HEIGHT) + ",framerate=" + std::to_string(TARGET_FPS) + "/1\" ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "videoconvert ! "
        "video/x-raw,format=NV12 ! "  // NV12 포맷 사용 (하드웨어 가속에 최적)
        "v4l2h264enc extra-controls=\"encode,video_bitrate=4000000,h264_profile=1,h264_level=13\" ! "
        "video/x-h264,profile=baseline ! "
        "h264parse ! "
        "rtph264pay config-interval=1 name=pay0 pt=96 )";
    
    // 최적화된 소프트웨어 인코더 폴백 파이프라인 (더 빠른 설정)
    std::string pipeline_sw = 
        "( appsrc name=mysrc is-live=true format=time do-timestamp=true stream-type=0 "
        "caps=\"video/x-raw,format=RGB,width=" + std::to_string(STREAM_WIDTH) + 
        ",height=" + std::to_string(STREAM_HEIGHT) + ",framerate=" + std::to_string(TARGET_FPS) + "/1\" ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency bitrate=4000 speed-preset=veryfast profile=baseline key-int-max=30 threads=4 ! "
        "h264parse ! "
        "rtph264pay config-interval=1 name=pay0 pt=96 )";
    
    // 가장 간단한 폴백 파이프라인 (최고 호환성)
    std::string pipeline_fallback = 
        "( appsrc name=mysrc is-live=true format=time do-timestamp=true stream-type=0 "
        "caps=\"video/x-raw,format=RGB,width=" + std::to_string(STREAM_WIDTH) + 
        ",height=" + std::to_string(STREAM_HEIGHT) + ",framerate=" + std::to_string(TARGET_FPS) + "/1\" ! "
        "queue max-size-buffers=2 ! "
        "videoconvert ! "
        "x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast key-int-max=15 ! "
        "h264parse config-interval=1 ! "
        "rtph264pay name=pay0 pt=96 )";
    
    std::cout << "[RTSP] Trying hardware H264 encoder first..." << std::endl;
    
    // 파이프라인 테스트 함수 - 더 자세한 진단 정보 포함
    auto test_pipeline = [](const std::string& pipeline_str, const std::string& name) -> bool {
        std::cout << "[RTSP] Testing pipeline: " << name << std::endl;
        std::cout << "[RTSP] Pipeline string: " << pipeline_str << std::endl;
        
        GError *test_error = nullptr;
        GstElement *test_element = gst_parse_launch(pipeline_str.c_str(), &test_error);
        
        if (!test_element || test_error) {
            if (test_error) {
                std::cerr << "[RTSP] " << name << " test failed: " << test_error->message << std::endl;
                g_error_free(test_error);
            } else {
                std::cerr << "[RTSP] " << name << " test failed: Unknown error (null element)" << std::endl;
            }
            if (test_element) gst_object_unref(test_element);
            return false;
        } else {
            // 파이프라인 구조 검사
            std::cout << "[RTSP] Pipeline structure:" << std::endl;
            
            // appsrc 요소 검색
            GstElement *appsrc = gst_bin_get_by_name(GST_BIN(test_element), "mysrc");
            std::cout << "[RTSP] - appsrc 'mysrc' found: " << (appsrc ? "YES" : "NO") << std::endl;
            if (appsrc) {
                gst_object_unref(appsrc);
            }
            
            // 파이프라인 구조 검증
            std::cout << "[RTSP] - Testing pipeline state change to NULL->READY" << std::endl;
            GstStateChangeReturn ret = gst_element_set_state(test_element, GST_STATE_READY);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                std::cerr << "[RTSP] - Pipeline failed to change state to READY" << std::endl;
                gst_object_unref(test_element);
                return false;
            }
            
            // 다시 NULL 상태로
            gst_element_set_state(test_element, GST_STATE_NULL);
            gst_object_unref(test_element);
            std::cout << "[RTSP] " << name << " test successful" << std::endl;
            return true;
        }
    };
    
    // 파이프라인 선택 로직
    std::string selected_pipeline;
    std::string selected_name;
    
    if (test_pipeline(pipeline_hw, "Hardware H264 encoder")) {
        selected_pipeline = pipeline_hw;
        selected_name = "Hardware v4l2h264enc encoder";
    } else if (test_pipeline(pipeline_sw, "Optimized software encoder")) {
        selected_pipeline = pipeline_sw;
        selected_name = "Optimized x264 encoder";
    } else if (test_pipeline(pipeline_fallback, "Fallback encoder")) {
        selected_pipeline = pipeline_fallback;
        selected_name = "Fallback x264 encoder";
    } else {
        std::cerr << "[ERROR] All pipeline tests failed!" << std::endl;
        return false;
    }
    
    gst_rtsp_media_factory_set_launch(factory, selected_pipeline.c_str());
    std::cout << "[RTSP] Using: " << selected_name << std::endl;
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_media_factory_set_eos_shutdown(factory, FALSE);
    gst_rtsp_media_factory_set_stop_on_disconnect(factory, FALSE);
    
    g_signal_connect(factory, "media-configure", (GCallback)media_configure, nullptr);
    g_signal_connect(rtsp_server, "client-connected", G_CALLBACK(
        // 클라이언트 연결 콜백 함수
        +[](GstRTSPServer *server, GstRTSPClient *client, gpointer user_data) {
            // IP 주소 가져오기 (안전하게)
            std::string host = "unknown";
            if (GstRTSPConnection* conn = gst_rtsp_client_get_connection(client)) {
                const gchar* ip = gst_rtsp_connection_get_ip(conn);
                if (ip) host = ip;
            }
            std::cout << "[RTSP] Client connected from IP: " << host << ", client: " << client << std::endl;
            
            // 클라이언트가 연결되면 스트리밍 가능 상태인지 로그에 기록
            std::lock_guard<std::mutex> lock(appsrc_mutex);
            std::cout << "[RTSP] Stream ready status: appsrc=" << current_appsrc << ", available=" 
                      << (appsrc_available ? "yes" : "no") << std::endl;
        }
    ), nullptr);
    
    g_signal_connect(rtsp_server, "client-disconnected", G_CALLBACK(
        // 클라이언트 연결 해제 콜백 함수
        +[](GstRTSPServer *server, GstRTSPClient *client, gpointer user_data) {
            std::cout << "[RTSP] Client disconnected: " << client << std::endl;
        }
    ), nullptr);
    
    gst_rtsp_mount_points_add_factory(mounts, RTSP_PATH.c_str(), factory);
    g_object_unref(mounts);
    
    // GMainLoop를 먼저 생성
    main_loop = g_main_loop_new(nullptr, FALSE);
    if (!main_loop) {
        std::cerr << "[ERROR] Failed to create GMainLoop" << std::endl;
        return false;
    }
    
    // Default context를 사용하여 RTSP 서버 연결 시도
    std::cout << "[RTSP] Attempting to attach RTSP server..." << std::endl;
    guint server_id = gst_rtsp_server_attach(rtsp_server, nullptr);
    
    std::cout << "[RTSP] Server attach returned ID: " << server_id << std::endl;
    
    if (server_id == 0) {
        std::cerr << "[ERROR] Failed to attach RTSP server to main context" << std::endl;
        g_main_loop_unref(main_loop);
        main_loop = nullptr;
        return false;
    }
    
    std::cout << "[RTSP] RTSP server attached successfully with ID: " << server_id << std::endl;
    std::cout << "[RTSP] RTSP server initialized. Stream will be available at rtsp://localhost:" 
              << RTSP_PORT << RTSP_PATH << std::endl;
    
    // 메타데이터 스레드 시작
    metadata_thread_running = true;
    std::thread metadata_thread(rtsp_metadata_thread);
    metadata_thread.detach();
    
    return true;
}

// RTSP 서버 정리
void cleanup_rtsp_server() {
    static std::atomic<bool> cleaned_up{false};
    
    // 중복 정리 방지
    bool expected = false;
    if (!cleaned_up.compare_exchange_strong(expected, true)) {
        return;  // 이미 정리됨
    }
    
    std::cout << "[RTSP] Cleaning up RTSP server" << std::endl;
    
    // 메타데이터 스레드 정지
    metadata_thread_running = false;
    metadata_cv.notify_all();
    
    // GStreamer 메인 루프 정지
    if (main_loop && g_main_loop_is_running(main_loop)) {
        std::cout << "[RTSP] Stopping GStreamer main loop" << std::endl;
        g_main_loop_quit(main_loop);
    }
    
    rtsp_running = false;
    
    // 약간의 대기 시간을 줘서 스레드가 정리될 시간을 확보
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // RTSP 서버 리소스 정리
    if (rtsp_server) {
        std::cout << "[RTSP] Unreferencing RTSP server" << std::endl;
        g_object_unref(rtsp_server);
        rtsp_server = nullptr;
    }
    
    if (main_loop) {
        std::cout << "[RTSP] Unreferencing main loop" << std::endl;
        g_main_loop_unref(main_loop);
        main_loop = nullptr;
    }
    
    std::cout << "[RTSP] RTSP server cleanup completed" << std::endl;
}

// RTSP 메인 루프 스레드
void rtsp_main_loop_thread() {
    std::cout << "[RTSP] Starting GStreamer main loop" << std::endl;
    
    if (!main_loop) {
        std::cerr << "[ERROR] Main loop not initialized" << std::endl;
        return;
    }
    
    rtsp_running = true;
    g_main_loop_run(main_loop);
    std::cout << "[RTSP] GStreamer main loop stopped" << std::endl;
}

// Line crossing 계산 및 위험 분석이 통합된 LineCrossingDetector (RTSP 버전)
class RTSPLineCrossingDetector {
private:
    std::vector<LineCrossingZone> zones;
    std::unordered_map<int, std::unordered_map<std::string, float>> prev_positions;
    std::vector<CrossingEvent> recent_crossings;
    std::mutex zones_mutex;
    std::chrono::steady_clock::time_point last_config_load;
    
public:
    RTSPLineCrossingDetector() {
        last_config_load = std::chrono::steady_clock::now();
        loadConfigurations();
    }
    
    // 서버에서 설정 로드
    void loadConfigurations() {
        std::cout << "[RTSPLineCrossingDetector] Loading configurations from server..." << std::endl;
        
        // 서버에서 라인 설정과 base line pairs 로드
        bool lines_ok = loadLineConfigsFromServer();
        bool base_ok = loadBaseLinePairsFromServer();
        
        if (lines_ok && base_ok) {
            std::cout << "[RTSPLineCrossingDetector] Server configurations loaded successfully." << std::endl;
            
            // rule_lines를 zones로 변환
            std::lock_guard<std::mutex> lock(zones_mutex);
            zones.clear();
            
            for (const auto& [name, line] : rule_lines) {
                LineCrossingZone zone(line.start, line.end, name, line.mode);
                zones.push_back(zone);
                std::cout << "[DEBUG] Added zone from server: " << zone.name << std::endl;
            }
        } else {
            std::cout << "[RTSPLineCrossingDetector] Failed to load from server, using fallback zones..." << std::endl;
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
        
        std::cout << "[RTSPLineCrossingDetector] Initialized " << zones.size() << " fallback zones." << std::endl;
    }
    
    // 주기적으로 서버 설정 업데이트 확인 (메타데이터 처리에 영향 주지 않도록 개선)
    void checkForUpdates() {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_config_load).count() >= 60) { // 60초로 변경
            // 별도 스레드에서 비동기적으로 설정 업데이트 수행
            std::thread update_thread([this, now]() {
                try {
                    std::cout << "[RTSPLineCrossingDetector] Checking for configuration updates..." << std::endl;
                    
                    // 데이터베이스 연결 테스트
                    bool db_available = false;
                    try {
                        SQLite::Database test_db(DB_FILE, SQLite::OPEN_READONLY);
                        db_available = true;
                    } catch (const std::exception& e) {
                        std::cout << "[RTSPLineCrossingDetector] Database not available, using current configuration" << std::endl;
                        db_available = false;
                    }
                    
                    if (db_available) {
                        loadConfigurations();
                        std::cout << "[RTSPLineCrossingDetector] Configuration update completed." << std::endl;
                    }
                    
                    last_config_load = now;
                } catch (const std::exception& e) {
                    std::cerr << "[ERROR] Configuration update failed: " << e.what() << std::endl;
                    // 업데이트 실패 시 다음 업데이트를 10초 후에 재시도
                    last_config_load = now - std::chrono::seconds(50);
                }
            });
            update_thread.detach(); // 비동기적으로 실행
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
    
    // Line crossing 검사 및 위험 분석 통합 (RTSP 메타데이터 생성)
    std::vector<CrossingEvent> checkCrossingsWithRiskAnalysis(const std::vector<Track>& tracks, RTSPMetadata& metadata) {
        std::lock_guard<std::mutex> lock(zones_mutex);
        
        // 주기적으로 서버 설정 업데이트 확인
        checkForUpdates();
        
        // 차량 위치 업데이트
        updateVehiclePositions(tracks);
        
        std::vector<CrossingEvent> new_crossings;
        std::string combined_alerts;
        
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
                        std::string alert = analyze_risk_and_alert_edge(track.id, zone.name);
                        if (!alert.empty()) {
                            combined_alerts += alert;
                        }
                    }
                }
                
                prev_positions[track.id][zone.name] = position;
            }
        }
        
        // 메타데이터에 위험 경고 추가
        metadata.risk_alerts = combined_alerts;
        metadata.crossings = new_crossings;
        
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

// RTSP용 OpenVINO YOLOv5 inference class
class RTSPOpenVINOYOLOv5TrackerWithRiskAnalysis {
public:
    RTSPOpenVINOYOLOv5TrackerWithRiskAnalysis(const std::string& model_xml, const std::string& device = "CPU") 
        : frame_counter(0), sort_tracker(5, 2, 0.3f) {
        core = std::make_shared<ov::Core>();
        model = core->read_model(model_xml);
        compiled_model = core->compile_model(model, device);
        infer_request = compiled_model.create_infer_request();
        
        std::cout << "RTSP OpenVINO YOLOv5 + SORT + Risk Analysis 초기화 완료" << std::endl;
        std::cout << "입력 크기: " << input_width << "x" << input_height << std::endl;
    }

    std::vector<Detection> detections;
    std::vector<Track> tracks;
    RTSPLineCrossingDetector rtsp_detector;
    int frame_counter;

    RTSPMetadata inferTrackAndAnalyzeForRTSP(const cv::Mat& frame, double fps) {
        auto start_time = std::chrono::steady_clock::now();
        
        RTSPMetadata metadata;
        metadata.frame_number = ++frame_counter;
        metadata.timestamp = start_time;
        metadata.fps = fps;
        
        // 1. YOLOv5 추론
        performInference(frame);
        metadata.detections = detections;
        metadata.detection_count = detections.size();
        
        auto inference_time = std::chrono::steady_clock::now();
        
        // 2. SORT 트래킹 수행
        performTracking();
        metadata.tracks = tracks;
        metadata.tracking_count = tracks.size();
        
        auto tracking_time = std::chrono::steady_clock::now();
        
        // 3. Line crossing 검사 및 위험 분석 (main_control.cpp 로직 포함)
        auto crossings = rtsp_detector.checkCrossingsWithRiskAnalysis(tracks, metadata);
        rtsp_detector.clearOldTracks(tracks);
        
        auto analysis_time = std::chrono::steady_clock::now();

        // 시간 측정 결과
        auto inference_ms = std::chrono::duration_cast<std::chrono::milliseconds>(inference_time - start_time).count();
        auto tracking_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tracking_time - inference_time).count();
        auto analysis_ms = std::chrono::duration_cast<std::chrono::milliseconds>(analysis_time - tracking_time).count();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(analysis_time - start_time).count();

        // 결과 출력
        if (!tracks.empty() || !crossings.empty()) {
            std::cout << "=== RTSP YOLOv5 + SORT + Risk Analysis 결과 ===" << std::endl;
            std::cout << "프레임 #" << frame_counter << " - 감지: " << detections.size() << ", 추적: " << tracks.size() 
                      << ", 신규 crossing: " << crossings.size() << ", FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;
            std::cout << "처리 시간 - 추론: " << inference_ms << "ms, 트래킹: " << tracking_ms 
                      << "ms, 위험분석: " << analysis_ms << "ms, 총: " << total_ms << "ms" << std::endl;
            
            for (const auto& track : tracks) {
                std::cout << "  - ID: " << track.id << ", person (신뢰도: " << std::fixed << std::setprecision(2) << track.confidence 
                         << ", 위치: " << track.bbox.x << "," << track.bbox.y << "," << track.bbox.width << "," << track.bbox.height << ")" << std::endl;
            }
            
            if (!crossings.empty()) {
                std::cout << "🚨 새로운 Line Crossing 이벤트 및 위험 분석 완료: " << crossings.size() << "개" << std::endl;
            }
            
            if (!metadata.risk_alerts.empty()) {
                std::cout << "⚠️  위험 경고: " << metadata.risk_alerts << std::endl;
            }
            std::cout << "=========================================" << std::endl;
        }
        
        return metadata;
    }

    const std::vector<Detection>& getDetections() const { return detections; }
    const std::vector<Track>& getTracks() const { return tracks; }
    const RTSPLineCrossingDetector& getRTSPDetector() const { return rtsp_detector; }

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

class RTSPZeroCopyOpenVINOTracker {
    // libcamera 및 버퍼 관련 멤버
    std::shared_ptr<libcamera::Camera> camera;
    std::unique_ptr<libcamera::CameraManager> cameraManager;
    std::unique_ptr<libcamera::CameraConfiguration> config;
    libcamera::Stream* stream;
    std::shared_ptr<libcamera::FrameBufferAllocator> allocator;
    std::vector<std::vector<void*>> bufferPlaneMappings;
    std::vector<std::vector<size_t>> bufferPlaneSizes;
    std::atomic<bool> stopping{false};

    std::unique_ptr<RTSPOpenVINOYOLOv5TrackerWithRiskAnalysis> rtsp_tracker;

    // FPS 측정을 위한 변수들
    std::chrono::steady_clock::time_point lastTime;
    int frameCounter = 0;
    double fps = 0.0;

    // GStreamer appsrc 관련
    GstElement *appsrc = nullptr;

public:
    RTSPZeroCopyOpenVINOTracker(const std::string& model_xml) {
        rtsp_tracker = std::make_unique<RTSPOpenVINOYOLOv5TrackerWithRiskAnalysis>(model_xml);
        lastTime = std::chrono::steady_clock::now();
    }

    bool initialize() {
        std::cout << "RTSP 카메라 초기화 중..." << std::endl;
        cameraManager = std::make_unique<libcamera::CameraManager>();
        if (cameraManager->start()) {
            std::cerr << "카메라 매니저 시작 실패" << std::endl;
            return false;
        }

        auto cameras = cameraManager->cameras();
        if (cameras.empty()) {
            std::cerr << "카메라를 찾을 수 없습니다" << std::endl;
            return false;
        }

        camera = cameras[0];
        if (camera->acquire()) {
            std::cerr << "카메라 획득 실패" << std::endl;
            return false;
        }

        config = camera->generateConfiguration({libcamera::StreamRole::Viewfinder});
        auto& streamConfig = config->at(0);
        streamConfig.size = libcamera::Size(STREAM_WIDTH, STREAM_HEIGHT);
        streamConfig.pixelFormat = libcamera::formats::RGB888;
        streamConfig.bufferCount = 4;

        config->validate();
        if (camera->configure(config.get())) {
            std::cerr << "카메라 설정 실패" << std::endl;
            return false;
        }

        // 30fps 설정을 위한 카메라 컨트롤
        libcamera::ControlList controls;
        // 프레임 지속시간을 33.33ms로 설정 (30fps = 1/30 = 0.0333... 초)
        controls.set(libcamera::controls::FrameDurationLimits, 
                     libcamera::Span<const int64_t, 2>({33333, 33333}));  // 마이크로초 단위
        
        std::cout << "[CAMERA] Setting 30fps frame rate limit" << std::endl;

        stream = streamConfig.stream();
        allocator = std::make_shared<libcamera::FrameBufferAllocator>(camera);

        std::cout << "RTSP 카메라 초기화 완료" << std::endl;
        return true;
    }

    bool setupBuffers() {
        if (allocator->allocate(stream) < 0) {
            std::cerr << "버퍼 할당 실패" << std::endl;
            return false;
        }

        size_t bufferCount = allocator->buffers(stream).size();
        std::cout << "할당된 버퍼 수: " << bufferCount << std::endl;

        bufferPlaneMappings.resize(bufferCount);
        bufferPlaneSizes.resize(bufferCount);

        for (size_t i = 0; i < bufferCount; ++i) {
            auto buffer = allocator->buffers(stream)[i].get();
            const auto& planes = buffer->planes();

            std::cout << "버퍼 " << i << " - " << planes.size() << " 플레인" << std::endl;

            for (size_t j = 0; j < planes.size(); ++j) {
                const libcamera::FrameBuffer::Plane& plane = planes[j];
                size_t length = plane.length;

                void* memory = mmap(nullptr, length, PROT_READ | PROT_WRITE, 
                                  MAP_SHARED, plane.fd.get(), plane.offset);

                if (memory == MAP_FAILED) {
                    std::cerr << "mmap 실패: 버퍼 " << i << ", 플레인 " << j << std::endl;
                    return false;
                }

                bufferPlaneMappings[i].push_back(memory);
                bufferPlaneSizes[i].push_back(length);
                std::cout << "  플레인 " << j << ": " << length << " 바이트" << std::endl;
            }
        }

        std::cout << "버퍼 매핑 완료" << std::endl;
        return true;
    }

    void run() {
        std::cout << "카메라 스트리밍 시작..." << std::endl;

        std::vector<std::unique_ptr<libcamera::Request>> requests;
        for (auto& buffer : allocator->buffers(stream)) {
            auto request = camera->createRequest();
            request->addBuffer(stream, buffer.get());
            requests.push_back(std::move(request));
        }

        camera->requestCompleted.connect(this, &RTSPZeroCopyOpenVINOTracker::processRequest);

        // 30fps 카메라 컨트롤 적용
        libcamera::ControlList startControls;
        startControls.set(libcamera::controls::FrameDurationLimits, 
                         libcamera::Span<const int64_t, 2>({33333, 33333}));  // 30fps
        
        if (camera->start(&startControls)) {
            std::cerr << "카메라 시작 실패" << std::endl;
            return;
        }
        
        std::cout << "[CAMERA] Started with 30fps configuration" << std::endl;

        for (auto& request : requests) {
            camera->queueRequest(request.get());
        }

        std::cout << "카메라 스트리밍 중... (Ctrl+C로 종료)" << std::endl;
        
        while (!stopping.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        camera->stop();
        std::cout << "카메라 스트리밍 종료" << std::endl;
    }

    void stop() {
        std::cout << "카메라 중지 중..." << std::endl;
        stopping.store(true);
        
        if (camera) {
            camera->stop();
            camera->requestCompleted.disconnect(this, &RTSPZeroCopyOpenVINOTracker::processRequest);
        }
        
        std::cout << "카메라 중지 완료" << std::endl;
    }

    ~RTSPZeroCopyOpenVINOTracker() {
        // 리소스 정리
        for (size_t i = 0; i < bufferPlaneMappings.size(); ++i) {
            for (size_t j = 0; j < bufferPlaneMappings[i].size(); ++j) {
                munmap(bufferPlaneMappings[i][j], bufferPlaneSizes[i][j]);
            }
        }
    }

private:
    void processRequest(libcamera::Request* request) {
        if (stopping.load()) {
            // 정리하고 요청을 다시 큐에 넣지 않음
            std::cout << "[CAMERA] Request processing stopped" << std::endl;
            return;
        }

        frameCounter++;
        auto currentTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);

        if (duration.count() >= 1000) {
            fps = frameCounter * 1000.0 / duration.count();
            frameCounter = 0;
            lastTime = currentTime;
        }

        auto buffer = request->findBuffer(stream);
        if (!buffer) {
            if (!stopping.load()) {
                camera->queueRequest(request);
            }
            return;
        }

        size_t bufferIndex = 0;
        auto& buffers = allocator->buffers(stream);
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (buffers[i].get() == buffer) {
                bufferIndex = i;
                break;
            }
        }                // Zero-copy: 메모리 매핑된 데이터를 직접 OpenCV Mat으로 변환 (RGB888 포맷)
        uint8_t* data = static_cast<uint8_t*>(bufferPlaneMappings[bufferIndex][0]);
        
        // 중요: libcamera에서 RGB888 포맷으로 설정했으므로, CV_8UC3 데이터는 이미 RGB 순서임
        // OpenCV에서는 BGR을 기본으로 사용하지만, 여기서는 RGB 데이터를 그대로 사용
        cv::Mat frame(STREAM_HEIGHT, STREAM_WIDTH, CV_8UC3, data);

        try {
            // YOLOv5 추론 + 트래킹 + 위험 분석 수행 및 RTSP 메타데이터 생성
            RTSPMetadata metadata = rtsp_tracker->inferTrackAndAnalyzeForRTSP(frame, fps);

            // RTSP 메타데이터 큐에 추가 (스레드 안전)
            if (metadata_thread_running) {
                std::lock_guard<std::mutex> lock(metadata_mutex);
                metadata_queue.push(metadata);
                metadata_cv.notify_one();
            }

            // RTSP 스트림으로 프레임 전송
            sendFrameToRTSP(frame);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Frame processing failed: " << e.what() << std::endl;
        }

        // 요청 재사용 (stopping이 아닌 경우에만)
        if (!stopping.load()) {
            request->reuse(libcamera::Request::ReuseBuffers);
            camera->queueRequest(request);
        }
    }

    void sendFrameToRTSP(const cv::Mat& frame) {
        static int rtsp_frame_count = 0;
        static int rtsp_error_count = 0;
        static int rtsp_recovery_attempts = 0;
        static std::chrono::steady_clock::time_point last_log = std::chrono::steady_clock::now();
        
        rtsp_frame_count++;
        
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_log);
        
        // 로그 출력 및 appsrc 상태 확인
        if (duration.count() >= 5) { // 5초마다 로그
            // appsrc 상태 확인 및 로그
            bool is_appsrc_ready = false;
            GstAppSrc *app_src = nullptr;
            
            {
                std::lock_guard<std::mutex> lock(appsrc_mutex);
                is_appsrc_ready = (current_appsrc != nullptr && appsrc_available);
                app_src = current_appsrc;
            }
            
            if (app_src && !GST_IS_APP_SRC(app_src)) {
                std::cerr << "[RTSP] WARNING: appsrc is invalid but marked as available" << std::endl;
                std::lock_guard<std::mutex> lock(appsrc_mutex);
                current_appsrc = nullptr;
                appsrc_available = false;
                is_appsrc_ready = false;
            }
            
            std::cout << "[RTSP_STREAM] Sent " << rtsp_frame_count 
                      << " frames in last " << duration.count() << "s"
                      << " (" << frame.cols << "x" << frame.rows 
                      << ", avg FPS: " << std::fixed << std::setprecision(1) 
                      << (rtsp_frame_count / (double)duration.count()) << ")"
                      << ", RTSP ready: " << (is_appsrc_ready ? "YES" : "NO")
                      << ", Error count: " << rtsp_error_count
                      << ", Recovery attempts: " << rtsp_recovery_attempts << std::endl;
            rtsp_frame_count = 0;
            rtsp_error_count = 0;
            last_log = now;
        }
        
        std::lock_guard<std::mutex> lock(appsrc_mutex);
        
        // 기본 유효성 검사 (current_appsrc가 nullptr이 아니고, 사용 가능 상태인지)
        if (current_appsrc && appsrc_available) {
            // GstAppSrc 타입 검사
            if (!GST_IS_APP_SRC(current_appsrc)) {
                std::cerr << "[RTSP] ERROR: current_appsrc is not a valid GstAppSrc object" << std::endl;
                current_appsrc = nullptr;
                appsrc_available = false;
                return;
            }
            
            try {
                // Zero-copy에서는 이미 RGB 형식으로 데이터를 받으므로 변환이 필요 없음
                // frame은 이미 RGB 데이터를 담고 있으므로 직접 사용
                
                // GstBuffer 생성 및 데이터 복사
                gsize buffer_size = frame.total() * frame.elemSize();
                GstBuffer *buffer = gst_buffer_new_and_alloc(buffer_size);
                
                if (buffer) {
                    // 데이터 매핑 및 복사
                    GstMapInfo map;
                    if (gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
                        memcpy(map.data, frame.data, buffer_size);
                        gst_buffer_unmap(buffer, &map);
                        
                        // 30fps에 맞는 타임스탬프 설정
                        GstClockTime timestamp = gst_util_uint64_scale(rtsp_frame_count, GST_SECOND, TARGET_FPS);
                        GST_BUFFER_PTS(buffer) = timestamp;
                        GST_BUFFER_DTS(buffer) = timestamp;
                        GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, TARGET_FPS);
                        
                        // 버퍼 푸시 (비동기적으로)
                        GstFlowReturn ret = gst_app_src_push_buffer(current_appsrc, buffer);
                        if (ret != GST_FLOW_OK) {
                            rtsp_error_count++;
                            
                            // 최대 5초에 한 번만 상세한 오류 메시지를 출력
                            static std::chrono::steady_clock::time_point last_error_log = std::chrono::steady_clock::now();
                            auto now = std::chrono::steady_clock::now();
                            auto error_duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_error_log);
                            
                            if (error_duration.count() >= 5) {
                                std::cerr << "[WARNING] Failed to push buffer to appsrc: " << ret;
                                switch(ret) {
                                    case GST_FLOW_FLUSHING:
                                        std::cerr << " (FLUSHING - pipeline stopping)";
                                        // appsrc를 재설정하도록 표시
                                        appsrc_available = false;
                                        break;
                                    case GST_FLOW_EOS:
                                        std::cerr << " (EOS - end of stream)";
                                        // EOS 상태에서는 재시작이 필요할 수 있음
                                        appsrc_available = false;
                                        break;
                                    case GST_FLOW_NOT_LINKED:
                                        std::cerr << " (NOT_LINKED - pipeline not connected)";
                                        // 파이프라인 연결 복구 시도
                                        appsrc_available = false;
                                        break;
                                    case GST_FLOW_ERROR:
                                        std::cerr << " (ERROR)";
                                        break;
                                    default:
                                        std::cerr << " (OTHER: " << ret << ")";
                                        break;
                                }
                                std::cerr << std::endl;
                                last_error_log = now;
                            }
                        }
                    } else {
                        gst_buffer_unref(buffer);
                        if (rtsp_frame_count % 100 == 0) {
                            std::cerr << "[WARNING] Failed to map GstBuffer" << std::endl;
                        }
                    }
                } else {
                    if (rtsp_frame_count % 100 == 0) {
                        std::cerr << "[WARNING] Failed to allocate GstBuffer" << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Exception in sendFrameToRTSP: " << e.what() << std::endl;
            }
        } else {
            // appsrc가 사용 불가능한 경우
            static std::chrono::steady_clock::time_point last_reinit_attempt = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_reinit_attempt);
            
            // 10초마다 한 번씩만 경고 메시지 출력
            if (rtsp_frame_count % 300 == 0) {
                std::cerr << "[WARNING] RTSP appsrc not available (current_appsrc: " 
                          << (void*)current_appsrc << ", available: " << appsrc_available.load() << ")" << std::endl;
            }
            
            // 30초마다 RTSP 파이프라인 재연결 시도
            if (duration.count() >= 30) {
                std::cout << "[RTSP] Attempting to reconnect RTSP pipeline..." << std::endl;
                // 여기에서 RTSP 파이프라인을 재초기화하는 코드를 추가할 수 있음
                // 현재는 단순히 타임스탬프만 업데이트
                last_reinit_attempt = now;
            }
        }
    }
};

// 시그널 핸들러
static std::atomic<bool> shouldExit{false};
std::unique_ptr<RTSPZeroCopyOpenVINOTracker> tracker;
static std::atomic<bool> cleanup_done{false};
static std::atomic<int> signal_count{0};

void signalHandler(int signum) {
    int count = signal_count.fetch_add(1) + 1;
    
    std::cout << "\n종료 신호 수신 (" << signum << ") - 시도 " << count << std::endl;
    
    if (count == 1) {
        // 첫 번째 시그널: 정상 종료 시도
        std::cout << "정상 종료를 시도합니다..." << std::endl;
        
        // 중복 실행 방지
        bool expected = false;
        if (!cleanup_done.compare_exchange_strong(expected, true)) {
            return;  // 이미 정리가 진행 중
        }
        
        shouldExit.store(true);
        
        // 별도 스레드에서 정리 작업 수행 (시그널 핸들러에서 블로킹 방지)
        std::thread cleanup_thread([]() {
            try {
                if (tracker) {
                    std::cout << "트래커 정지 중..." << std::endl;
                    tracker->stop();
                    std::cout << "트래커 정지 완료" << std::endl;
                }
                
                std::cout << "RTSP 서버 정리 중..." << std::endl;
                cleanup_rtsp_server();
                std::cout << "RTSP 서버 정리 완료" << std::endl;
                
                std::cout << "정상 종료 완료" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "정리 중 오류 발생: " << e.what() << std::endl;
            }
            
            // 메인 루프가 끝나지 않으면 강제 종료
            std::this_thread::sleep_for(std::chrono::seconds(3));
            if (shouldExit.load()) {
                std::cout << "강제 종료 실행" << std::endl;
                std::exit(0);
            }
        });
        cleanup_thread.detach();
        
    } else if (count == 2) {
        // 두 번째 시그널: 강제 종료
        std::cout << "강제 종료를 실행합니다..." << std::endl;
        std::exit(1);
    } else {
        // 세 번째 이상: 즉시 종료
        std::cout << "즉시 종료" << std::endl;
        std::abort();
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== RTSP Zero Copy OpenVINO YOLOv5 + SORT + Risk Analysis Demo ===" << std::endl;
    std::cout << "모델 경로: " << YOLO_MODEL_PATH << std::endl;
    std::cout << "RTSP 스트림: rtsp://localhost:" << RTSP_PORT << RTSP_PATH << std::endl;
    
    // DB 초기화
    if (!initializeDatabase()) {
        std::cerr << "데이터베이스 초기화 실패" << std::endl;
        return -1;
    }

    // 시그널 핸들러 등록
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    try {
        // RTSP 서버 초기화
        if (!init_rtsp_server()) {
            std::cerr << "RTSP 서버 초기화 실패" << std::endl;
            return -1;
        }
        
        // RTSP 메인 루프 스레드 시작
        rtsp_thread = std::thread(rtsp_main_loop_thread);
        rtsp_thread.detach();  // 메인 스레드에서 분리하여 독립적으로 실행
        
        // RTSP 서버가 안정화될 때까지 약간 대기
        std::cout << "[MAIN] Waiting for RTSP server to stabilize..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        tracker = std::make_unique<RTSPZeroCopyOpenVINOTracker>(YOLO_MODEL_PATH);
        
        if (!tracker->initialize()) {
            std::cerr << "트래커 초기화 실패" << std::endl;
            return -1;
        }

        if (!tracker->setupBuffers()) {
            std::cerr << "버퍼 설정 실패" << std::endl;
            return -1;
        }

        tracker->run();

        // 메인 루프
        std::cout << "YOLOv5 + SORT + Risk Analysis 실행 중... (Ctrl+C로 종료)" << std::endl;
        while (!shouldExit.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "Main loop exit detected" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "오류 발생: " << e.what() << std::endl;
        shouldExit.store(true);
        
        // 오류 발생 시 정리
        if (tracker) {
            try {
                tracker->stop();
            } catch (const std::exception& cleanup_e) {
                std::cerr << "트래커 정리 중 오류: " << cleanup_e.what() << std::endl;
            }
        }
        
        try {
            cleanup_rtsp_server();
        } catch (const std::exception& cleanup_e) {
            std::cerr << "RTSP 서버 정리 중 오류: " << cleanup_e.what() << std::endl;
        }
        
        return -1;
    }
    
    // 정리 (cleanup_done이 false인 경우에만)
    if (!cleanup_done.load()) {
        std::cout << "Starting cleanup process..." << std::endl;
        
        if (tracker) {
            try {
                std::cout << "Stopping tracker..." << std::endl;
                tracker->stop();
                std::cout << "Tracker stopped" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error stopping tracker: " << e.what() << std::endl;
            }
        }
        
        try {
            cleanup_rtsp_server();
        } catch (const std::exception& e) {
            std::cerr << "Error cleaning up RTSP server: " << e.what() << std::endl;
        }
        
        if (rtsp_thread.joinable()) {
            std::cout << "Waiting for RTSP thread to finish..." << std::endl;
            // 타임아웃과 함께 스레드 대기
            auto future = std::async(std::launch::async, [&]() {
                rtsp_thread.join();
            });
            
            if (future.wait_for(std::chrono::seconds(3)) == std::future_status::timeout) {
                std::cout << "RTSP thread did not finish, detaching..." << std::endl;
                rtsp_thread.detach();
            } else {
                std::cout << "RTSP thread finished" << std::endl;
            }
        }
        
        cleanup_done.store(true);
    }

    std::cout << "프로그램 종료" << std::endl;
    return 0;
}

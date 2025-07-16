#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

// Python obj_det_and_trk.py의 SORT 기반 트래커 및 draw_boxes 기능 C++ 헤더 뼈대
// 실제 구현은 object_tracker.cpp에 분리 가능

struct Track {
    int id;
    cv::Rect bbox;
    int class_id;
    float confidence;
    std::vector<cv::Point> trajectory; // centroid history
};

class ObjectTracker {
public:
    ObjectTracker();
    // 프레임별 detection 결과를 받아 트래킹 결과 반환
    std::vector<Track> update(const std::vector<cv::Rect>& boxes,
                              const std::vector<int>& class_ids,
                              const std::vector<float>& confidences);
    // 트랙 색상 계산 (Python compute_color_for_labels)
    static cv::Scalar computeColorForLabel(int label);
    // 트랙 박스 및 ID 시각화
    static void drawTracks(cv::Mat& img, const std::vector<Track>& tracks, bool color_box = false);

private:
    int next_id;
    std::vector<Track> tracks;
    // ... (SORT 알고리즘 내부 상태 등)
};

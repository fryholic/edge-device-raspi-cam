#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "kalman_tracker.hpp"

// Python sort.py의 SORT 클래스 구조를 C++로 포팅한 헤더(구현은 sort.cpp에 분리)

class Sort {
public:
    Sort(int max_age=5, int min_hits=2, float iou_threshold=0.2);
    // dets: [N, 6] (x1, y1, x2, y2, conf, class_id)
    std::vector<std::vector<float>> update(const std::vector<std::vector<float>>& dets);
    std::vector<std::shared_ptr<KalmanTracker>> getTrackers() const;
private:
    int max_age;
    int min_hits;
    float iou_threshold;
    int frame_count;
    std::vector<std::shared_ptr<KalmanTracker>> trackers;
    // ...Hungarian 알고리즘 등 내부 멤버...
};

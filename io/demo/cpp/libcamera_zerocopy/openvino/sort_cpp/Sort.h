#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Represents a tracked object with its bounding box, ID, and class.
struct TrackingBox
{
    int frame;
    int id;
    cv::Rect_<float> box;
    int class_id;
};

class KalmanBoxTracker
{
public:
    KalmanBoxTracker(cv::Rect_<float> bbox, int class_id);
    void predict();
    void update(cv::Rect_<float> bbox, int class_id);
    cv::Rect_<float> get_state();
    cv::Rect_<float> get_rect_xysr(float cx, float cy, float s, float r);

    static int count;
    int m_id;
    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_class_id;

private:
    cv::KalmanFilter kf;
    cv::Mat measurement;
};

class Sort
{
public:
    Sort(int max_age = 1, int min_hits = 3, float iou_threshold = 0.3);
    std::vector<TrackingBox> update(const std::vector<TrackingBox>& det_boxes);
    
private:
    std::vector<KalmanBoxTracker> trackers;
    int max_age;
    int min_hits;
    float iou_threshold;
    int frame_count;

    double GetIoU(const cv::Rect_<float>& bb_test, const cv::Rect_<float>& bb_gt);
};

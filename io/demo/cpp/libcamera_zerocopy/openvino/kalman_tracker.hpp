#pragma once
#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>

// Python의 KalmanBoxTracker를 C++로 포팅한 헤더
class KalmanTracker {
public:
    KalmanTracker(const std::vector<float>& bbox);
    KalmanTracker(const cv::Rect2f& initBox, int class_id, float confidence); // Overloaded constructor
    void update(const std::vector<float>& bbox);
    void update(const cv::Rect2f& box, int class_id, float confidence); // Overloaded update method
    std::vector<float> predict();
    std::vector<float> get_state();
    int get_id() const { return id; }
    int get_class() const { return detclass; }
    int get_hit_streak() const { return hit_streak; }
    int get_time_since_update() const { return time_since_update; }
    std::vector<std::pair<float, float>> get_centroidarr() const { return centroidarr; }
    int time_since_update;
    int id;
    int hits;
    int hit_streak;
    int age;
private:
    static int count;
    int detclass;
    Eigen::Matrix<float, 7, 1> x; // state
    Eigen::Matrix<float, 7, 7> F; // transition
    Eigen::Matrix<float, 4, 7> H; // measurement
    Eigen::Matrix<float, 7, 7> P; // covariance
    Eigen::Matrix<float, 7, 7> Q; // process noise
    Eigen::Matrix<float, 4, 4> R; // measurement noise
    std::vector<std::pair<float, float>> centroidarr;
    std::vector<Eigen::Matrix<float, 7, 1>> history;
};

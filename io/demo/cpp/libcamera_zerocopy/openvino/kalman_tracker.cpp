#include "kalman_tracker.hpp"
#include <cmath>
#include <opencv2/opencv.hpp>

int KalmanTracker::count = 0;

// [x1, y1, x2, y2] -> [x, y, s, r]
static Eigen::Matrix<float, 4, 1> convert_bbox_to_z(const std::vector<float>& bbox) {
    float w = bbox[2] - bbox[0];
    float h = bbox[3] - bbox[1];
    float x = bbox[0] + w / 2.0f;
    float y = bbox[1] + h / 2.0f;
    float s = w * h;
    float r = w / (h + 1e-6f);
    Eigen::Matrix<float, 4, 1> z;
    z << x, y, s, r;
    return z;
}
// [x, y, s, r] -> [x1, y1, x2, y2]
static std::vector<float> convert_x_to_bbox(const Eigen::Matrix<float, 7, 1>& x) {
    float w = std::sqrt(x(2) * x(3));
    float h = x(2) / (w + 1e-6f);
    float x1 = x(0) - w / 2.0f;
    float y1 = x(1) - h / 2.0f;
    float x2 = x(0) + w / 2.0f;
    float y2 = x(1) + h / 2.0f;
    return {x1, y1, x2, y2};
}

KalmanTracker::KalmanTracker(const std::vector<float>& bbox) {
    // 칼만 필터 초기화
    F.setIdentity();
    F(0,4) = 1; F(1,5) = 1; F(2,6) = 1;
    H.setZero();
    H(0,0) = 1; H(1,1) = 1; H(2,2) = 1; H(3,3) = 1;
    P.setIdentity();
    P.block<3,3>(4,4) *= 1000.0f;
    P *= 10.0f;
    Q.setIdentity();
    Q(6,6) *= 0.5f;
    Q.block<3,3>(4,4) *= 0.5f;
    R.setIdentity();
    R.block<2,2>(2,2) *= 10.0f;
    x.setZero();
    x.block<4,1>(0,0) = convert_bbox_to_z(bbox);
    time_since_update = 0;
    id = count++;
    hits = 0;
    hit_streak = 0;
    age = 0;
    detclass = bbox.size() > 5 ? (int)bbox[5] : -1;
    float cx = (bbox[0] + bbox[2]) / 2.0f;
    float cy = (bbox[1] + bbox[3]) / 2.0f;
    centroidarr.push_back({cx, cy});
}

KalmanTracker::KalmanTracker(const cv::Rect2f& initBox, int class_id, float confidence) {
    // Initialize Kalman filter with cv::Rect2f
    float cx = initBox.x + initBox.width / 2.0f;
    float cy = initBox.y + initBox.height / 2.0f;
    float s = initBox.width * initBox.height;
    float r = initBox.width / (initBox.height + 1e-6f);
    Eigen::Matrix<float, 4, 1> z;
    z << cx, cy, s, r;
    F.setIdentity();
    F(0,4) = 1; F(1,5) = 1; F(2,6) = 1;
    H.setZero();
    H(0,0) = 1; H(1,1) = 1; H(2,2) = 1; H(3,3) = 1;
    P.setIdentity();
    P.block<3,3>(4,4) *= 1000.0f;
    P *= 10.0f;
    Q.setIdentity();
    Q(6,6) *= 0.5f;
    Q.block<3,3>(4,4) *= 0.5f;
    R.setIdentity();
    R.block<2,2>(2,2) *= 10.0f;
    x.setZero();
    x.block<4,1>(0,0) = z;
    time_since_update = 0;
    id = count++;
    hits = 0;
    hit_streak = 0;
    age = 0;
    detclass = class_id;
    centroidarr.push_back({cx, cy});
}

void KalmanTracker::update(const std::vector<float>& bbox) {
    // 측정값 z
    Eigen::Matrix<float, 4, 1> z = convert_bbox_to_z(bbox);
    // 칼만 필터 업데이트
    Eigen::Matrix<float, 4, 1> y = z - H * x;
    Eigen::Matrix<float, 4, 4> S = H * P * H.transpose() + R;
    Eigen::Matrix<float, 7, 4> K = P * H.transpose() * S.inverse();
    x = x + K * y;
    P = (Eigen::Matrix<float, 7, 7>::Identity() - K * H) * P;
    time_since_update = 0;
    hits++;
    hit_streak++;
    detclass = bbox.size() > 5 ? (int)bbox[5] : detclass;
    float cx = (bbox[0] + bbox[2]) / 2.0f;
    float cy = (bbox[1] + bbox[3]) / 2.0f;
    centroidarr.push_back({cx, cy});
    history.clear();
}

void KalmanTracker::update(const cv::Rect2f& box, int class_id, float confidence) {
    // Update Kalman filter with cv::Rect2f
    float cx = box.x + box.width / 2.0f;
    float cy = box.y + box.height / 2.0f;
    float s = box.width * box.height;
    float r = box.width / (box.height + 1e-6f);
    Eigen::Matrix<float, 4, 1> z;
    z << cx, cy, s, r;
    Eigen::Matrix<float, 4, 1> y_kalman = z - H * x;
    Eigen::Matrix<float, 4, 4> S = H * P * H.transpose() + R;
    Eigen::Matrix<float, 7, 4> K = P * H.transpose() * S.inverse();
    x = x + K * y_kalman;
    P = (Eigen::Matrix<float, 7, 7>::Identity() - K * H) * P;
    time_since_update = 0;
    hits++;
    hit_streak++;
    detclass = class_id;
    centroidarr.push_back({cx, cy});
    history.clear();
}

std::vector<float> KalmanTracker::predict() {
    // 칼만 필터 예측
    if ((x(6) + x(2)) <= 0) x(6) = 0;
    x = F * x;
    P = F * P * F.transpose() + Q;
    age++;
    if (time_since_update > 0) hit_streak = 0;
    time_since_update++;
    history.push_back(x);
    return convert_x_to_bbox(x);
}

std::vector<float> KalmanTracker::get_state() {
    // 현재 상태 반환
    std::vector<float> bbox = convert_x_to_bbox(x);
    bbox.push_back((float)detclass);
    bbox.push_back(x(4)); // u_dot
    bbox.push_back(x(5)); // v_dot
    bbox.push_back(x(6)); // s_dot
    return bbox;
}

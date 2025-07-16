#include "Sort.h"
#include <iostream>

// Initialize static member
int KalmanBoxTracker::count = 0;

// Constructor
KalmanBoxTracker::KalmanBoxTracker(cv::Rect_<float> bbox, int class_id)
{
    kf.init(7, 4, 0, CV_32F);
    measurement = cv::Mat::zeros(4, 1, CV_32F);

    // Transition Matrix F
    kf.transitionMatrix = (cv::Mat_<float>(7, 7) << 
        1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1);

    // Measurement Matrix H
    kf.measurementMatrix = (cv::Mat_<float>(4, 7) <<
        1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0);

    // Process Noise Covariance Matrix Q
    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    kf.processNoiseCov.at<float>(4, 4) = 1e-1;
    kf.processNoiseCov.at<float>(5, 5) = 1e-1;
    kf.processNoiseCov.at<float>(6, 6) = 1e-2;

    // Measurement Noise Covariance Matrix R
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));

    // Error Covariance Post Matrix P
    setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    // Initial state
    kf.statePost.at<float>(0) = bbox.x + bbox.width / 2;
    kf.statePost.at<float>(1) = bbox.y + bbox.height / 2;
    kf.statePost.at<float>(2) = bbox.area();
    kf.statePost.at<float>(3) = bbox.width / bbox.height;

    m_time_since_update = 0;
    m_hits = 0;
    m_hit_streak = 0;
    m_age = 0;
    m_id = count++;
    m_class_id = class_id;
}

// Predict the state
void KalmanBoxTracker::predict()
{
    kf.predict();
    m_age++;
    if (m_time_since_update > 0)
        m_hit_streak = 0;
    m_time_since_update++;
}

// Update the state with observed bbox
void KalmanBoxTracker::update(cv::Rect_<float> bbox, int class_id)
{
    m_time_since_update = 0;
    m_hits++;
    m_hit_streak++;
    m_class_id = class_id;

    measurement.at<float>(0) = bbox.x + bbox.width / 2;
    measurement.at<float>(1) = bbox.y + bbox.height / 2;
    measurement.at<float>(2) = bbox.area();
    measurement.at<float>(3) = bbox.width / bbox.height;

    kf.correct(measurement);
}

// Get the current bounding box estimate
cv::Rect_<float> KalmanBoxTracker::get_state()
{
    cv::Mat state = kf.statePost;
    return get_rect_xysr(state.at<float>(0), state.at<float>(1), state.at<float>(2), state.at<float>(3));
}

// Convert [cx, cy, s, r] to [x1, y1, x2, y2]
cv::Rect_<float> KalmanBoxTracker::get_rect_xysr(float cx, float cy, float s, float r)
{
    float w = sqrt(s * r);
    float h = s / w;
    return cv::Rect_<float>(cx - w / 2, cy - h / 2, w, h);
}

// Sort constructor
Sort::Sort(int max_age, int min_hits, float iou_threshold)
    : max_age(max_age), min_hits(min_hits), iou_threshold(iou_threshold), frame_count(0) {}

// Computes IoU between two bounding boxes
double Sort::GetIoU(const cv::Rect_<float>& bb_test, const cv::Rect_<float>& bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON)
        return 0;
    return (double)(in / un);
}

// Main update function
std::vector<TrackingBox> Sort::update(const std::vector<TrackingBox>& det_boxes)
{
    frame_count++;

    // Predict next state for each tracker
    std::vector<cv::Rect_<float>> predicted_boxes;
    for (auto& tracker : trackers)
    {
        tracker.predict();
        predicted_boxes.push_back(tracker.get_state());
    }

    // Associate detections with trackers
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_detections;
    std::vector<int> unmatched_trackers;

    if (!trackers.empty())
    {
        cv::Mat iou_matrix(det_boxes.size(), trackers.size(), CV_32F);
        for (size_t i = 0; i < det_boxes.size(); i++)
        {
            for (size_t j = 0; j < trackers.size(); j++)
            {
                iou_matrix.at<float>(i, j) = GetIoU(det_boxes[i].box, predicted_boxes[j]);
            }
        }

        for (size_t i = 0; i < det_boxes.size(); i++)
        {
            double max_iou = 0;
            int max_idx = -1;
            for (size_t j = 0; j < trackers.size(); j++)
            {
                if (iou_matrix.at<float>(i, j) > max_iou)
                {
                    max_iou = iou_matrix.at<float>(i, j);
                    max_idx = j;
                }
            }
            if (max_iou > iou_threshold)
            {
                matches.push_back({(int)i, max_idx});
            }
            else
            {
                unmatched_detections.push_back(i);
            }
        }
    }
    else
    {
        for (size_t i = 0; i < det_boxes.size(); i++)
            unmatched_detections.push_back(i);
    }
    
    std::vector<bool> tracker_matched(trackers.size(), false);
    for(const auto& match : matches)
    {
        trackers[match.second].update(det_boxes[match.first].box, det_boxes[match.first].class_id);
        tracker_matched[match.second] = true;
    }

    // Create new trackers for unmatched detections
    for (int umd_idx : unmatched_detections)
    {
        trackers.emplace_back(det_boxes[umd_idx].box, det_boxes[umd_idx].class_id);
    }

    // Remove dead trackers and prepare output
    std::vector<TrackingBox> frame_tracking_result;
    auto it = trackers.begin();
    while (it != trackers.end())
    {
        if (it->m_time_since_update < 1 && (it->m_hit_streak >= min_hits || frame_count <= min_hits))
        {
            TrackingBox res;
            res.box = it->get_state();
            res.id = it->m_id;
            res.class_id = it->m_class_id;
            res.frame = frame_count;
            frame_tracking_result.push_back(res);
        }

        if (it->m_time_since_update > max_age)
            it = trackers.erase(it);
        else
            ++it;
    }

    return frame_tracking_result;
}

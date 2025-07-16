#include "object_tracker.hpp"
#include <cmath>
#include <algorithm>

// 간단한 IoU 계산 함수
static float iou(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = a.area() + b.area() - interArea;
    return unionArea > 0 ? (float)interArea / unionArea : 0.0f;
}

ObjectTracker::ObjectTracker() : next_id(0) {}

std::vector<Track> ObjectTracker::update(const std::vector<cv::Rect>& boxes,
                                         const std::vector<int>& class_ids,
                                         const std::vector<float>& confidences) {
    // 매우 단순한 IoU 기반 트래킹 (실제 SORT는 칼만필터+IoU Hungarian)
    std::vector<bool> matched(boxes.size(), false);
    for (auto& track : tracks) {
        float best_iou = 0.0f;
        int best_idx = -1;
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (matched[i]) continue;
            float iou_score = iou(track.bbox, boxes[i]);
            if (iou_score > best_iou) {
                best_iou = iou_score;
                best_idx = i;
            }
        }
        if (best_iou > 0.3 && best_idx != -1) {
            // 트랙 갱신
            track.bbox = boxes[best_idx];
            track.class_id = class_ids[best_idx];
            track.confidence = confidences[best_idx];
            cv::Point center(track.bbox.x + track.bbox.width/2, track.bbox.y + track.bbox.height/2);
            track.trajectory.push_back(center);
            matched[best_idx] = true;
        }
    }
    // 매칭 안된 detection은 새 트랙 생성
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!matched[i]) {
            Track t;
            t.id = next_id++;
            t.bbox = boxes[i];
            t.class_id = class_ids[i];
            t.confidence = confidences[i];
            t.trajectory.push_back(cv::Point(boxes[i].x + boxes[i].width/2, boxes[i].y + boxes[i].height/2));
            tracks.push_back(t);
        }
    }
    // 오래된 트랙 삭제(여기선 단순화)
    // 실제 SORT는 age, hit, miss 등 관리
    return tracks;
}

cv::Scalar ObjectTracker::computeColorForLabel(int label) {
    // Python의 palette와 유사하게 구현
    int palette[3] = {2047, 32767, 1048575};
    return cv::Scalar(
        (palette[0] * (label * label - label + 1)) % 255,
        (palette[1] * (label * label - label + 1)) % 255,
        (palette[2] * (label * label - label + 1)) % 255
    );
}

void ObjectTracker::drawTracks(cv::Mat& img, const std::vector<Track>& tracks, bool color_box) {
    for (const auto& t : tracks) {
        cv::Scalar color = color_box ? computeColorForLabel(t.id) : cv::Scalar(255,191,0);
        cv::rectangle(img, t.bbox, color, 2);
        std::string label = std::to_string(t.id);
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        cv::rectangle(img, cv::Rect(t.bbox.x, t.bbox.y - 20, labelSize.width, 20), cv::Scalar(255,191,0), -1);
        cv::putText(img, label, cv::Point(t.bbox.x, t.bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1);
        // 궤적 그리기
        for (size_t i = 1; i < t.trajectory.size(); ++i) {
            cv::line(img, t.trajectory[i-1], t.trajectory[i], color, 2);
        }
    }
}

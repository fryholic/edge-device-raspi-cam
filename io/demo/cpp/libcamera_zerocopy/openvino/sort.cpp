#include "sort.hpp"
#include "kalman_tracker.hpp"
#include <vector>
#include <memory>
#include <limits>
#include <algorithm>
#include <cmath>


// IoU 계산 함수 ([x1,y1,x2,y2] 형식)
inline float iou(const std::vector<float>& bb_test, const std::vector<float>& bb_gt) {
    float xx1 = std::max(bb_test[0], bb_gt[0]);
    float yy1 = std::max(bb_test[1], bb_gt[1]);
    float xx2 = std::min(bb_test[2], bb_gt[2]);
    float yy2 = std::min(bb_test[3], bb_gt[3]);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float wh = w * h;
    float o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6f);
    return o;
}

// cv::Rect2f -> std::vector<float> 변환 함수
inline std::vector<float> rect2f_to_vec(const cv::Rect2f& r) {
    return {r.x, r.y, r.x + r.width, r.y + r.height};
}

// Hungarian 알고리즘(최소 비용 할당) - 단순 구현 (O(n^3))
void hungarian(const std::vector<std::vector<float>>& cost, std::vector<int>& assignment) {
    // assignment[i]=j: i번째 detection은 j번째 tracker에 할당
    int n = cost.size(), m = cost[0].size();
    assignment.assign(n, -1);
    std::vector<bool> used(m, false);
    for (int i = 0; i < n; ++i) {
        float min_cost = std::numeric_limits<float>::max();
        int min_j = -1;
        for (int j = 0; j < m; ++j) {
            if (!used[j] && cost[i][j] < min_cost) {
                min_cost = cost[i][j];
                min_j = j;
            }
        }
        if (min_j != -1) {
            assignment[i] = min_j;
            used[min_j] = true;
        }
    }
}


// Sort 생성자 구현
Sort::Sort(int max_age, int min_hits, float iou_threshold)
    : max_age(max_age), min_hits(min_hits), iou_threshold(iou_threshold), frame_count(0) {}

// Sort::update 구현
std::vector<std::vector<float>> Sort::update(const std::vector<std::vector<float>>& dets) {
    frame_count++;
    // 1. 트래커 예측
    std::vector<std::vector<float>> trks;
    for (auto& t : trackers) {
        // t->predict()는 std::vector<float> 반환
        trks.push_back(t->predict());
    }
    // 2. IoU 매트릭스 계산
    int N = dets.size(), M = trks.size();
    std::vector<std::vector<float>> iou_matrix(N, std::vector<float>(M, 0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            iou_matrix[i][j] = -iou(dets[i], trks[j]); // -IoU (최소 비용)
    // 3. Hungarian 알고리즘으로 매칭
    std::vector<int> assignment;
    if (N > 0 && M > 0)
        hungarian(iou_matrix, assignment);
    else
        assignment.assign(N, -1);
    // 4. 매칭 결과로 트래커 업데이트
    std::vector<bool> matched_trk(M, false);
    for (int i = 0; i < N; ++i) {
        int j = assignment[i];
        if (j != -1 && -iou_matrix[i][j] >= iou_threshold) {
            // dets[i]: [x1, y1, x2, y2, conf, class_id]
            cv::Rect2f box(dets[i][0], dets[i][1], dets[i][2]-dets[i][0], dets[i][3]-dets[i][1]);
            int class_id = dets[i].size() > 5 ? (int)dets[i][5] : -1;
            float conf = dets[i].size() > 4 ? dets[i][4] : 0.f;
            trackers[j]->update(box, class_id, conf);
            matched_trk[j] = true;
        } else {
            // 새로운 트래커 생성
            cv::Rect2f box(dets[i][0], dets[i][1], dets[i][2]-dets[i][0], dets[i][3]-dets[i][1]);
            int class_id = dets[i].size() > 5 ? (int)dets[i][5] : -1;
            float conf = dets[i].size() > 4 ? dets[i][4] : 0.f;
            trackers.push_back(std::make_shared<KalmanTracker>(box, class_id, conf));
        }
    }
    // 5. 오래된 트래커 삭제
    for (int i = (int)trackers.size()-1; i >= 0; --i) {
        if (trackers[i]->get_time_since_update() > max_age)
            trackers.erase(trackers.begin() + i);
    }
    // 6. 결과 반환 (id 포함)
    std::vector<std::vector<float>> ret;
    for (auto& t : trackers) {
        auto rect = t->get_state();
        std::vector<float> state = rect;
        // state: [x1, y1, x2, y2, class, u_dot, v_dot, s_dot]
        // class는 get_class()로, id는 get_id()로
        state.push_back((float)t->get_class());
        if (t->get_time_since_update() < 1 && (t->get_hit_streak() >= min_hits || frame_count <= min_hits)) {
            state.push_back((float)(t->get_id()+1)); // MOT benchmark는 1-base
            ret.push_back(state);
        }
    }
    return ret;
}

std::vector<std::shared_ptr<KalmanTracker>> Sort::getTrackers() const { return trackers; }

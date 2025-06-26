#include <iostream>
#include <memory>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sys/mman.h>

#include <libcamera/libcamera.h>
#include <opencv2/opencv.hpp>

using namespace libcamera;

const int CAPTURE_WIDTH = 1280;
const int CAPTURE_HEIGHT = 720;

std::queue<Request*> completed_requests;
std::mutex request_mutex;
std::condition_variable request_cond;

void requestCompleteCallback(Request* r) {
    std::lock_guard<std::mutex> lock(request_mutex);
    completed_requests.push(r);
    request_cond.notify_one();
}

int main() {
    auto cm = std::make_unique<CameraManager>();
    cm->start();

    if (cm->cameras().empty()) {
        std::cerr << "카메라를 찾을 수 없습니다." << std::endl;
        return 1;
    }

    std::shared_ptr<Camera> camera = cm->cameras()[0];
    camera->acquire();

    std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration({ StreamRole::Viewfinder });
    StreamConfiguration& streamConfig = config->at(0);
    streamConfig.size.width = CAPTURE_WIDTH;
    streamConfig.size.height = CAPTURE_HEIGHT;
    // 픽셀 포맷을 YUYV로 설정 (RGB888 대신 시도)
    streamConfig.pixelFormat = formats::YUYV;
    config->validate();
    camera->configure(config.get());

    // 실제 설정된 값 확인
    std::cout << "Actual resolution: " 
              << streamConfig.size.width << "x" 
              << streamConfig.size.height << std::endl;
    std::cout << "Stride: " << streamConfig.stride << std::endl;
    std::cout << "Pixel Format: " << streamConfig.pixelFormat.toString() << std::endl;

    FrameBufferAllocator* allocator = new FrameBufferAllocator(camera);
    allocator->allocate(streamConfig.stream());

    std::vector<std::unique_ptr<Request>> requests;
    for (const std::unique_ptr<FrameBuffer>& buffer : allocator->buffers(streamConfig.stream())) {
        auto request = camera->createRequest();
        request->addBuffer(streamConfig.stream(), buffer.get());
        requests.push_back(std::move(request));
    }

    camera->requestCompleted.connect(requestCompleteCallback);
    camera->start();
    for (auto& req : requests) {
        camera->queueRequest(req.get());
    }

    std::cout << "카메라 스트리밍 시작... (종료하려면 'q' 키를 누르세요)" << std::endl;

    bool running = true;
    int frame_counter = 0;
    auto fps_start_time = std::chrono::high_resolution_clock::now();

    while (running) {
        Request* completed_request;
        {
            std::unique_lock<std::mutex> lock(request_mutex);
            if (request_cond.wait_for(lock, std::chrono::milliseconds(200), 
                [&] { return !completed_requests.empty(); })) {
                completed_request = completed_requests.front();
                completed_requests.pop();
            } else {
                continue;
            }
        }

        frame_counter++;
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - fps_start_time).count();
        if (duration >= 1) {
            printf("Average FPS: %.2f\n", static_cast<double>(frame_counter) / duration);
            frame_counter = 0;
            fps_start_time = now;
        }

        FrameBuffer* buffer = completed_request->buffers().at(streamConfig.stream());
        const std::vector<FrameBuffer::Plane>& planes = buffer->planes();
        
        // YUYV 포맷은 단일 평면을 가집니다.
        if (planes.empty()) {
            std::cerr << "오류: YUYV 포맷에 필요한 평면이 없습니다." << std::endl;
            completed_request->reuse(Request::ReuseBuffers);
            camera->queueRequest(completed_request);
            continue;
        }

        // 디버깅 메시지 줄이기 - 첫 번째 프레임만 출력
        static bool first_frame = true;
        if (first_frame) {
            for (size_t i = 0; i < planes.size(); ++i) {
                std::cout << "Plane " << i << " fd: " << planes[i].fd.get()
                          << ", length: " << planes[i].length << std::endl;
            }
            first_frame = false;
        }

        // 메모리 매핑
        std::vector<void*> mapped_data;
        for (const auto& plane : planes) {
            void* ptr = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
            if (ptr == MAP_FAILED) {
                perror("mmap 실패");
                continue;
            }
            mapped_data.push_back(ptr);
        }

        if (mapped_data.empty()) {
            std::cerr << "메모리 매핑 실패" << std::endl;
            for (size_t i = 0; i < mapped_data.size(); ++i) {
                munmap(mapped_data[i], planes[i].length);
            }
            completed_request->reuse(Request::ReuseBuffers);
            camera->queueRequest(completed_request);
            continue;
        }

        // 스트라이드 정보
        const int stride = streamConfig.stride;
        const int width = streamConfig.size.width;
        const int height = streamConfig.size.height;

        // 버퍼 길이 확인 (디버깅용) - 첫 번째 프레임만 출력
        static bool first_buffer_check = true;
        if (first_buffer_check) {
            std::cout << "Plane 0 length: " << planes[0].length << " (expected min: " << stride * height << ")" << std::endl;
            first_buffer_check = false;
        }

        // 안전한 메모리 접근을 위한 검증 추가
        if (planes[0].length < static_cast<size_t>(stride * height)) {
            std::cerr << "YUYV buffer too small!" << std::endl;
        }

        // YUYV 데이터를 OpenCV로 처리
        // YUYV는 픽셀당 2바이트이므로 CV_8UC2로 생성
        cv::Mat yuyv_mat(height, width, CV_8UC2, mapped_data[0]);
        
        // YUYV → BGR 변환 (OpenCV가 직접 지원)
        cv::Mat bgr_image;
        cv::cvtColor(yuyv_mat, bgr_image, cv::COLOR_YUV2BGR_YUYV);

        // 화면에 표시
        cv::imshow("libcamera C++ OpenCV Demo", bgr_image);

        // 매핑 해제
        for (size_t i = 0; i < mapped_data.size(); ++i) {
            munmap(mapped_data[i], planes[i].length);
        }

        completed_request->reuse(Request::ReuseBuffers);
        camera->queueRequest(completed_request);

        if (cv::waitKey(1) == 'q') {
            running = false;
        }
    }

    camera->stop();
    allocator->free(streamConfig.stream());
    delete allocator;
    camera->release();
    cm->stop();
    cv::destroyAllWindows();

    return 0;
}
#include <iostream>
#include <memory>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>

// [수정 3] mmap 관련 헤더 파일 추가
#include <sys/mman.h>

#include <libcamera/libcamera.h>
#include <SDL2/SDL.h>

using namespace libcamera;

// --- 설정 ---
const int CAPTURE_WIDTH = 1280;
const int CAPTURE_HEIGHT = 720;

// [수정 1] main 함수 내의 공유 변수들을 전역으로 이동
std::queue<Request *> completed_requests;
std::mutex request_mutex;
std::condition_variable request_cond;

// [수정 1] 람다 대신 전역 콜백 함수로 변경
void requestCompleteCallback(Request *r) {
    std::unique_lock<std::mutex> lock(request_mutex);
    completed_requests.push(r);
    request_cond.notify_one();
}


// YUV420 to ARGB 변환 함수 (이전과 동일)
void YUV420_to_ARGB(const uint8_t* y_plane, const uint8_t* u_plane, const uint8_t* v_plane,
                     int y_stride, int uv_stride, int width, int height, uint8_t* argb_data, int argb_pitch) {
    for (int y_coord = 0; y_coord < height; ++y_coord) {
        for (int x_coord = 0; x_coord < width; ++x_coord) {
            int y_val = y_plane[y_coord * y_stride + x_coord];
            int u_val = u_plane[(y_coord / 2) * uv_stride + (x_coord / 2)];
            int v_val = v_plane[(y_coord / 2) * uv_stride + (x_coord / 2)];

            int c = y_val - 16;
            int d = u_val - 128;
            int e = v_val - 128;

            int r = std::max(0, std::min(255, (298 * c + 409 * e + 128) >> 8));
            int g = std::max(0, std::min(255, (298 * c - 100 * d - 208 * e + 128) >> 8));
            int b = std::max(0, std::min(255, (298 * c + 516 * d + 128) >> 8));

            uint8_t* pixel = argb_data + y_coord * argb_pitch + x_coord * 4;
            pixel[0] = b; pixel[1] = g; pixel[2] = r; pixel[3] = 255;
        }
    }
}


int main() {
    auto cm = std::make_unique<CameraManager>();
    cm->start();

    if (cm->cameras().empty()) { std::cerr << "카메라를 찾을 수 없습니다." << std::endl; return 1; }
    std::shared_ptr<Camera> camera = cm->cameras()[0];
    camera->acquire();

    std::unique_ptr<CameraConfiguration> config = camera->generateConfiguration({ StreamRole::Viewfinder });
    StreamConfiguration &streamConfig = config->at(0);
    streamConfig.size.width = CAPTURE_WIDTH;
    streamConfig.size.height = CAPTURE_HEIGHT;
    streamConfig.pixelFormat = formats::YUV420;
    config->validate();
    camera->configure(config.get());

    FrameBufferAllocator *allocator = new FrameBufferAllocator(camera);
    allocator->allocate(streamConfig.stream());

    std::vector<std::unique_ptr<Request>> requests;
    for (const std::unique_ptr<FrameBuffer> &buffer : allocator->buffers(streamConfig.stream())) {
        auto request = camera->createRequest();
        request->addBuffer(streamConfig.stream(), buffer.get());
        requests.push_back(std::move(request));
    }

    // --- [최종 수정] 모든 SDL 생성 함수에 오류 검사 추가 ---
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL_Init 실패: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("libcamera C++ Demo (SDL2)", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, CAPTURE_WIDTH, CAPTURE_HEIGHT, 0);
    if (window == nullptr) {
        std::cerr << "SDL_CreateWindow 실패: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr) {
        std::cerr << "SDL_CreateRenderer 실패: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, CAPTURE_WIDTH, CAPTURE_HEIGHT);
    if (texture == nullptr) {
        std::cerr << "SDL_CreateTexture 실패: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // [수정 1] 전역 콜백 함수를 connect에 전달
    camera->requestCompleted.connect(requestCompleteCallback);

    camera->start();
    for (auto &req : requests) {
        camera->queueRequest(req.get());
    }
    std::cout << "카메라 스트리밍 시작..." << std::endl;

    bool running = true;
    SDL_Event event;
    auto last_frame_time = std::chrono::high_resolution_clock::now();

    while (running) {
        while(SDL_PollEvent(&event)) { if (event.type == SDL_QUIT) running = false; }

        Request *completed_request;
        {
            std::unique_lock<std::mutex> lock(request_mutex);
            if (request_cond.wait_for(lock, std::chrono::milliseconds(200), [&]{ return !completed_requests.empty(); })) {
                completed_request = completed_requests.front();
                completed_requests.pop();
            } else {
                continue;
            }
        }
        
        auto now = std::chrono::high_resolution_clock::now();
        double fps = 1e6 / std::chrono::duration_cast<std::chrono::microseconds>(now - last_frame_time).count();
        last_frame_time = now;
        printf("FPS: %.2f\n", fps);

        FrameBuffer *buffer = completed_request->buffers().at(streamConfig.stream());
        
        const std::vector<FrameBuffer::Plane> &planes = buffer->planes();
        std::vector<void *> mapped_data;
        bool mmap_success = true;

        if (planes.size() < 3) {
            std::cerr << "오류: YUV420 포맷에 필요한 3개의 plane이 감지되지 않았습니다." << std::endl;
            mmap_success = false;
        } else {
            for(const auto& plane : planes) {
                void *ptr = mmap(nullptr, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
                if (ptr == MAP_FAILED) {
                    perror("mmap 실패");
                    mmap_success = false;
                    break;
                }
                mapped_data.push_back(ptr);
            }
        }
        
        if (mmap_success) {
            void* texture_pixels;
            int texture_pitch;

            // --- [최종 수정] SDL_LockTexture의 반환값을 확인 ---
            if (SDL_LockTexture(texture, NULL, &texture_pixels, &texture_pitch) != 0) {
                std::cerr << "SDL_LockTexture 실패: " << SDL_GetError() << std::endl;
            } else {
                // Lock 성공 시에만 변환 및 렌더링 수행
                YUV420_to_ARGB(static_cast<const uint8_t*>(mapped_data[0]),
                               static_cast<const uint8_t*>(mapped_data[1]),
                               static_cast<const uint8_t*>(mapped_data[2]),
                               streamConfig.stride, streamConfig.stride / 2,
                               streamConfig.size.width, streamConfig.size.height,
                               static_cast<uint8_t*>(texture_pixels), texture_pitch);

                SDL_UnlockTexture(texture);
                
                // 화면에 최종 렌더링
                SDL_RenderClear(renderer);
                SDL_RenderCopy(renderer, texture, NULL, NULL);
                SDL_RenderPresent(renderer);
            }
        }

        for(size_t i = 0; i < mapped_data.size(); ++i) {
            munmap(mapped_data[i], planes[i].length);
        }

        completed_request->reuse(Request::ReuseBuffers);
        camera->queueRequest(completed_request);
    }

    camera->stop();
    allocator->free(streamConfig.stream());
    delete allocator;
    camera->release();
    cm->stop();

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <cstring>

// V4L2 헤더
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

// SDL2 헤더
#include <SDL2/SDL.h>

// --- 설정 ---
const char* DEVICE_PATH = "/dev/video0";
const int CAPTURE_WIDTH = 640;
const int CAPTURE_HEIGHT = 480;
const int BUFFER_COUNT = 4;

struct Buffer {
    void* start;
    size_t length;
};

// YUYV 변환 함수는 더 이상 필요 없음

int main() {
    // --- V4L2 초기화 ---
    int fd = open(DEVICE_PATH, O_RDWR);
    if (fd == -1) { /* ... 에러 처리 ... */ return 1; }

    v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) { /* ... */ return 1; }
    std::cout << "드라이버: " << cap.driver << ", 카드: " << cap.card << std::endl;

    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = CAPTURE_WIDTH;
    fmt.fmt.pix.height = CAPTURE_HEIGHT;
    // --- [변경점 1] 픽셀 포맷을 RGB24로 요청 ---
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24; 
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("VIDIOC_S_FMT 실패");
        close(fd);
        return 1;
    }
    // 드라이버가 실제로 설정한 값 확인 (디버깅에 유용)
    std::cout << "실제 적용된 해상도: " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << std::endl;


    // 버퍼 요청 및 mmap 부분은 이전 코드와 동일
    v4l2_requestbuffers reqbuf = {};
    reqbuf.count = BUFFER_COUNT;
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &reqbuf) == -1) { /* ... */ return 1; }

    std::vector<Buffer> buffers(reqbuf.count);
    for (unsigned int i = 0; i < reqbuf.count; ++i) {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) { /* ... */ return 1; }
        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED) { /* ... */ return 1; }
    }

    // 버퍼 큐잉 및 스트림 시작 부분은 이전 코드와 동일
    for (unsigned int i = 0; i < reqbuf.count; ++i) {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) { /* ... */ return 1; }
    }
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        perror("VIDIOC_STREAMON 실패");
        return 1;
    }
    std::cout << "스트리밍 시작..." << std::endl;

    // --- SDL2 초기화 ---
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("V4L2 (unicam) RGB Test", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, CAPTURE_WIDTH, CAPTURE_HEIGHT, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    // --- [변경점 2] SDL 텍스처 포맷도 RGB24로 변경 ---
    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, CAPTURE_WIDTH, CAPTURE_HEIGHT);


    // --- 메인 루프 ---
    bool running = true;
    SDL_Event event;
    auto last_frame_time = std::chrono::high_resolution_clock::now();

    while (running) {
        while(SDL_PollEvent(&event)) { if (event.type == SDL_QUIT) running = false; }

        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) { break; }
        
        // FPS 계산 (이전과 동일)
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_frame_time).count();
        last_frame_time = current_time;
        double fps = (duration > 0) ? 1000000.0 / duration : 0.0;
        printf("프레임 획득 속도: %.2f FPS\n", fps);


        // --- [변경점 3] YUYV 변환 대신 직접 메모리 복사 ---
        void* pixels;
        int pitch;
        SDL_LockTexture(texture, NULL, &pixels, &pitch);
        
        // V4L2 버퍼의 데이터(RGB)를 SDL 텍스처로 바로 복사
        // pitch는 한 줄의 바이트 수, 보통 width * 3과 같지만 다를 수 있으므로 pitch를 사용
        if (pitch == CAPTURE_WIDTH * 3) {
            memcpy(pixels, buffers[buf.index].start, buf.bytesused);
        } else { // pitch와 실제 이미지 너비가 다를 경우 한 줄씩 복사
            unsigned char* src = static_cast<unsigned char*>(buffers[buf.index].start);
            unsigned char* dst = static_cast<unsigned char*>(pixels);
            for (int y = 0; y < CAPTURE_HEIGHT; ++y) {
                memcpy(dst + y * pitch, src + y * (CAPTURE_WIDTH * 3), CAPTURE_WIDTH * 3);
            }
        }
        SDL_UnlockTexture(texture);

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) { break; }
    }

    // --- 정리 --- (이전과 동일)
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    for (const auto& b : buffers) munmap(b.start, b.length);
    close(fd);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
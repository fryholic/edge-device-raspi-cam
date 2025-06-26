#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <SDL2/SDL.h>

const char* DEVICE_PATH = "/dev/video0";
const int CAPTURE_WIDTH = 640;
const int CAPTURE_HEIGHT = 480;
const int REQUEST_FPS = 30; // 요청할 프레임률
const int BUFFER_COUNT = 4;

struct Buffer {
    void* start;
    size_t length;
};

int main() {
    int fd = open(DEVICE_PATH, O_RDWR);
    if (fd == -1) { perror("장치 열기 실패"); return 1; }

    v4l2_capability cap;
    ioctl(fd, VIDIOC_QUERYCAP, &cap);
    std::cout << "드라이버: " << cap.driver << ", 카드: " << cap.card << std::endl;

    v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = CAPTURE_WIDTH;
    fmt.fmt.pix.height = CAPTURE_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("VIDIOC_S_FMT 실패");
        close(fd);
        return 1;
    }
    std::cout << "실제 적용된 해상도: " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << std::endl;
    
    // --- [새로운 부분] 스트리밍 파라미터(프레임률) 설정 ---
    v4l2_streamparm parm = {};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    // 현재 파라미터 가져오기 (필수는 아니지만 좋은 습관)
    if (ioctl(fd, VIDIOC_G_PARM, &parm) == -1) {
        perror("VIDIOC_G_PARM 실패");
        // 실패해도 계속 진행해볼 수 있음
    }
    
    std::cout << "현재 프레임률: " << parm.parm.capture.timeperframe.denominator << "/" << parm.parm.capture.timeperframe.numerator << std::endl;
    
    // 새로운 프레임률 설정
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = REQUEST_FPS;

    if (ioctl(fd, VIDIOC_S_PARM, &parm) == -1) {
        perror("VIDIOC_S_PARM 실패");
        // 이 부분이 실패하면 스트리밍이 안 될 가능성이 높음
    }
    std::cout << "요청 후 프레임률: " << parm.parm.capture.timeperframe.denominator << "/" << parm.parm.capture.timeperframe.numerator << std::endl;
    // --- [여기까지] ---

    // 버퍼 요청 및 나머지 부분은 이전과 동일...
    v4l2_requestbuffers reqbuf = {};
    // ... (이하 모든 코드는 이전과 동일)

    // 이 코드를 테스트하기 위해 나머지 부분을 간략하게 표시합니다.
    // 실제로는 이전 코드의 버퍼 요청, mmap, SDL 초기화, 메인 루프, 정리 코드가 모두 필요합니다.
    reqbuf.count = BUFFER_COUNT;
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &reqbuf) == -1) { perror("VIDIOC_REQBUFS 실패"); return 1; }
    
    // (mmap, QBUF 로직 생략)
    for (unsigned int i=0; i<reqbuf.count; ++i) {
        v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if(ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) { perror("VIDIOC_QUERYBUF 실패"); return 1; }
        // mmap(...);
        if(ioctl(fd, VIDIOC_QBUF, &buf) == -1) { perror("VIDIOC_QBUF 실패"); return 1; }
    }


    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        perror("VIDIOC_STREAMON 실패");
        close(fd);
        return 1;
    }
    
    std::cout << "성공! 스트리밍이 시작되었습니다." << std::endl;
    ioctl(fd, VIDIOC_STREAMOFF, &type);
    close(fd);

    return 0; // 테스트를 위해 SDL 부분 없이 스트림 시작/종료만 확인
}
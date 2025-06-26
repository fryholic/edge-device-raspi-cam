#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 카메라 장치 번호. 보통 0번이 기본 카메라(/dev/video0) 입니다.
    const int DEVICE_ID = 0;

    // VideoCapture 객체 생성
    cv::VideoCapture cap(DEVICE_ID, cv::CAP_V4L2);

    // 카메라가 정상적으로 열렸는지 확인
    if (!cap.isOpened()) {
        std::cerr << "오류: 카메라를 열 수 없습니다 (장치 ID: " << DEVICE_ID << ")" << std::endl;
        return -1;
    }

    std::cout << "카메라를 성공적으로 열었습니다. 영상을 실시간으로 표시합니다." << std::endl;
    std::cout << "종료하려면 'q' 또는 'ESC' 키를 누르세요." << std::endl;

    // 프레임을 저장할 Mat 객체 생성
    cv::Mat frame;

    // 무한 루프
    for (;;) {
        // 카메라에서 한 프레임을 읽어와 frame 객체에 저장
        cap >> frame;

        // 프레임이 비어있으면(읽기 실패) 루프 종료
        if (frame.empty()) {
            std::cerr << "오류: 빈 프레임이 감지되었습니다." << std::endl;
            break;
        }

        // "Live Video" 라는 이름의 윈도우에 프레임 표시
        cv::imshow("Live Video", frame);

        // 1ms 동안 키 입력을 대기. 'q' 또는 ESC(ASCII 27) 키가 입력되면 루프 종료
        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 27) {
            break;
        }
    }
    
    // VideoCapture 객체는 소멸자에서 자동으로 해제되지만 명시적으로 해제할 수도 있습니다.
    // cap.release();
    cv::destroyAllWindows();
    
    std::cout << "프로그램을 종료합니다." << std::endl;

    return 0;
}
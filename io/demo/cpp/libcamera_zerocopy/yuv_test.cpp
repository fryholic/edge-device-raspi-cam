#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <opencv2/opencv.hpp>

// YUV 파일에서 세 개의 평면을 추출하여 RGB로 변환 후 저장하는 프로그램
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "사용법: " << argv[0] << " <yuv_file_path>" << std::endl;
        return -1;
    }
    
    const char* filename = argv[1];
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return -1;
    }
    
    // 파일 크기 계산
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 1920x1080 YUV420P 형식 가정
    const int width = 1920;
    const int height = 1080;
    
    // 예상 크기 확인
    size_t ySize = width * height;
    size_t uvSize = (width / 2) * (height / 2);
    size_t expectedSize = ySize + uvSize * 2;
    
    if (fileSize != expectedSize) {
        std::cerr << "파일 크기가 예상과 다릅니다: " << fileSize << " != " << expectedSize << std::endl;
        return -1;
    }
    
    // 버퍼 생성
    std::vector<uint8_t> yuvData(fileSize);
    file.read(reinterpret_cast<char*>(yuvData.data()), fileSize);
    file.close();
    
    std::cout << "YUV 파일 로드 완료: " << filename << " (" << fileSize << " bytes)" << std::endl;
    
    // 세 개의 채널로 분리
    cv::Mat y(height, width, CV_8UC1, yuvData.data());
    cv::Mat u(height/2, width/2, CV_8UC1, yuvData.data() + ySize);
    cv::Mat v(height/2, width/2, CV_8UC1, yuvData.data() + ySize + uvSize);
    
    // 일부 U, V 값 출력 (디버깅용)
    std::cout << "U 채널 샘플 (10x10):" << std::endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << static_cast<int>(u.at<uint8_t>(i, j)) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "V 채널 샘플 (10x10):" << std::endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << static_cast<int>(v.at<uint8_t>(i, j)) << " ";
        }
        std::cout << std::endl;
    }
    
    // OpenCV에서는 YUV를 BGR로 변환
    cv::Mat yuv[3] = {y, u, v};
    cv::Mat bgr;
    
    // 기본 YUV420p -> BGR 변환 (기본 방식)
    cv::cvtColor(y, bgr, cv::COLOR_YUV2BGR_I420);
    cv::imwrite("output_default.png", bgr);
    std::cout << "기본 변환 이미지 저장: output_default.png" << std::endl;
    
    // U와 V 위치를 서로 바꾼 YUV -> BGR 변환 (U,V 교체 테스트)
    cv::Mat yvu[3] = {y, v, u}; // U와 V 교체
    cv::Mat bgr_swapped;
    cv::cvtColor(yvu[0], bgr_swapped, cv::COLOR_YUV2BGR_YV12);
    cv::imwrite("output_swapped.png", bgr_swapped);
    std::cout << "U,V 교체 변환 이미지 저장: output_swapped.png" << std::endl;
    
    // U와 V에 중간값(128) 설정 (그레이스케일 테스트)
    cv::Mat u_neutral(height/2, width/2, CV_8UC1, cv::Scalar(128));
    cv::Mat v_neutral(height/2, width/2, CV_8UC1, cv::Scalar(128));
    cv::Mat yuv_gray[3] = {y, u_neutral, v_neutral};
    cv::Mat bgr_gray;
    cv::cvtColor(y, bgr_gray, cv::COLOR_YUV2BGR_I420);
    cv::imwrite("output_gray.png", bgr_gray);
    std::cout << "그레이스케일 변환 이미지 저장: output_gray.png" << std::endl;
    
    return 0;
}

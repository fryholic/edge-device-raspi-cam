#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// YUV 파일 분석 및 다양한 형식으로 변환 시도
int main(int argc, char** argv) {
    if (argc < 2) {
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
    
    // YUV420 형식에 필요한 크기 계산
    size_t ySize = width * height;
    size_t uvSize = (width / 2) * (height / 2);
    size_t expectedSize = ySize + uvSize * 2;
    
    std::cout << "파일 분석: " << filename << std::endl;
    std::cout << "파일 크기: " << fileSize << " 바이트" << std::endl;
    std::cout << "예상 크기 (YUV420P): " << expectedSize << " 바이트" << std::endl;
    
    if (fileSize != expectedSize) {
        std::cout << "경고: 파일 크기가 1920x1080 YUV420P 형식과 일치하지 않습니다." << std::endl;
    }
    
    // 파일 읽기
    std::vector<uint8_t> yuv_data(fileSize);
    file.read(reinterpret_cast<char*>(yuv_data.data()), fileSize);
    file.close();
    
    // Y, U, V 평면 분리
    uint8_t* y_plane = yuv_data.data();
    uint8_t* u_plane = yuv_data.data() + ySize;
    uint8_t* v_plane = yuv_data.data() + ySize + uvSize;
    
    // 평면 통계 분석
    std::cout << "\n평면 통계 분석:" << std::endl;
    
    // Y 평면 분석
    int y_min = 255, y_max = 0, y_sum = 0;
    for (size_t i = 0; i < ySize; i++) {
        y_min = std::min(y_min, (int)y_plane[i]);
        y_max = std::max(y_max, (int)y_plane[i]);
        y_sum += y_plane[i];
    }
    double y_avg = (double)y_sum / ySize;
    
    std::cout << "Y 평면 (Luma): 최소=" << y_min << ", 최대=" << y_max 
              << ", 평균=" << y_avg << std::endl;
    
    // U 평면 분석
    int u_min = 255, u_max = 0, u_sum = 0;
    for (size_t i = 0; i < uvSize; i++) {
        u_min = std::min(u_min, (int)u_plane[i]);
        u_max = std::max(u_max, (int)u_plane[i]);
        u_sum += u_plane[i];
    }
    double u_avg = (double)u_sum / uvSize;
    
    std::cout << "U 평면 (Cb): 최소=" << u_min << ", 최대=" << u_max 
              << ", 평균=" << u_avg << std::endl;
    
    // V 평면 분석
    int v_min = 255, v_max = 0, v_sum = 0;
    for (size_t i = 0; i < uvSize; i++) {
        v_min = std::min(v_min, (int)v_plane[i]);
        v_max = std::max(v_max, (int)v_plane[i]);
        v_sum += v_plane[i];
    }
    double v_avg = (double)v_sum / uvSize;
    
    std::cout << "V 평면 (Cr): 최소=" << v_min << ", 최대=" << v_max 
              << ", 평균=" << v_avg << std::endl;
    
    // U와 V 평면이 유사한지 확인 (그레이스케일 문제 진단)
    int diff_count = 0;
    double diff_sum = 0;
    for (size_t i = 0; i < uvSize; i++) {
        int diff = std::abs((int)u_plane[i] - (int)v_plane[i]);
        if (diff > 5) { // 약간의 차이는 허용
            diff_count++;
            diff_sum += diff;
        }
    }
    double diff_percentage = (double)diff_count / uvSize * 100;
    double avg_diff = diff_count > 0 ? diff_sum / diff_count : 0;
    
    std::cout << "U/V 차이 분석: " << diff_count << " 픽셀 (" << diff_percentage << "%)이 " 
              << "유의미한 차이를 보임, 평균 차이=" << avg_diff << std::endl;
    
    if (diff_percentage < 10) {
        std::cout << "경고: U와 V 평면이 매우 유사함 - 색상이 제대로 추출되지 않았을 수 있음" << std::endl;
    }
    
    // 첫 번째 행의 픽셀 값 출력
    std::cout << "\nY 평면 첫 행 (16 픽셀):" << std::endl;
    for (int i = 0; i < 16; i++) {
        std::cout << (int)y_plane[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "U 평면 첫 행 (8 픽셀):" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << (int)u_plane[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "V 평면 첫 행 (8 픽셀):" << std::endl;
    for (int i = 0; i < 8; i++) {
        std::cout << (int)v_plane[i] << " ";
    }
    std::cout << std::endl;
    
    // 파일명에서 확장자 제외한 이름 가져오기
    std::string basename(filename);
    size_t pos = basename.rfind('.');
    if (pos != std::string::npos) {
        basename = basename.substr(0, pos);
    }
    
    // 다양한 변환 시도
    // 1. 기본 YUV420P (I420)
    cv::Mat yuv_i420(height * 3 / 2, width, CV_8UC1, yuv_data.data());
    cv::Mat bgr_i420;
    cv::cvtColor(yuv_i420, bgr_i420, cv::COLOR_YUV2BGR_I420);
    cv::imwrite(basename + "_i420.png", bgr_i420);
    
    // 2. YV12 (U와 V가 바뀐 형식)
    // YV12는 Y 다음에 V, 그리고 U 순서
    std::vector<uint8_t> yv12_data(fileSize);
    std::memcpy(yv12_data.data(), y_plane, ySize); // Y는 그대로
    std::memcpy(yv12_data.data() + ySize, v_plane, uvSize); // V를 두 번째 위치에
    std::memcpy(yv12_data.data() + ySize + uvSize, u_plane, uvSize); // U를 세 번째 위치에
    
    cv::Mat yuv_yv12(height * 3 / 2, width, CV_8UC1, yv12_data.data());
    cv::Mat bgr_yv12;
    cv::cvtColor(yuv_yv12, bgr_yv12, cv::COLOR_YUV2BGR_YV12);
    cv::imwrite(basename + "_yv12.png", bgr_yv12);
    
    // 3. NV12 (Y 다음에 UV가 인터리브된 형식)
    std::vector<uint8_t> nv12_data(ySize + uvSize * 2);
    std::memcpy(nv12_data.data(), y_plane, ySize); // Y는 그대로
    
    // UV 인터리브 생성 (UVUVUV...)
    for (size_t i = 0; i < uvSize; i++) {
        nv12_data[ySize + i * 2] = u_plane[i];
        nv12_data[ySize + i * 2 + 1] = v_plane[i];
    }
    
    cv::Mat yuv_nv12(height * 3 / 2, width, CV_8UC1, nv12_data.data());
    cv::Mat bgr_nv12;
    cv::cvtColor(yuv_nv12, bgr_nv12, cv::COLOR_YUV2BGR_NV12);
    cv::imwrite(basename + "_nv12.png", bgr_nv12);
    
    // 4. NV21 (Y 다음에 VU가 인터리브된 형식)
    std::vector<uint8_t> nv21_data(ySize + uvSize * 2);
    std::memcpy(nv21_data.data(), y_plane, ySize); // Y는 그대로
    
    // VU 인터리브 생성 (VUVUVU...)
    for (size_t i = 0; i < uvSize; i++) {
        nv21_data[ySize + i * 2] = v_plane[i];
        nv21_data[ySize + i * 2 + 1] = u_plane[i];
    }
    
    cv::Mat yuv_nv21(height * 3 / 2, width, CV_8UC1, nv21_data.data());
    cv::Mat bgr_nv21;
    cv::cvtColor(yuv_nv21, bgr_nv21, cv::COLOR_YUV2BGR_NV21);
    cv::imwrite(basename + "_nv21.png", bgr_nv21);
    
    std::cout << "\n다양한 YUV 형식으로 변환 완료:" << std::endl;
    std::cout << "1. I420: " << basename << "_i420.png" << std::endl;
    std::cout << "2. YV12: " << basename << "_yv12.png" << std::endl;
    std::cout << "3. NV12: " << basename << "_nv12.png" << std::endl;
    std::cout << "4. NV21: " << basename << "_nv21.png" << std::endl;
    
    return 0;
}

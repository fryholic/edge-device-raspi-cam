# Python YOLOv5 객체 추적 C++ 포팅 가이드

## 1. 소개

이 문서는 Python 기반의 객체 탐지 및 추적 애플리케이션을 C++로 포팅하는 과정을 상세히 설명합니다. 원본 Python 애플리케이션은 객체 탐지를 위해 YOLOv5를, 추적을 위해 SORT 알고리즘을 사용합니다. C++ 버전은 고성능 추론을 위해 Intel OpenVINO™ 툴킷을, 비디오 처리를 위해 OpenCV를 활용하며, 핵심 SORT 추적 로직은 C++로 직접 구현했습니다.

최종 C++ 프로젝트는 두 가지 주요 실행 파일을 제공합니다:
- `yolo_visual`: 객체 추적을 수행하고 비디오 스트림에 결과(바운딩 박스, ID)를 시각화합니다.
- `yolo_console`: 객체 추적을 수행하고 그래픽 인터페이스 없이 터미널에 결과를 출력합니다.

## 2. 사전 요구 사항

시작하기 전에 다음 소프트웨어가 설치되어 있는지 확인하십시오:

- **C++ 컴파일러**: 최신 C++ 컴파일러 (예: GCC, Clang, MSVC).
- **CMake**: 버전 3.10 이상.
- **OpenCV**: OpenCV 라이브러리, 버전 4.x 권장.
- **Intel OpenVINO™ 툴킷**: OpenVINO 개발 툴킷.

## 3. 포팅 과정

포팅 과정은 크게 네 단계로 나뉩니다.

### 1단계: YOLOv5 모델을 OpenVINO IR 형식으로 변환

원본 프로젝트는 PyTorch 모델(`.pt`)을 사용합니다. 이 모델을 OpenVINO™ 런타임과 함께 사용하려면, 먼저 중간 표현(Intermediate Representation, IR) 형식으로 변환해야 합니다. IR 형식은 네트워크 토폴로지를 설명하는 `.xml` 파일과 가중치 및 편향을 포함하는 `.bin` 파일로 구성됩니다.

이 변환은 OpenVINO™ 모델 변환기(`ovc`) 도구를 사용하여 수행됩니다. `yolov5n.pt` 모델의 경우 명령어는 다음과 같습니다:

```bash
ovc export --model yolov5n.pt --input_shape [1,3,640,640] --model_name yolov5n
```

이 명령어는 C++ 애플리케이션에서 사용할 `yolov5n.xml`과 `yolov5n.bin` 파일을 생성합니다.

### 2단계: SORT 알고리즘을 C++로 포팅

추적 로직의 핵심은 `sort.py` 파일에 있었습니다. 이 파일을 C++로 포팅하기 위해 `Sort` 클래스를 만들었습니다.

#### `Sort.h`
이 헤더 파일은 C++ SORT 구현의 구조를 정의합니다.

- **`TrackingBox` 구조체**: 탐지 및 추적 결과를 담기 위한 간단한 데이터 구조로, 바운딩 박스(`cv::Rect_<float>`), 객체 ID, 클래스 ID를 포함합니다.
- **`KalmanBoxTracker` 클래스**: 단일 추적 객체를 나타냅니다. OpenCV의 `cv::KalmanFilter`를 캡슐화하여 객체의 상태(중심 좌표, 면적, 종횡비)를 예측하고 업데이트합니다.
- **`Sort` 클래스**: 추적 프로세스를 총괄하는 메인 클래스입니다. `KalmanBoxTracker` 인스턴스 컬렉션을 관리하고, 각 프레임의 새로운 탐지 결과를 처리하기 위한 `update` 메서드를 구현합니다.

#### `Sort.cpp`
이 파일은 `Sort.h`에 정의된 클래스의 구현을 포함합니다.

- **`KalmanBoxTracker`**: 생성자는 원본 Python 버전과 유사한 모션 및 측정 모델을 사용하여 7x4 칼만 필터(상태 변수 7개, 측정 변수 4개)를 초기화합니다. `predict()` 및 `update()` 메서드는 해당 `cv::KalmanFilter` 호출을 래핑합니다.
- **`Sort::update`**: 핵심 함수입니다. 각 프레임에 대해 다음을 수행합니다:
    1.  **예측**: 모든 기존 추적기에 대해 `predict()`를 호출합니다.
    2.  **연결**: 새로운 탐지를 기존 추적기와 매칭합니다. 이는 모든 탐지와 예측된 추적기 위치 사이의 IoU(Intersection over Union) 행렬을 계산하여 수행됩니다. 그리디 할당 전략이 사용됩니다: 각 탐지는 특정 임계값(`iou_threshold`)을 초과하는 가장 높은 IoU를 가진 추적기에 매칭됩니다.
    3.  **업데이트**: 매칭된 추적기는 `KalmanBoxTracker::update` 메서드를 사용하여 해당 탐지 결과로 업데이트됩니다.
    4.  **생명 주기 관리**: 매칭되지 않은 탐지는 새로운 `KalmanBoxTracker` 인스턴스를 생성하는 데 사용됩니다. 특정 프레임 수(`max_age`) 동안 매칭되지 않은 추적기는 제거됩니다.

### 3단계: 메인 C++ 애플리케이션 생성

시각적 출력과 콘솔 전용 출력을 모두 제공하기 위해 두 개의 개별 메인 파일을 만들었습니다.

#### `main_visual.cpp`
이 애플리케이션은 추적을 수행하고 출력을 표시합니다.
1.  **초기화**: OpenVINO™ Core를 초기화하고, `.xml` 모델을 읽고, 전처리 파이프라인을 설정합니다. 또한 `cv::VideoCapture`를 사용하여 입력 비디오 파일을 엽니다.
2.  **프레임 루프**: 비디오에서 프레임을 하나씩 읽습니다.
3.  **추론**: 각 프레임은 전처리(크기 조정 및 블롭 변환)를 거쳐 OpenVINO™ 추론 엔진에 입력됩니다.
4.  **후처리**: 모델의 원시 출력은 바운딩 박스, 신뢰도 점수, 클래스 ID를 추출하기 위해 처리됩니다. 겹치는 박스를 필터링하기 위해 비최대 억제(NMS)가 적용됩니다.
5.  **추적**: 필터링된 탐지 결과는 `Sort::update` 메서드에 전달되어 현재 추적 중인 객체 목록을 반환받습니다.
6.  **시각화**: 추적된 객체의 바운딩 박스와 ID가 OpenCV 함수(`cv::rectangle`, `cv::putText`)를 사용하여 프레임에 그려집니다. 최종 프레임은 창에 표시됩니다.

#### `main_console.cpp`
이 버전은 추론 및 추적 로직에서 `main_visual.cpp`와 거의 동일합니다. 주요 차이점은 프레임에 그림을 그리고 표시하는 대신, 각 프레임의 추적 결과를 콘솔에 출력한다는 것입니다. 각 객체에 대한 출력에는 ID, 클래스, 바운딩 박스 좌표가 포함됩니다.

### 4단계: CMake로 빌드 시스템 생성

C++ 소스 파일을 컴파일하고 OpenVINO™ 및 OpenCV에 연결하기 위해 `CMakeLists.txt` 파일을 만들었습니다.

- 프로젝트 이름과 필요한 C++ 표준을 설정합니다.
- `find_package`를 사용하여 시스템에서 OpenVINO™ 및 OpenCV 라이브러리를 찾습니다.
- SORT 라이브러리(`Sort.cpp`)의 소스 파일을 정의합니다.
- 각각의 `main_*.cpp` 파일과 SORT 소스로부터 `yolo_visual`과 `yolo_console`이라는 두 개의 개별 실행 파일을 생성합니다.
- 두 실행 파일을 OpenVINO™ 및 OpenCV 라이브러리에 연결합니다.

## 4. 빌드 및 실행 방법

C++ 프로젝트를 컴파일하고 실행하려면 다음 단계를 따르십시오.

1.  **빌드 디렉토리 생성**: 소스 외부에서 프로젝트를 빌드하는 것이 가장 좋습니다.
    ```bash
    mkdir build
    cd build
    ```

2.  **CMake를 실행하여 프로젝트 구성**: CMake에 OpenCV 및 OpenVINO 설치 위치를 알려줍니다.
    ```bash
    # 아래 경로를 시스템의 실제 위치로 바꾸십시오
    cmake .. -DOpenCV_DIR=/path/to/opencv/build -DOpenVINO_DIR=/path/to/openvino_sdk/runtime/cmake
    ```

3.  **프로젝트 빌드**:
    ```bash
    cmake --build .
    ```
    Linux에서는 `make`를 대신 사용할 수 있습니다.

4.  **실행 파일 실행**: 빌드가 성공하면 `build` 디렉토리에 실행 파일이 생성됩니다. `build` 디렉토리 내에서 모델 경로와 비디오 파일을 인자로 전달하여 실행합니다.

    ```bash
    # 시각화 버전 실행:
    ./yolo_visual ../yolov5n.xml /path/to/your/video.mp4

    # 콘솔 전용 버전 실행:
    ./yolo_console ../yolov5n.xml /path/to/your/video.mp4
    ```
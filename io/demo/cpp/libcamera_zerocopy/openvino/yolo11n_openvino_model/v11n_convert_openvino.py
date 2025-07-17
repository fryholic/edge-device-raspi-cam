from ultralytics import YOLO

# YOLOv11n 모델 로드 (자동으로 다운로드 됩니다)
model = YOLO('yolo11n.pt')

# OpenVINO IR 형식으로 변환
model.export(
    format='openvino',
    imgsz=640,
    half=True
)
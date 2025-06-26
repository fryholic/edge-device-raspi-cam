#!/bin/bash

# V4L2 카메라 캡처를 위한 설정 스크립트

echo "V4L2 카메라 준비 중..."

# PipeWire 서비스 중지
echo "PipeWire 서비스 중지 중..."
systemctl --user stop pipewire.service wireplumber.service pipewire.socket

# 카메라 해제 확인
sleep 1

# 센서 포맷 설정
echo "센서 포맷 설정 중 (1920x1080 SRGGB10)..."
#v4l2-ctl -d /dev/v4l-subdev0 --set-subdev-fmt pad=0,width=1920,height=1080,code=0x300f
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1920,height=1080,pixelformat=RGGB

# 설정 확인
echo "현재 센서 설정:"
#v4l2-ctl -d /dev/v4l-subdev0 --get-subdev-fmt
v4l2-ctl -d /dev/video0 --get-fmt-video

echo "V4L2 카메라 준비 완료!"
echo "이제 v4l2_capture_cpp를 실행할 수 있습니다."
echo ""
echo "사용 후 PipeWire 재시작하려면:"
echo "systemctl --user start pipewire.service wireplumber.service"

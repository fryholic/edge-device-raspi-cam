#!/bin/bash
# RGB888 파일을 PNG로 변환하는 스크립트
for file in frame_rgb_*_*.rgb; do
  if [ -f "$file" ]; then
    # 파일 이름에서 해상도 추출
    resolution=$(echo $file | grep -oP "\d+x\d+" | head -1)
    if [ -z "$resolution" ]; then
      echo "해상도를 찾을 수 없습니다: $file"
      continue
    fi
    echo "$file ($resolution) 변환 중..."
    ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size $resolution -i "$file" "${file%.rgb}.png"
  fi
done

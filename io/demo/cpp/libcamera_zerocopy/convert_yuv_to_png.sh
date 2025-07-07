#!/bin/bash

# YUV to PNG 일괄 변환 스크립트
# 사용법: ./convert_yuv_to_png.sh [pattern]
# 예시: ./convert_yuv_to_png.sh "*1920x1080*"

echo "=== YUV to PNG 일괄 변환 시작 ==="

# 패턴이 주어지지 않으면 모든 YUV 파일 처리
PATTERN=${1:-"*.yuv"}

# 변환된 파일 개수 카운터
CONVERTED=0
FAILED=0

# PNG 출력 디렉토리 생성
mkdir -p png_output

# YUV 파일들 처리
for yuv_file in $PATTERN; do
    if [[ -f "$yuv_file" ]]; then
        echo "처리 중: $yuv_file"
        
        # 파일명에서 해상도 추출
        if [[ $yuv_file =~ ([0-9]+)x([0-9]+) ]]; then
            width=${BASH_REMATCH[1]}
            height=${BASH_REMATCH[2]}
            
            # 출력 파일명 생성
            base_name=$(basename "$yuv_file" .yuv)
            output_file="png_output/${base_name}.png"
            
            echo "  해상도: ${width}x${height}"
            echo "  출력: $output_file"
            
            # FFmpeg 변환 실행
            if ffmpeg -loglevel quiet -f rawvideo -pixel_format yuv420p -video_size ${width}x${height} \
                      -i "$yuv_file" -frames:v 1 -update 1 -y "$output_file" 2>/dev/null; then
                echo "  ✅ 변환 성공"
                ((CONVERTED++))
            else
                echo "  ❌ 변환 실패"
                ((FAILED++))
            fi
        else
            echo "  ⚠️  해상도를 파일명에서 추출할 수 없음: $yuv_file"
            ((FAILED++))
        fi
        echo ""
    fi
done

echo "=== 변환 완료 ==="
echo "성공: $CONVERTED 개"
echo "실패: $FAILED 개"
echo "출력 디렉토리: png_output/"

if [[ $CONVERTED -gt 0 ]]; then
    echo ""
    echo "생성된 PNG 파일들:"
    ls -la png_output/*.png 2>/dev/null | head -10
    
    if [[ $(ls png_output/*.png 2>/dev/null | wc -l) -gt 10 ]]; then
        echo "... (더 많은 파일들)"
    fi
fi

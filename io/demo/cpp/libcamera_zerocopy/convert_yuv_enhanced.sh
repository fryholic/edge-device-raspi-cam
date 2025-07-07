#!/bin/bash

# YUV/NV12 to PNG 향상된 변환 스크립트
# 사용법: ./convert_yuv_enhanced.sh [pattern]
# 예시: ./convert_yuv_enhanced.sh "*yuv420p*"

echo "=== YUV/NV12 to PNG 향상된 변환 시작 ==="

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
            
            # 포맷 결정 (파일명에 따라)
            format="yuv420p"
            if [[ $yuv_file == *"nv12"* ]]; then
                format="nv12"
                echo "  포맷: NV12 감지"
            else
                echo "  포맷: YUV420P 가정"
            fi
            
            # FFmpeg 변환 실행
            if ffmpeg -loglevel warning -f rawvideo -pixel_format $format -video_size ${width}x${height} \
                      -i "$yuv_file" -frames:v 1 -update 1 -y "$output_file" 2>&1; then
                echo "  ✅ 변환 성공"
                ((CONVERTED++))
            else
                echo "  ❌ 변환 실패"
                # 실패 시 다른 포맷으로 재시도
                if [[ $format == "yuv420p" ]]; then
                    echo "  💡 YUV420P 실패, NV12로 재시도..."
                    if ffmpeg -loglevel warning -f rawvideo -pixel_format nv12 -video_size ${width}x${height} \
                              -i "$yuv_file" -frames:v 1 -update 1 -y "$output_file" 2>&1; then
                        echo "  ✅ NV12 변환 성공"
                        ((CONVERTED++))
                        continue
                    fi
                elif [[ $format == "nv12" ]]; then
                    echo "  💡 NV12 실패, YUV420P로 재시도..."
                    if ffmpeg -loglevel warning -f rawvideo -pixel_format yuv420p -video_size ${width}x${height} \
                              -i "$yuv_file" -frames:v 1 -update 1 -y "$output_file" 2>&1; then
                        echo "  ✅ YUV420P 변환 성공"
                        ((CONVERTED++))
                        continue
                    fi
                fi
                ((FAILED++))
            fi
            
            # 성공한 경우 다양한 출력 테스트 생성
            if [ -f "$output_file" ]; then
                # 이미지 색상 상태 분석
                echo "  🔍 이미지 분석:"
                average_color=$(convert "$output_file" -resize 1x1\! -format "%[pixel:u]" info:-)
                echo "    평균 색상: $average_color"
                
                # RGB 색상 채널 추출 (디버깅용)
                convert "$output_file" -channel R -separate "png_output/${base_name}_r.png"
                convert "$output_file" -channel G -separate "png_output/${base_name}_g.png"
                convert "$output_file" -channel B -separate "png_output/${base_name}_b.png"
                echo "    RGB 채널 분리 이미지 생성됨"
            fi
            
        else
            echo "  ⚠️  해상도를 파일명에서 추출할 수 없음: $yuv_file"
            ((FAILED++))
        fi
        echo ""
    fi
done

# 녹색 테스트 이미지 생성 (비교용)
echo "📊 비교용 테스트 이미지 생성..."
convert -size 100x100 xc:"rgb(0,255,0)" png_output/test_green.png
convert -size 100x100 xc:"rgb(255,0,0)" png_output/test_red.png
convert -size 100x100 xc:"rgb(0,0,255)" png_output/test_blue.png
convert -size 100x100 xc:"rgb(128,128,128)" png_output/test_gray.png

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

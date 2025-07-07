#!/bin/bash

# 향상된 YUV to PNG 변환 스크립트
# 사용법: ./convert_yuv_all.sh [pattern]

echo "=== 다양한 형식으로 YUV 파일 변환 시작 ==="

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
            
            echo "  해상도: ${width}x${height}"
            
            # 1. I420/YUV420P 형식으로 변환 시도
            output_i420="png_output/${base_name}_i420.png"
            echo "  출력(I420): $output_i420"
            if ffmpeg -loglevel error -f rawvideo -pixel_format yuv420p -video_size ${width}x${height} \
                      -i "$yuv_file" -frames:v 1 -y "$output_i420"; then
                echo "  ✅ I420 변환 성공"
                ((CONVERTED++))
            else
                echo "  ❌ I420 변환 실패"
                ((FAILED++))
            fi
            
            # 2. YV12 형식으로 변환 시도 (U,V 채널이 바뀐 형식)
            output_yv12="png_output/${base_name}_yv12.png"
            echo "  출력(YV12): $output_yv12"
            if ffmpeg -loglevel error -f rawvideo -pixel_format yv12 -video_size ${width}x${height} \
                      -i "$yuv_file" -frames:v 1 -y "$output_yv12"; then
                echo "  ✅ YV12 변환 성공"
                ((CONVERTED++))
            else
                echo "  ❌ YV12 변환 실패"
                ((FAILED++))
            fi
            
            # 3. NV12 형식으로 변환 시도
            # 주의: NV12 형식은 YUV420P와 인터리브 구조가 다름
            # 원본이 YUV420P라면 그대로 사용할 수 없음
            # 실험적 테스트를 위해 포함
            output_nv12="png_output/${base_name}_nv12.png"
            echo "  출력(NV12): $output_nv12"
            if ffmpeg -loglevel error -f rawvideo -pixel_format nv12 -video_size ${width}x${height} \
                      -i "$yuv_file" -frames:v 1 -y "$output_nv12"; then
                echo "  ✅ NV12 변환 성공"
                ((CONVERTED++))
            else
                echo "  ❌ NV12 변환 실패"
                ((FAILED++))
            fi
            
            # 4. NV21 형식으로 변환 시도
            # 주의: NV21도 인터리브 구조임
            # 실험적 테스트를 위해 포함
            output_nv21="png_output/${base_name}_nv21.png"
            echo "  출력(NV21): $output_nv21"
            if ffmpeg -loglevel error -f rawvideo -pixel_format nv21 -video_size ${width}x${height} \
                      -i "$yuv_file" -frames:v 1 -y "$output_nv21"; then
                echo "  ✅ NV21 변환 성공"
                ((CONVERTED++))
            else
                echo "  ❌ NV21 변환 실패"
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

// V4L2 RG10 포맷 디코딩 함수 (정확한 구현)
void unpack_v4l2_rg10(const uint8_t* packed, uint16_t* unpacked, int width, int height) {
    // V4L2 RG10: 4픽셀을 5바이트에 패킹
    // 각 픽셀은 10비트, 4픽셀 = 40비트 = 5바이트
    // 패킹 방식: [P0_8MSB][P1_8MSB][P2_8MSB][P3_8MSB][P0_2LSB|P1_2LSB|P2_2LSB|P3_2LSB]
    
    int total_pixels = width * height;
    int packed_idx = 0;
    
    for (int i = 0; i < total_pixels; i += 4) {
        if (i + 3 < total_pixels && packed_idx + 4 < (total_pixels * 5) / 4) {
            // 5바이트 읽기
            uint8_t b0 = packed[packed_idx];     // P0 상위 8비트
            uint8_t b1 = packed[packed_idx + 1]; // P1 상위 8비트  
            uint8_t b2 = packed[packed_idx + 2]; // P2 상위 8비트
            uint8_t b3 = packed[packed_idx + 3]; // P3 상위 8비트
            uint8_t b4 = packed[packed_idx + 4]; // 4픽셀의 하위 2비트들
            
            // 10비트 픽셀 재구성
            unpacked[i]     = (b0 << 2) | ((b4 >> 0) & 0x03);
            unpacked[i + 1] = (b1 << 2) | ((b4 >> 2) & 0x03);
            unpacked[i + 2] = (b2 << 2) | ((b4 >> 4) & 0x03);
            unpacked[i + 3] = (b3 << 2) | ((b4 >> 6) & 0x03);
            
            packed_idx += 5;
        }
    }
}

// 대안: 간단한 16비트 방식 (각 픽셀이 2바이트로 저장된 경우)
void unpack_simple_16bit(const uint8_t* packed, uint16_t* unpacked, int width, int height) {
    uint16_t* src16 = (uint16_t*)packed;
    for (int i = 0; i < width * height; i++) {
        // 리틀 엔디안에서 10비트 추출
        unpacked[i] = src16[i] & 0x3FF; // 하위 10비트
    }
}

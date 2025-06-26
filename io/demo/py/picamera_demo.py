import cv2

DEVICE_ID = 0
API_PREFERENCE = cv2.CAP_GSTREAMER

print("--- 프로그램 시작 ---")
cap = cv2.VideoCapture(DEVICE_ID, API_PREFERENCE)

if not cap.isOpened():
    print("!!! 오류: cap.isOpened()가 False를 반환했습니다.")
else:
    print(">>> 성공: cap.isOpened()가 True를 반환했습니다.")
    
    # --- 수동 설정 부분 ---
    # 가장 표준적인 640x480 해상도를 먼저 시도합니다.
    print(">>> 해상도를 640x480으로 수동 설정합니다...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 비디오 인코딩 포맷을 MJPG로 설정 시도 (호환성이 좋음)
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    # ----------------------

    # 설정이 적용되었는지 확인
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f">>> 현재 설정된 해상도: {int(width)}x{int(height)}")

    print("--- 캡처 시작 ---")
    while True:
        ret, frame = cap.read()

        if not ret:
            print("!!! 오류: 프레임을 읽지 못했습니다. 1초 후 재시도...")
            cv2.waitKey(1000) # 잠시 대기 후 다시 시도
            continue

        cv2.imshow('Manual Set Test', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n--- 루프 종료 ---")
    cap.release()
    cv2.destroyAllWindows()

print("--- 프로그램 종료 ---")
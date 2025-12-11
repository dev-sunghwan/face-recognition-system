# camera/test_opencv_insightface.py

import os
import sys

# ⭐ RTSP TCP transport 설정 (가장 먼저!)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|rtsp_flags;prefer_tcp'

import cv2
import argparse
import numpy as np
from insightface.app import FaceAnalysis


def align_face(img, landmarks_5):
    """ArcFace 스타일 얼굴 정렬 (112x112)"""
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    dst = landmarks_5.astype(np.float32)
    M = cv2.estimateAffinePartial2D(dst, src)[0]
    
    if M is None:
        return None
        
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return aligned


def main():
    parser = argparse.ArgumentParser(description="Face Recognition - OpenCV + InsightFace")
    parser.add_argument("--ip", required=True, help="Camera IP (can include port: IP:PORT)")
    parser.add_argument("--user", default="admin", help="Username")
    parser.add_argument("--password", default="Sunap1!!", help="Password")
    parser.add_argument("--stride", type=int, default=3, help="Detection every N frames")
    parser.add_argument("--display_width", type=int, default=1280)
    parser.add_argument("--display_height", type=int, default=720)
    parser.add_argument("--headless", action="store_true", help="Run without display")
    args = parser.parse_args()

    # InsightFace 초기화
    print("[INFO] Initializing InsightFace...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    print("[INFO] InsightFace (buffalo_l) loaded")

    # IP:PORT 형식 처리
    camera_ip = args.ip
    if ':' in camera_ip:
        # IP와 포트 분리
        ip_parts = camera_ip.split(':')
        host = ip_parts[0]
        port = ip_parts[1]
        rtsp_url = f"rtsp://{args.user}:{args.password}@{host}:{port}/profile2/media.smp"
    else:
        # 포트 없으면 기본 RTSP 포트(554) 사용
        rtsp_url = f"rtsp://{args.user}:{args.password}@{camera_ip}/profile2/media.smp"
    
    print(f"[INFO] Connecting to: {rtsp_url}")
    print("[INFO] Using TCP transport for RTSP")

    # OpenCV VideoCapture with FFmpeg backend
    # TCP transport는 이미 환경변수로 설정됨
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # 타임아웃 및 버퍼 설정
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 연결 확인
    if not cap.isOpened():
        print("[ERROR] Failed to open RTSP stream")
        print("[ERROR] Possible causes:")
        print("  - Wrong credentials")
        print("  - Camera not accessible from Docker network")
        print("  - Firewall blocking connection")
        return

    print("[INFO] ✅ Stream opened successfully")
    print("[INFO] Attempting to grab first frame...")

    frame_count = 0
    last_faces = []
    face_detected_count = 0
    retry_count = 0
    max_retries = 100  # 100번 재시도
    first_frame_grabbed = False

    while True:
        ret, img = cap.read()
        
        if not ret:
            retry_count += 1
            
            # 첫 프레임을 못 받는 경우
            if not first_frame_grabbed and retry_count > max_retries:
                print(f"[ERROR] Failed to grab first frame after {max_retries} retries")
                print("[ERROR] This usually means:")
                print("  1. Stream format not supported")
                print("  2. Network timeout (check port forwarding)")
                print("  3. Camera requires different RTSP parameters")
                print()
                print("[DEBUG] Try this command on your local machine:")
                print(f"  ffplay -rtsp_transport tcp '{rtsp_url}'")
                break
            
            # 주기적으로 상태 출력
            if retry_count % 10 == 0:
                print(f"[WARNING] Waiting for frame... ({retry_count}/{max_retries})")
            
            continue
        
        # ✅ 첫 프레임 성공!
        if not first_frame_grabbed:
            h, w = img.shape[:2]
            print(f"[INFO] ✅✅✅ First frame grabbed successfully! Resolution: {w}x{h}")
            print(f"[INFO] Took {retry_count} attempts")
            first_frame_grabbed = True
            retry_count = 0

        # InsightFace detection
        if frame_count % args.stride == 0:
            faces = app.get(img)
            last_faces = faces
            if len(faces) > 0:
                face_detected_count += 1
        else:
            faces = last_faces

        frame_count += 1

        aligned_face = None

        for f in faces:
            # Bounding box
            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 5 Landmarks
            lm5 = f.landmark_2d_5
            if lm5 is not None and len(lm5) == 5:
                for (lx, ly) in lm5:
                    cv2.circle(img, (int(lx), int(ly)), 2, (0, 255, 255), -1)

                # Alignment
                aligned_face = align_face(img, lm5)

            # Embedding (선택사항)
            # embedding = f.embedding  # 512-dim vector
            # print(f"[DEBUG] Embedding shape: {embedding.shape}")

            break  # 첫 번째 얼굴만

        # 통계 출력
        if frame_count % 100 == 0:
            print(f"[INFO] Processed {frame_count} frames, {face_detected_count} faces detected")

        # Display (headless 모드가 아닐 때만)
        if not args.headless:
            display_img = cv2.resize(img, (args.display_width, args.display_height))
            cv2.imshow("InsightFace Detection", display_img)

            if aligned_face is not None:
                cv2.imshow("Aligned Face (112x112)", aligned_face)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                print("[INFO] ESC pressed, exiting...")
                break

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    
    print(f"[INFO] Total frames: {frame_count}, Faces detected: {face_detected_count}")


if __name__ == "__main__":
    main()
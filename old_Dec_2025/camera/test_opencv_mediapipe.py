# camera/test_opencv_mediapipe.py

import cv2
import argparse
import numpy as np
import mediapipe as mp

# Mediapipe 초기화
mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

# 5 Landmark Index
FACEMESH_LANDMARK_IDXS = {
    "left_eye": 33,
    "right_eye": 263,
    "nose": 1,
    "mouth_left": 61,
    "mouth_right": 291,
}


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
    parser = argparse.ArgumentParser(description="Face Recognition - OpenCV + Mediapipe")
    parser.add_argument("--ip", required=True, help="Camera IP")
    parser.add_argument("--user", default="admin", help="Username")
    parser.add_argument("--password", default="Sunap1!!", help="Password")
    parser.add_argument("--stride", type=int, default=2, help="Detection every N frames")
    parser.add_argument("--display_width", type=int, default=1280)
    parser.add_argument("--display_height", type=int, default=720)
    parser.add_argument("--headless", action="store_true", help="Run without display")
    args = parser.parse_args()

    # Mediapipe 초기화
    print("[INFO] Initializing Mediapipe...")
    detector = mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    mesh = mp_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[INFO] Mediapipe initialized")

    # RTSP URL 구성
    rtsp_url = f"rtsp://{args.user}:{args.password}@{args.ip}/profile2/media.smp"
    print(f"[INFO] Connecting to: {rtsp_url}")

    # OpenCV VideoCapture로 RTSP 연결
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # 버퍼 설정 (지연 최소화)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 연결 확인
    if not cap.isOpened():
        print("[ERROR] Failed to open RTSP stream")
        print("[ERROR] Check IP, username, password, and network connection")
        return

    print("[INFO] ✅ Stream opened successfully")
    print("[INFO] Press ESC to exit (if display enabled)")

    frame_count = 0
    last_faces = []
    face_detected_count = 0

    while True:
        ret, img = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame, retrying...")
            continue

        h, w = img.shape[:2]

        # 1) Downscale for detection
        small_w, small_h = 640, 360
        small = cv2.resize(img, (small_w, small_h))

        # 2) Detection every N frames
        if frame_count % args.stride == 0:
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            det_result = detector.process(rgb_small)

            faces = []
            if det_result.detections:
                for det in det_result.detections:
                    bbox = det.location_data.relative_bounding_box

                    xs = int(bbox.xmin * small_w)
                    ys = int(bbox.ymin * small_h)
                    xe = int((bbox.xmin + bbox.width) * small_w)
                    ye = int((bbox.ymin + bbox.height) * small_h)

                    # Scale back to original
                    X1 = int(xs * (w / small_w))
                    Y1 = int(ys * (h / small_h))
                    X2 = int(xe * (w / small_w))
                    Y2 = int(ye * (h / small_h))

                    faces.append((X1, Y1, X2, Y2))

            last_faces = faces
        else:
            faces = last_faces

        frame_count += 1

        # 3) FaceMesh Landmark + Alignment
        aligned_preview = None

        if len(faces) > 0:
            face_detected_count += 1
            x1, y1, x2, y2 = faces[0]

            face_roi = img[y1:y2, x1:x2]
            if face_roi.size != 0:
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                mesh_result = mesh.process(face_rgb)

                if mesh_result.multi_face_landmarks:
                    lm = mesh_result.multi_face_landmarks[0].landmark

                    landmarks_5 = []
                    idxs = FACEMESH_LANDMARK_IDXS

                    for key in ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]:
                        i = idxs[key]
                        lx = lm[i].x * (x2 - x1) + x1
                        ly = lm[i].y * (y2 - y1) + y1
                        landmarks_5.append([lx, ly])

                    landmarks_5 = np.array(landmarks_5, dtype=np.float32)
                    aligned_preview = align_face(img, landmarks_5)

                    # Draw landmarks
                    for (lx, ly) in landmarks_5:
                        cv2.circle(img, (int(lx), int(ly)), 3, (0, 255, 255), -1)

        # 4) Draw bounding boxes
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 5) 통계 출력
        if frame_count % 100 == 0:
            print(f"[INFO] Processed {frame_count} frames, {face_detected_count} faces detected")

        # 6) Display (headless 모드가 아닐 때만)
        if not args.headless:
            display_img = cv2.resize(img, (args.display_width, args.display_height))
            cv2.imshow("Face Detection - Mediapipe", display_img)

            if aligned_preview is not None:
                cv2.imshow("Aligned Face (112x112)", aligned_preview)

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
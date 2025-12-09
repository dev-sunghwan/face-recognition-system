# camera/test_camera_local.py

import cv2
import argparse
import numpy as np
import mediapipe as mp

from rtsp_client import RTSPClient
from decoder import H264Decoder
from h264_rtp_parser import H264RTPParser


# 1) Ininitalize Mediapipe
mp_face = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

# 5 Landmark Index for Mediapipe FaceMesh
FACEMESH_LANDMARK_IDXS = {
    "left_eye": 33,
    "right_eye": 263,
    "nose": 1,
    "mouth_left": 61,
    "mouth_right": 291,
}


# 2) Arcface alighment Function
def align_face(img, landmarks_5):
    """
    img: BGR frame
    landmarks_5: (5,2) array
    return: 112 x 112 aligned face
    """

    # ArcFace Reference 5 Points
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    dst = landmarks_5.astype(np.float32)
    M = cv2.estimateAffinePartial2D(dst, src)[0]
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

    return aligned


# 3) Main Streaming Loop
def main():
    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="Hanwha Camera Face Recognition Pipeline (Mediapipe)")
    parser.add_argument("--ip", required=True, help="Camera IP Address")
    parser.add_argument("--user", default="admin", help="Camera Username")
    parser.add_argument("--password", default="Sunap1!!", help="Camera Password")
    parser.add_argument("--stride", type=int, default=2, help="Do detection every N frames")
    parser.add_argument("--display_width", type=int, default=1280)
    parser.add_argument("--display_height", type=int, default=720)
    args = parser.parse_args()

    camera_ip = args.ip
    username = args.user
    password = args.password
    stride = args.stride

    # Initialize Face Detector - Mediapipe
    detector = mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    print("[Info] Mediapipe Face Detection initialized.")

    # Initialize Face Mesh for landmarks - Mediapipe
    mesh = mp_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[Info] Mediapipe Face Mesh initialized.")

    # Construct RTSP URL using Hanwha official format 
    rtsp_url = f"rtsp://{username}:{password}@{camera_ip}/profile2/media.smp"
    print(f"[INFO] Connecting to: {rtsp_url}")

    # Initialize RTSP client + decoder
    client = RTSPClient(rtsp_url)
    client.connect()
    client.options()
    client.describe()
    client.setup()
    client.play()
    client.open_rtp_socket()

    parser_rtp = H264RTPParser()
    decoder = H264Decoder()

    frame_count = 0
    last_faces = []

    print("[INFO] Starting stream... Press ESC to exit")

    # Main loop

    while True:
        packet = client.receive_rtp_packet()
        if not packet:
            continue

        # For debugging:
        # print(f"[RTP] received: {len(packet)} bytes")

        nal = parser_rtp.feed(packet)
        if nal is None:
            continue

        frames = decoder.decode(nal)

        for frame in frames:
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]

            # 1) Downscale for detection
            small_w, small_h = 640, 360
            small = cv2.resize(img, (small_w, small_h))

            # 2) Detection every N frames
            if frame_count % stride == 0:
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

                        # Scale back to original frame size
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
                x1, y1, x2, y2 = faces[0]

                # Crop face for FaceMesh
                face_roi = img[y1:y2, x1:x2]
                if face_roi.size != 0:
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    mesh_result = mesh.process(face_rgb)

                    if mesh_result.multi_face_landmarks:
                        lm = mesh_result.multi_face_landmarks[0].landmark

                        # Compute Landmark (x,y) in full image coordinates
                        landmarks_5 = []
                        idxs = FACEMESH_LANDMARK_IDXS

                        for key in ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]:
                            i = idxs[key]
                            lx = lm[i].x * (x2 - x1) + x1
                            ly = lm[i].y * (y2 - y1) + y1
                            landmarks_5.append([lx, ly])
                        
                        landmarks_5 = np.array(landmarks_5, dtype=np.float32)

                        # Align face
                        aligned_preview = align_face(img, landmarks_5)

                        # Draw 5 landmarks on screen
                        for (lx, ly) in landmarks_5:
                            cv2.circle(img, (int(lx), int(ly)), 3, (0, 255, 255), -1)


            # 4) Face Detection - Draw boxes scaled to original image
            for (x1, y1, x2, y2) in faces:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


            # 5) Resize for display
            display_img = cv2.resize(img, (args.display_width, args.display_height))
            cv2.imshow("Face Detection - Hanwha Camera - Mediapipe", display_img)

            if aligned_preview is not None:
                cv2.imshow("Aligned Face (112x112)", aligned_preview)
            
            if cv2.waitKey(1) == 27:  # ESC
                return


if __name__ == "__main__":
    main()

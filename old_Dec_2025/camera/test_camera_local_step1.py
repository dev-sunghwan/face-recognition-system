# camera/test_camera_local_step1.py

import cv2
import argparse
import numpy as np

from rtsp_client import RTSPClient
from decoder import H264Decoder
from h264_rtp_parser import H264RTPParser

from insightface.app import FaceAnalysis


# 1) Arcface 5-point alighment Function
def align_face(img, landmarks_5):
    """
    img: BGR frame
    landmarks_5: (5,2) array
    return: 112 x 112 aligned face (ArcFace Style)
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


# 2) Main Streaming Loop
def main():
    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="Hanwha Camera Face Recognition Pipeline (InsightFace)")
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

    # Initialize InsightFace (Detection + 5 Landmark)
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320)) 
    print("[INFO] InsightFace (buffalo_l) loaded")

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


    # Main Streaming loop

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

            # InsightFace Detection
            if frame_count % 3 == 0:
                faces = app.get(img)

            aligned_face = None

            for f in faces:
                # 1) Bounding box
                x1, y1, x2, y2 = map(int, f.bbox)

                # Draw Rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 2) 5 Landmark (ArcFace standard)
                lm5 = f.landmark_2d_5 # shape (5,2)
                if lm5 is None or len(lm5) != 5:
                    continue

                for (lx, ly) in lm5:
                    cv2.circle(img, (int(lx), int(ly)), 2, (0, 255, 255), -1)

                # 3) Alignment (ArcFace standard)
                aligned_face = align_face(img, lm5)

                # Embedding would be next step

                break # Handle the first face for now


            # 4) Resize for display
            display_img = cv2.resize(img, (args.display_width, args.display_height))
            cv2.imshow("InsightFace Detection + Landmark + Alighment", display_img)

            if aligned_face is not None:
                cv2.imshow("Aligned Face (112x112)", aligned_face)
            
            if cv2.waitKey(1) == 27:  # ESC
                return


if __name__ == "__main__":
    main()

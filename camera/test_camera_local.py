# camera/test_camera_local.py

import cv2
import argparse
import numpy as np
import mediapipe as mp

from rtsp_client import RTSPClient
from decoder import H264Decoder
from h264_rtp_parser import H264RTPParser
# from insightface.app import FaceAnalysis



def main():
    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="Hanwha Camera Face Detection Test (Mediapipe)")
    parser.add_argument("--ip", required=True, help="Camera IP Address")
    parser.add_argument("--user", default="admin", help="Camera Username")
    parser.add_argument("--password", default="Sunap1!!", help="Camera Password")
    parser.add_argument("--stride", type=int, default=3, help="Do detection every N frames")
    parser.add_argument("--display_width", type=int, default=1280)
    parser.add_argument("--display_height", type=int, default=720)
    args = parser.parse_args()

    camera_ip = args.ip
    username = args.user
    password = args.password
    stride = args.stride

    # Construct RTSP URL using Hanwha official format 
    rtsp_url = f"rtsp://{username}:{password}@{camera_ip}/profile2/media.smp"
    print(f"[INFO] Connecting to: {rtsp_url}")

    # Initialize Face Detector - Mediapipe
    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    print("[Info] Mediapipe Face Detection initialized.")


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
                mp_input = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = detector.process(mp_input)

                faces = []
                if results.detections:
                    for det in results.detections:
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


            # Face Detection - Draw boxes scaled to original image
            for (x1, y1, x2, y2) in faces:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


            # 4) Resize for display
            display_img = cv2.resize(img, (args.display_width, args.display_height))
            cv2.imshow("Face Detection - Hanwha Camera - Mediapipe", display_img)
            
            if cv2.waitKey(1) == 27:  # ESC
                return


if __name__ == "__main__":
    main()

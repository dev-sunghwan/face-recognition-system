# camera/test_camera_local.py

import cv2
import argparse
from rtsp_client import RTSPClient
from decoder import H264Decoder
from h264_rtp_parser import H264RTPParser
from insightface.app import FaceAnalysis


def main():
    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="Hanwha Camera Face Detection Test")
    parser.add_argument("--ip", required=True, help="Camera IP Address")
    parser.add_argument("--user", default="admin", help="Camera Username")
    parser.add_argument("--password", default="Sunap1!!", help="Camera Password")
    args = parser.parse_args()

    camera_ip = args.ip
    username = args.user
    password = args.password 

    # Construct RTSP URL using Hanwha official format 
    rtsp_url = f"rtsp://{username}:{password}@{camera_ip}/profile2/media.smp"
    print(f"[INFO] Connecting to: {rtsp_url}")

    # Initialize Face Detector
    app = FaceAnalysis(name="buffalo_1")
    app.prepare(ctx_id=0, det_size=(640,640))


    # Initialize RTSP client + decoder
    client = RTSPClient(rtsp_url)
    client.connect()
    client.options()
    client.describe()
    client.setup()
    client.play()
    client.open_rtp_socket()

    parser = H264RTPParser()
    decoder = H264Decoder()

    print("[INFO] Starting stream... Press ESC to exit")

    while True:
        packet = client.receive_rtp_packet()
        if not packet:
            continue

        # For debugging:
        # print(f"[RTP] received: {len(packet)} bytes")

        nal = parser.feed(packet)
        if nal is None:
            continue

        frames = decoder.decode(nal)
        for frame in frames:
            img = frame.to_ndarray(format="bgr24")

            # Face Detection
            faces = app.get(img)

            for f in faces:
                x1, y1, x2, y2 = f.bbox.astype(int)
                cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), 2)

                # 5 Landmarks
                for (x, y) in f.landmark.astype(int):
                    cv2.circle(img, (x,y), 2, (0,0,255), -1)

            cv2.imshow("Face Detection - Hanwha Camera", img)
            if cv2.waitKey(1) == 27:  # ESC
                return


if __name__ == "__main__":
    main()

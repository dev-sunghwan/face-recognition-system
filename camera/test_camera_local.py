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
    parser.add_argument("--stride", type=int, default=3, help="Do detection every N frames")
    args = parser.parse_args()

    camera_ip = args.ip
    username = args.user
    password = args.password
    stride = args.stride

    # Construct RTSP URL using Hanwha official format 
    rtsp_url = f"rtsp://{username}:{password}@{camera_ip}/profile2/media.smp"
    print(f"[INFO] Connecting to: {rtsp_url}")

    # Initialize Face Detector
    app = FaceAnalysis(name="buffalo_l")
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
    frame_count = 0
    last_faces = []

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
            h, w = img.shape[:2]

            # 1) Downscale for detection
            small_w, small_h = 640, 360
            small = cv2.resize(img, (small_w, small_h))

            # 2) Detection every N frames
            if frame_count % stride == 0:
                faces = app.get(small)
                last_faces = faces
            else:
                faces = last_faces
            
            frame_count += 1

            # Face Detection - Draw boxes scaled to original image
            for f in faces:
                if f.landmark is None or f.bbox is None:
                    continue

                x1, y1, x2, y2 = f.bbox.astype(int)

                # scale bbox back to original
                x1 = int(x1 * (w / small_w))
                x2 = int(x2 * (w / small_w))
                y1 = int(y1 * (h / small_h))
                y2 = int(y2 * (h / small_h))

                cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), 2)

                # scale landmarks back
                for (lx, ly) in f.landmark.astype(int):
                    lx = int(lx * (w / small_w))
                    ly = int(ly * (h / small_h))
                    cv2.circle(img, (lx,ly), 2, (0,0,255), -1)

            # 4) Resize for display
            display_img = cv2.resize(img, (1280, 720))
            cv2.imshow("Face Detection - Hanwha Camera", display_img)
            
            if cv2.waitKey(1) == 27:  # ESC
                return


if __name__ == "__main__":
    main()

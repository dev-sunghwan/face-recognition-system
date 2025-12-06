# camera/test_camera_local.py

import cv2
from rtsp_client import RTSPClient
from decoder import H264Decoder
from h264_rtp_parser import H264RTPParser


def main():
    client = RTSPClient("rtsp://admin:Sunap1!!@192.168.4.73/profile2/media.smp")
    client.connect()
    client.options()
    client.describe()
    client.setup()
    client.play()
    client.open_rtp_socket()

    parser = H264RTPParser()
    decoder = H264Decoder()

    print("Starting frame decode loop...")

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
            cv2.imshow("Hanwha Camera", img)
            if cv2.waitKey(1) == 27:  # ESC
                return


if __name__ == "__main__":
    main()

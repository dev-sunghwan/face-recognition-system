#!/usr/bin/env python3
# camera/test_rtsp_connection.py
# RTSP 스트림 연결만 테스트하는 간단한 스크립트

import cv2
import sys
import os

def test_rtsp(url):
    """RTSP 스트림 연결 테스트"""
    print("=" * 60)
    print("RTSP Connection Test")
    print("=" * 60)
    print(f"Testing URL: {url}")
    print()
    
    # TCP transport 강제
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    
    # 연결 시도
    print("[1/3] Opening stream...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("❌ Failed to open stream")
        return False
    
    print("✅ Stream opened")
    
    # 프레임 읽기 시도
    print("[2/3] Reading first frame...")
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"✅ Frame grabbed: {w}x{h}")
            cap.release()
            return True
        print(f"   Attempt {i+1}/10...")
    
    print("❌ Failed to grab frame")
    cap.release()
    return False


if __name__ == "__main__":
    # 테스트할 URL들
    base_urls = [
        "rtsp://admin:Sunap1!!@45.92.235.163:8082/profile2/media.smp",
        "rtsp://admin:Sunap1!!@45.92.235.163:8082/stream1",
        "rtsp://admin:Sunap1!!@45.92.235.163:8082/stream",
        "rtsp://admin:Sunap1!!@45.92.235.163:8082/profile1/media.smp",
        "rtsp://admin:Sunap1!!@45.92.235.163:8082/",
    ]
    
    if len(sys.argv) > 1:
        # 명령줄 인자로 URL 제공
        test_rtsp(sys.argv[1])
    else:
        # 기본 URL들 테스트
        for url in base_urls:
            if test_rtsp(url):
                print()
                print("=" * 60)
                print("✅ SUCCESS! Working URL:")
                print(url)
                print("=" * 60)
                break
            print()
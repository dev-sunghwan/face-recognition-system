#!/usr/bin/env python3
# camera/main.py - Docker 진입점

import os
import sys
import subprocess

def main():
    # 환경변수로 설정 읽기
    mode = os.getenv("MODE", "mediapipe")  # mediapipe 또는 insightface
    camera_ip = os.getenv("CAMERA_IP", "192.168.1.100")
    username = os.getenv("CAMERA_USER", "admin")
    password = os.getenv("CAMERA_PASSWORD", "Sunap1!!")
    stride = os.getenv("STRIDE", "2")
    
    print("=" * 60)
    print("Face Recognition System - Docker Mode")
    print("=" * 60)
    print(f"MODE: {mode}")
    print(f"CAMERA_IP: {camera_ip}")
    print(f"USERNAME: {username}")
    print(f"STRIDE: {stride}")
    print("=" * 60)
    
    # 실행할 스크립트 선택
    if mode == "insightface":
        script = "test_opencv_insightface.py"
    else:
        script = "test_opencv_mediapipe.py"
    
    # 명령어 구성
    cmd = [
        "python",
        script,
        "--ip", camera_ip,
        "--user", username,
        "--password", password,
        "--stride", stride,
        "--headless"  # Docker에서는 항상 headless 모드
    ]
    
    print(f"[INFO] Running: {' '.join(cmd)}")
    print("=" * 60)
    
    # 스크립트 실행
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Received interrupt signal, shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Process failed with exit code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
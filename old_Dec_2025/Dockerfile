FROM python:3.11-slim-bookworm

# 빌드 도구 + 런타임 라이브러리 설치
RUN apt-get update && apt-get install -y \
    # 컴파일 도구
    build-essential \
    g++ \
    # OpenCV 의존성
    libgl1 \
    libglib2.0-0 \
    # FFmpeg
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "numpy>=1.24.0,<2.0.0" \
    opencv-python-headless==4.8.1.78 \
    onnxruntime==1.16.3 \
    mediapipe==0.10.9 \
    insightface==0.7.3

WORKDIR /app

# Copy whole project
COPY . .

# Go to camera directory
WORKDIR /app/camera

# execute main.py 
CMD ["python", "main.py"]
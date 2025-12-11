# Face Recognition System - OpenCV Version

PyAV를 제거하고 OpenCV VideoCapture로 전환한 버전입니다.

## 📁 파일 구조

```
face-recognition-system/
├── camera/
│   ├── test_opencv_mediapipe.py   # Mediapipe 버전
│   ├── test_opencv_insightface.py # InsightFace 버전
│   ├── main.py                    # Docker 진입점
│   └── frame_buffer.py            # 프레임 버퍼
├── Dockerfile
├── docker-compose.yml
└── README_OPENCV.md
```

## 🚀 로컬 실행 (Windows/Linux/Mac)

### Mediapipe 버전
```bash
cd camera
python test_opencv_mediapipe.py --ip 192.168.1.100 --user admin --password Sunap1!!
```

### InsightFace 버전
```bash
cd camera
python test_opencv_insightface.py --ip 192.168.1.100 --user admin --password Sunap1!!
```

### 옵션
```bash
--ip              # 카메라 IP (필수)
--user            # 사용자명 (기본: admin)
--password        # 비밀번호 (기본: Sunap1!!)
--stride 2        # N 프레임마다 감지 (기본: 2)
--headless        # 화면 출력 없이 실행
```

## 🐳 Docker 실행

### 방법 1: Docker Compose (추천)

**docker-compose.yml 수정:**
```yaml
environment:
  - MODE=mediapipe          # 또는 insightface
  - CAMERA_IP=192.168.1.100
  - CAMERA_USER=admin
  - CAMERA_PASSWORD=Sunap1!!
  - STRIDE=2
```

**실행:**
```bash
# 빌드 + 실행
docker-compose up --build

# 백그라운드 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

### 방법 2: Docker 명령어

```bash
# 빌드
docker build -t face-recognition-system .

# Mediapipe 버전 실행
docker run --rm \
  -e MODE=mediapipe \
  -e CAMERA_IP=192.168.1.100 \
  -e CAMERA_USER=admin \
  -e CAMERA_PASSWORD=Sunap1!! \
  face-recognition-system

# InsightFace 버전 실행
docker run --rm \
  -e MODE=insightface \
  -e CAMERA_IP=192.168.1.100 \
  -e CAMERA_USER=admin \
  -e CAMERA_PASSWORD=Sunap1!! \
  face-recognition-system
```

## 🔧 문제 해결

### RTSP 연결 실패
```
[ERROR] Failed to open RTSP stream
```
**해결:**
- IP 주소 확인
- 사용자명/비밀번호 확인
- 카메라 RTSP 포트 열려있는지 확인 (기본: 554)
- 방화벽 설정 확인

### 느린 프레임 레이트
```bash
--stride 5  # 감지 간격 늘리기
```

### 메모리 부족
- Mediapipe 버전 사용 (더 가볍고 빠름)
- `det_size=(160, 160)` 축소

## 📊 성능 비교

| 버전 | 속도 | 정확도 | 메모리 | 추천 |
|------|------|--------|--------|------|
| **Mediapipe** | ⚡⚡⚡ 빠름 | ⭐⭐⭐ 좋음 | 💾 낮음 | CPU 환경 |
| **InsightFace** | ⚡⚡ 보통 | ⭐⭐⭐⭐ 매우 좋음 | 💾💾 높음 | GPU 환경 |

## ✅ PyAV vs OpenCV 비교

| 항목 | PyAV (이전) | OpenCV (현재) |
|------|-------------|---------------|
| Docker 빌드 | ❌ 실패 | ✅ 성공 |
| 코드 복잡도 | 높음 (500줄) | 낮음 (200줄) |
| 파일 수 | 6개 | 3개 |
| 유지보수 | 어려움 | 쉬움 |
| RTSP 안정성 | 보통 | 좋음 |

## 🗑️ 삭제 가능한 파일 (PyAV 관련)

Docker로 정상 작동 확인 후 삭제:
- `camera/decoder.py`
- `camera/h264_rtp_parser.py`
- `camera/rtsp_client.py`
- `camera/test_camera_local.py` (구버전)
- `camera/test_camera_local_step1.py` (구버전)

## 📝 다음 단계

1. ✅ 로컬에서 테스트
2. ✅ Docker로 실행
3. ⬜ 얼굴 임베딩 추출
4. ⬜ 데이터베이스 연동
5. ⬜ 실시간 인식 시스템 구축
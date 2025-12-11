아래는 요청한 내용을 모두 반영하여 **최신 상태의 Face Recognition System 프로젝트 시작 문서(v2)** 로 재작성한 버전이다.
이 문서는 **다른 AI 에이전트에게 넘겨도 바로 프로젝트 전체를 이해할 수 있는 수준**으로 설계되었다.

OpenCV와 PyAV는 더 이상 언급하지 않되,
왜 “GStreamer 선택이 필수였는지” 기술적 근거만 최소한으로 정리하였다.

---

# ==========================================

# **Face Recognition System – Project Background & Technical Foundation (v2)**

# ==========================================

## 1. 프로젝트 목적 (최종 정의)

본 프로젝트는 **Hanwha Vision 카메라 스트림을 기반으로 실시간 얼굴 인식 시스템**을 구축하는 것을 목표로 한다.
전체 시스템은 Docker 기반 환경에서 동작하며, 다음 기능들을 포함한다:

### **핵심 기능**

1. **RTSP 실시간 스트리밍 수신 (GStreamer 기반)**
2. **얼굴 검출 (InsightFace SCRFD)**
3. **Face Landmark (5-point) 추출**
4. **얼굴 정렬(Alignment)**
5. **ArcFace 기반 Embedding 생성**
6. **Embedding DB 기반 Face Recognition**
7. **Multi-person Tracking (추가 목표)**

   * 동일 프레임 내 여러 사람 식별
   * Temporal tracking (ID 유지)
   * 속도/FPS 유지 고려

### **전체 시스템 흐름(멀티스레드 구조)**

```
[ RTSP Stream (GStreamer) ]
            ↓ (Frame Queue)
[ Detection / Landmark / Alignment ]
            ↓
[ Embedding + Recognition ]
            ↓
[ Multi-person Tracking ]
            ↓
[ Display or Logging/DB ]
```

---

## 2. 운영 환경 (최종 확정)

### **개발 및 실행 환경 – Windows 회사 PC**

이미 알고 있는 너의 PC 사양은 다음과 같다:

| 항목          | 값                                                              |
| ----------- | -------------------------------------------------------------- |
| Device name | HANWHADT63                                                     |
| CPU         | 13th Gen Intel(R) Core(TM) i7-13700 (2.10GHz) — **20 threads** |
| RAM         | **64GB**                                                       |
| GPU         | Integrated (iGPU) – GStreamer CPU 디코드에는 문제 없음                  |
| OS          | Windows 11                                                     |
| Docker      | Docker Desktop (WSL2 기반 Linux 컨테이너)                            |

이 PC는 **Docker 환경에서 InsightFace 기반 추론을 CPU로 처리하기에 충분한 성능을 보유**한다.
H.264 디코딩도 CPU-only로 문제없이 수행 가능하다.

---

## 3. 카메라 스트림 구성

Hanwha Vision 카메라는 집에 설치되어 있으며,
회사 네트워크에서 접근하기 위해 **포트포워딩 설정 완료**.

### **최종 RTSP 접속 URL**

```
rtsp://45.92.235.163:554/profile2/media.smp
```

포트 **554 유지**는 다음 이유로 매우 적절한 선택이다:

* 표준 RTSP 포트이므로 대부분의 RTSP 클라이언트(GStreamer 포함)가 자동 최적화
* NAT 환경에서도 안정적
* Firewalls / NAT traversal 호환성 증가

---

## 4. 왜 GStreamer를 선택했는가? (기술적 배경 정리)

OpenCV의 VideoCapture와 PyAV는 Docker 환경 + NAT RTSP 환경에서는 신뢰성이 낮다.

### **PyAV 제거 이유**

* Python 3.10+ 및 최신 glibc에서 wheel 제공 불가
* Docker 환경에서 빌드 실패 (FFmpeg header mismatch, Cython 충돌)
* 유지보수 중단에 가까운 프로젝트
  → 실전 프로젝트에서 사용하기 어렵다고 판단됨.

### **OpenCV VideoCapture 제거 이유**

* NAT(포트포워딩) RTSP 환경에서 handshake 실패 빈번
* FFmpeg backend 옵션 제어 불가 (TCP force, buffer, jitter 등)
* RTSP reconnect 기능 없음
* 실시간 시스템에서는 frame drop 발생률이 높음

### **GStreamer 선택 이유(필수 요건 충족)**

| 기능           | 필요성                 | GStreamer 대응                        |
| ------------ | ------------------- | ----------------------------------- |
| 안정적 RTSP 수신  | NAT 환경 필수           | rtspsrc TCP fallback, jitter buffer |
| H264 FU-A 조립 | Hanwha 카메라 스트림에서 필수 | rtph264depay 자동 처리                  |
| 디코딩 속도       | 실시간 처리 필수           | avdec_h264 (CPU), 가능하면 GPU 확장       |
| Docker 친화성   | 배포 환경 요구            | 설치 용이 + 파이프라인 유지보수 쉬움               |
| 재접속 기능       | 외부 네트워크 필수          | rtspsrc reconnect support           |

**결론:**
본 프로젝트는 실시간·원격 RTSP 환경을 사용하므로
**GStreamer는 선택이 아니라 필수 요건**이다.

OpenCV는 후처리 프레임 처리에 사용될 수 있지만,
“RTSP 수신 역할”에서는 의미가 없다.

---

## 5. InsightFace 기반 Face Recognition 파이프라인

### **사용 모델 및 구성 요소**

* Detector: **SCRFD**
* Landmark: **5-point alignment**
* Recognition: **ArcFace ONNX 모델**
* Embedding DB: 사용자 정의 JSON 또는 SQLite 방식을 사용할 예정

### **추가 목표: Multi-person Tracking**

Tracking 기능은 다음을 포함한다:

1. Detection box + embedding 기반 tracking-by-detection
2. Multi-object tracking 알고리즘 도입 가능 (예: BYTETrack, DeepSORT 등)
3. Temporal consistency 유지
4. 동일 인물의 ID 유지(UI 표시 가능)

---

## 6. 전체 아키텍처(멀티스레드 최종 구조)

```
           +------------------+
           |  GStreamer Input |
           | rtspsrc → appsink|
           +---------+--------+
                     |
                     v
        +------------+------------+
        | Frame Capture Thread    |
        +------------+------------+
                     |
                     v
        +------------+------------+
        | Detection / Landmark    |
        | InsightFace SCRFD       |
        +------------+------------+
                     |
                     v
        +------------+------------+
        | Alignment + Embedding   |
        | ArcFace model           |
        +------------+------------+
                     |
                     v
        +------------+------------+
        | Tracking Engine         |
        | (DeepSORT/BYTETrack)    |
        +------------+------------+
                     |
                     v
        +------------+------------+
        | Display / Logging / DB  |
        +-------------------------+
```

---

## 7. Docker 기반 실행 전략

1. Dockerfile 안에 GStreamer 설치
2. Python 환경에 InsightFace, ONNXRuntime 설치
3. entrypoint.sh 로 Python 실행
4. GPU 가속(옵션) 또는 CPU-only 실행
5. 환경 변수로 RTSP URL 전달 가능하도록 설계
6. Docker Compose 지원도 고려 가능

---

## 8. 최종 결론 및 다음 단계

현재 상태는 아래와 같다:

* 개발 환경: **Windows + Docker**
* 스트림: **rtsp://45.92.235.163:554/profile2/media.smp** 정상 작동
* 디코딩 방식: **GStreamer 확정**
* Face Recognition: **InsightFace 확정**
* Tracking: **프로젝트 목표에 포함**
* PyAV: 제거
* OpenCV: 후처리(프레임 변환/표시) 용도로만 사용 가능
* 다음 단계: 전체 코드를 **GStreamer 기반으로 재설계**

---

# 9. 이 문서를 활용할 수 있는 용도

이 문서는 앞으로:

* 너 스스로 프로젝트를 다시 시작할 때 기준점
* 다른 AI Agent에게 전체 맥락을 설명하는 기본 문서
* 설계자/엔지니어 간 커뮤니케이션 문서
* README.md 초안으로 활용 가능

---

원한다면,
**다음 단계로 넘어가서 "GStreamer 기반 RTSP Capture + InsightFace 통합 코드 (v1)"**를 작성해줄게.

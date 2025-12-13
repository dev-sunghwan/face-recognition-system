import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics.pairwise import cosine_similarity


# ======================================================
# 1) Mediapipe Face Landmarker 초기화
# ======================================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

base_options = BaseOptions(model_asset_path='face_landmarker.task')

options = FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE,  # frame-by-frame
    num_faces=5
)

landmarker = FaceLandmarker.create_from_options(options)


# ======================================================
# 2) Embedding DB (임시)
# ======================================================
embedding_db = {}   # {"Name": vector}


def compute_embedding(landmarks_result):
    if len(landmarks_result.face_landmarks) == 0:
        return None

    lmks = landmarks_result.face_landmarks[0]
    arr = np.array([[p.x, p.y, p.z] for p in lmks]).flatten()
    return arr


def match_face(emb, threshold=0.65):
    if not embedding_db:
        return "Unknown"

    names = list(embedding_db.keys())
    vecs = np.array(list(embedding_db.values()))

    sims = cosine_similarity([emb], vecs)[0]
    idx = np.argmax(sims)

    return names[idx] if sims[idx] >= threshold else "Unknown"


# ======================================================
# 3) RTSP Camera Stream (Digest 문제 없음)
# ======================================================
RTSP_URL = "rtsp://admin:Sunap1!!@192.168.4.102:554/profile2/media.smp"

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    raise RuntimeError("Failed to open RTSP stream. Check camera IP / credentials.")


print("Running Face Recognition v2 ...")


# ======================================================
# 4) Main Loop
# ======================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe는 mp.Image 객체를 입력으로 받는다
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Landmarks detection
    landmarks_result = landmarker.detect(mp_image)
    embedding = compute_embedding(landmarks_result)

    if embedding is not None:
        name = match_face(embedding)
        # bounding box는 mediapipe tasks API에서 직접 제공됨
        lmks = landmarks_result.face_landmarks[0]
        xs = [p.x for p in lmks]
        ys = [p.y for p in lmks]
        h, w, _ = frame.shape
        x1 = int(min(xs) * w)
        y1 = int(min(ys) * h)
        x2 = int(max(xs) * w)
        y2 = int(max(ys) * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition - Mediapipe v2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

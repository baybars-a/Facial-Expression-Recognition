"""
Facial Expression Recognition
================================
Detects faces via webcam and classifies emotions in real-time.
Outputs: happy, sad, angry, surprise, fear, disgust, neutral

Usage:
    python facial.py

Requirements (one-time install):
    pip install fer

Controls:
    Q / ESC - quit
    B       - toggle facial landmark overlay
"""

import cv2
import sys
import time
import os
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections as FLC

try:
    from fer.fer import FER
except ImportError:
    print("[ERROR] fer library not found. Install it with:")
    print("  pip install fer")
    sys.exit(1)


EMOTION_COLORS = {
    "happy":    (0, 220, 0),
    "sad":      (255, 80, 0),
    "angry":    (0, 0, 255),
    "surprise": (0, 220, 255),
    "fear":     (180, 0, 200),
    "disgust":  (0, 160, 80),
    "neutral":  (180, 180, 180),
}

EMOTION_LABEL = {
    "happy":    "Happy",
    "sad":      "Sad",
    "angry":    "Angry",
    "surprise": "Surprised",
    "fear":     "Fearful",
    "disgust":  "Disgusted",
    "neutral":  "Neutral",
}


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# Feature groups: (FLC connection list, BGR color, line thickness)
FEATURE_GROUPS = [
    (FLC.FACE_LANDMARKS_LEFT_EYE,       (255,  60,   0), 2),  # blue  – eyes
    (FLC.FACE_LANDMARKS_RIGHT_EYE,      (255,  60,   0), 2),
    (FLC.FACE_LANDMARKS_LEFT_IRIS,      (255, 180,  80), 1),  # pale  – iris
    (FLC.FACE_LANDMARKS_RIGHT_IRIS,     (255, 180,  80), 1),
    (FLC.FACE_LANDMARKS_LEFT_EYEBROW,   (  0, 200, 255), 2),  # amber – brows
    (FLC.FACE_LANDMARKS_RIGHT_EYEBROW,  (  0, 200, 255), 2),
    (FLC.FACE_LANDMARKS_LIPS,           ( 80,  60, 255), 2),  # red   – lips
    (FLC.FACE_LANDMARKS_NOSE,           (  0, 230, 140), 2),  # green – nose
    (FLC.FACE_LANDMARKS_FACE_OVAL,      ( 70,  70,  70), 1),  # grey  – oval
]


def draw_landmarks(frame, landmarker):
    """Draw full MediaPipe face feature contours using the Tasks API."""
    h, w = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return
    for face_lms in result.face_landmarks:
        def pt(idx):
            lm = face_lms[idx]
            return (int(lm.x * w), int(lm.y * h))
        for connections, color, thickness in FEATURE_GROUPS:
            for conn in connections:
                cv2.line(frame, pt(conn.start), pt(conn.end),
                         color, thickness, cv2.LINE_AA)


def draw_probability_bars(frame, emotions: dict, x: int, y: int):
    """Draw a small probability bar for each emotion above the face box."""
    bar_w = 110
    bar_h = 11
    row_h = 15
    font = cv2.FONT_HERSHEY_SIMPLEX

    sorted_emotions = sorted(emotions.items(), key=lambda e: -e[1])
    for i, (emotion, score) in enumerate(sorted_emotions):
        row_y = y - (len(sorted_emotions) - i) * row_h - 6
        if row_y < 0:
            continue
        color = EMOTION_COLORS.get(emotion, (200, 200, 200))
        filled = int(bar_w * score)
        cv2.rectangle(frame, (x, row_y), (x + bar_w, row_y + bar_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, row_y), (x + filled, row_y + bar_h), color, -1)
        label = f"{emotion[:3]}  {score:.0%}"
        cv2.putText(frame, label, (x + bar_w + 5, row_y + bar_h - 1),
                    font, 0.38, color, 1, cv2.LINE_AA)


def main():
    detector = FER(mtcnn=True)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
        num_faces=4,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_mesh = mp_vision.FaceLandmarker.create_from_options(opts)
    print("  Detector ready.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        sys.exit(1)

    print("Facial expression recognition running.")
    print("Controls: Q / ESC = quit\n")

    font = cv2.FONT_HERSHEY_DUPLEX
    show_landmarks = False

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        results = detector.detect_emotions(frame)

        for face in results:
            x, y, fw, fh = face["box"]
            emotions = face["emotions"]
            dominant, confidence = max(emotions.items(), key=lambda e: e[1])

            color = EMOTION_COLORS.get(dominant, (200, 200, 200))
            label = EMOTION_LABEL.get(dominant, dominant)

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            text = f"{label}  {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(text, font, 0.85, 2)
            label_y = y + fh + th + 12
            cv2.rectangle(frame, (x - 2, y + fh + 4),
                          (x + tw + 6, label_y + 4), color, -1)
            cv2.putText(frame, text, (x + 2, label_y),
                        font, 0.85, (0, 0, 0), 2, cv2.LINE_AA)

            draw_probability_bars(frame, emotions, x, y)

        if show_landmarks:
            draw_landmarks(frame, face_mesh)

        if not results:
            cv2.putText(frame, "No face detected. look at the camera",
                        (20, 55), font, 0.9, (0, 100, 255), 2, cv2.LINE_AA)

        lm_color  = (0, 255, 180) if show_landmarks else (140, 140, 140)
        lm_status = "Landmarks: ON  [B]" if show_landmarks else "Landmarks: OFF [B]"
        cv2.putText(frame, lm_status,
                    (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, lm_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "Q / ESC: quit",
                    (w - 185, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (140, 140, 140), 1, cv2.LINE_AA)

        cv2.imshow("Facial Expression Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('b'):
            show_landmarks = not show_landmarks

    cap.release()
    face_mesh.__exit__(None, None, None)
    cv2.destroyAllWindows()
    print("Exited.")


if __name__ == "__main__":
    main()

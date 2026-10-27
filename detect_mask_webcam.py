"""
detect_mask_webcam.py
---------------------
Real-time face mask detection using your laptop's webcam.

Usage:
    python detect_mask_webcam.py

Controls:
    Q  – Quit
    S  – Save screenshot
"""

import os
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
MODEL_PATH    = os.path.join(BASE_DIR, "model", "mask_detector.h5")
PROTOTXT_PATH = os.path.join(BASE_DIR, "face_detector", "deploy.prototxt")
WEIGHTS_PATH  = os.path.join(BASE_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5   # minimum face detection confidence
IMG_SIZE = (224, 224)

# Label → (display text, BGR colour)
LABELS = {
    0: ("Mask ✓",    (0, 200, 0)),    # green
    1: ("No Mask ✗", (0, 0, 220)),    # red
}


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────
def load_face_detector():
    """Load OpenCV DNN face detector (Caffe SSD)."""
    if not os.path.isfile(PROTOTXT_PATH) or not os.path.isfile(WEIGHTS_PATH):
        raise FileNotFoundError(
            "[ERROR] Face detector files not found in face_detector/.\n"
            "        Run: python download_face_detector.py"
        )
    net = cv2.dnn.readNet(PROTOTXT_PATH, WEIGHTS_PATH)
    print("[OK]  Face detector loaded.")
    return net


def load_mask_model():
    """Load our trained Keras mask classification model."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "[ERROR] Mask model not found.\n"
            "        Run: python train_model.py first."
        )
    model = load_model(MODEL_PATH)
    print("[OK]  Mask classifier loaded.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame processing
# ─────────────────────────────────────────────────────────────────────────────
def detect_and_classify(frame, face_net, mask_model):
    """
    Detect faces in `frame`, classify each as mask / no-mask, and
    draw bounding boxes + labels. Returns the annotated frame.
    """
    (h, w) = frame.shape[:2]

    # ── Face detection ───────────────────────────────────────────────────────
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0),
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    faces, locs, preds = [], [], []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Bounding box (clamped to frame)
        box     = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Pre-process face for mask classifier
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, IMG_SIZE)
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((x1, y1, x2, y2))

    # ── Mask classification (batch) ──────────────────────────────────────────
    if faces:
        faces = np.array(faces, dtype="float32")
        preds = mask_model.predict(faces, batch_size=32)

    # ── Draw results ─────────────────────────────────────────────────────────
    for (box, pred) in zip(locs, preds):
        (x1, y1, x2, y2) = box
        (mask_prob, no_mask_prob) = pred

        label_idx = 0 if mask_prob > no_mask_prob else 1
        label_text, color = LABELS[label_idx]
        prob = max(mask_prob, no_mask_prob)

        display = f"{label_text}: {prob * 100:.1f}%"

        # Rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Filled label background
        (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

        # Label text
        cv2.putText(
            frame, display, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
        )

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(frame, fps: float):
    """Draw FPS and keybind hints on the frame."""
    h, w = frame.shape[:2]

    # FPS
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1,
    )

    # Key hints
    hints = "Q: Quit  |  S: Screenshot"
    cv2.putText(
        frame, hints, (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
    )

    # Title bar
    cv2.putText(
        frame, "Face Mask Detection System", (w // 2 - 160, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 50), 2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Face Mask Detection System – Webcam Mode")
    print("=" * 60)

    face_net   = load_face_detector()
    mask_model = load_mask_model()

    # Open webcam (index 0 = default laptop camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("[ERROR] Cannot open webcam. Check if it is connected.")

    print("[INFO] Webcam started. Press Q to quit, S to screenshot.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame – retrying …")
            continue

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Detect + classify
        frame = detect_and_classify(frame, face_net, mask_model)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        draw_hud(frame, fps)

        cv2.imshow("Face Mask Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit requested.")
            break
        elif key == ord("s"):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"[OK]  Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Session ended.")


if __name__ == "__main__":
    main()

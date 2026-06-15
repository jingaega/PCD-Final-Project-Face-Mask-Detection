"""
predict.py — Inference utilities for image and real-time webcam detection.

Workflow:
  1. OpenCV SSD detects face bounding boxes.
  2. Each crop is classified by MobileNetV2.
  3. Results are annotated and returned (or displayed live).

Usage (webcam):
    python -m src.predict
"""

import cv2
import numpy as np
import tensorflow as tf

from src.config import (
    CLASSES, IMG_SIZE,
    MODEL_SAVE_PATH, FACE_DET_DIR,
    FACE_CONF_THRESHOLD, MASK_THRESHOLD,
    PREPROCESSING_CONFIG,
)
from src.preprocessing import apply_preprocessing

# BGR color per class
CLASS_COLORS = {
    "WithMask":    (0, 200, 0),    # green
    "WithoutMask": (0, 0, 220),    # red
}

# OpenCV SSD face detector weights
PROTOTXT   = str(FACE_DET_DIR / "deploy.prototxt")
CAFFEMODEL = str(FACE_DET_DIR / "res10_300x300_ssd_iter_140000.caffemodel")


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_classifier() -> tf.keras.Model:
    return tf.keras.models.load_model(str(MODEL_SAVE_PATH))


def load_face_detector() -> cv2.dnn_Net:
    return cv2.dnn.readNet(CAFFEMODEL, PROTOTXT)


def load_models() -> tuple:
    """Return (classifier, face_net). Both loaded from disk."""
    print("Loading classifier ...")
    classifier = load_classifier()
    print("Loading face detector ...")
    face_net = load_face_detector()
    return classifier, face_net


# ── Core detection ────────────────────────────────────────────────────────────

def detect_and_predict(
    frame: np.ndarray,
    classifier: tf.keras.Model,
    face_net: cv2.dnn_Net,
    conf_threshold: float = FACE_CONF_THRESHOLD,
    preprocessing_config: dict | None = None,
) -> list[tuple]:
    """
    Detect faces in a BGR frame and classify each as masked / unmasked.

    Returns:
        List of (x1, y1, x2, y2, label, confidence) tuples.
    """
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1.0, size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []
    for i in range(detections.shape[2]):
        face_conf = float(detections[0, 0, i, 2])
        if face_conf < conf_threshold:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Classical preprocessing (BGR → BGR)
        cfg = preprocessing_config if preprocessing_config is not None else PREPROCESSING_CONFIG
        face_crop = apply_preprocessing(face_crop, cfg)

        # Preprocess for MobileNetV2
        face_rgb     = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, IMG_SIZE)
        face_input   = np.expand_dims(face_resized / 255.0, axis=0)

        prob = float(classifier.predict(face_input, verbose=0)[0][0])

        # sigmoid output: WithMask → index 0 (prob < threshold)
        #                 WithoutMask → index 1 (prob >= threshold)
        if prob < MASK_THRESHOLD:
            label      = CLASSES[0]   # WithMask
            confidence = 1.0 - prob
        else:
            label      = CLASSES[1]   # WithoutMask
            confidence = prob

        results.append((x1, y1, x2, y2, label, confidence))

    return results


def annotate_frame(frame: np.ndarray, results: list) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the frame."""
    out = frame.copy()
    for (x1, y1, x2, y2, label, conf) in results:
        color = CLASS_COLORS[label]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    return out


# ── Webcam ────────────────────────────────────────────────────────────────────

def run_webcam() -> None:
    """Run real-time mask detection on the default webcam."""
    classifier, face_net = load_models()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    print("Press  q  to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_and_predict(frame, classifier, face_net)
        annotated = annotate_frame(frame, results)

        # Stats overlay
        n_mask    = sum(1 for *_, lbl, _ in results if lbl == "WithMask")
        n_no_mask = sum(1 for *_, lbl, _ in results if lbl == "WithoutMask")
        cv2.putText(
            annotated,
            f"With mask: {n_mask}  Without: {n_no_mask}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        cv2.imshow("Face Mask Detection  [q to quit]", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()

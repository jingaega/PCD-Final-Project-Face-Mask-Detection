"""
evaluate.py — Model evaluation on the test set.

Outputs:
  - Classification report (Precision, Recall, F1 per class)
  - Macro-averaged F1-Score and Accuracy
  - Inference FPS on test set
  - Confusion matrix heatmap
  - Grad-CAM samples for a handful of test images

Usage:
    python -m src.evaluate
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from src.config import CLASSES, MODEL_SAVE_PATH, MODEL_DIR
from src.dataset import build_generators
from src.gradcam import compute_heatmap, make_gradcam_figure


def measure_fps(model: tf.keras.Model, test_gen, n_batches: int = 20) -> float:
    """
    Measure average inference FPS over n_batches of the test generator.

    Args:
        model:     Trained Keras model.
        test_gen:  Test data generator.
        n_batches: Number of batches to time.

    Returns:
        fps: Frames (images) processed per second.
    """
    total_images = 0
    total_time   = 0.0

    for i, (batch_x, _) in enumerate(test_gen):
        if i >= n_batches:
            break
        start = time.perf_counter()
        model.predict(batch_x, verbose=0)
        elapsed = time.perf_counter() - start
        total_images += len(batch_x)
        total_time   += elapsed

    fps = total_images / total_time if total_time > 0 else 0.0
    return fps


def evaluate(model: tf.keras.Model = None) -> tuple[float, float]:
    """
    Full evaluation pipeline on the test set.

    Args:
        model: Optional pre-loaded model. Loads from disk if None.

    Returns:
        (accuracy, macro_f1)
    """
    if model is None:
        print(f"Loading model from {MODEL_SAVE_PATH} ...")
        model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))

    _, _, test_gen = build_generators(augment=False)

    # ── Predictions ───────────────────────────────────────────────────────
    print("\nRunning predictions on test set...")
    y_prob = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    y_true = test_gen.classes

    # ── Classification report ─────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1-Score: {macro_f1:.4f}")

    # ── FPS ───────────────────────────────────────────────────────────────
    print("\nMeasuring inference speed...")
    _, _, test_gen_fps = build_generators(augment=False)
    fps = measure_fps(model, test_gen_fps)
    print(f"  Inference FPS : {fps:.1f}  ({1000/fps:.1f} ms/frame)")
    print("=" * 55)

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
    )
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = MODEL_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved → {cm_path}")

    # ── Grad-CAM samples ──────────────────────────────────────────────────
    print("\nGenerating Grad-CAM samples...")
    _, _, test_gen_gc = build_generators(augment=False)
    batch_x, batch_y = next(iter(test_gen_gc))

    n_samples   = min(4, len(batch_x))
    face_crops  = []
    heatmaps    = []
    pred_labels = []

    for i in range(n_samples):
        img_array = np.expand_dims(batch_x[i], axis=0)   # (1, 224, 224, 3)
        heatmap   = compute_heatmap(img_array, model)

        # Convert back to BGR uint8 for display
        bgr = (batch_x[i][..., ::-1] * 255).astype(np.uint8)
        prob  = float(model.predict(img_array, verbose=0)[0][0])
        label = CLASSES[int(prob > 0.5)]

        face_crops.append(bgr)
        heatmaps.append(heatmap)
        pred_labels.append(label)

    gc_path = MODEL_DIR / "gradcam_samples.png"
    make_gradcam_figure(face_crops, heatmaps, pred_labels, save_path=gc_path)
    print(f"Grad-CAM figure saved → {gc_path}")

    return acc, macro_f1


if __name__ == "__main__":
    evaluate()

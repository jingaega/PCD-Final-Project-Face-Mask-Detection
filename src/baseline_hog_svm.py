"""
baseline_hog_svm.py — Classical HOG + Linear SVM baseline.

Compares against the MobileNetV2 deep learning model using the same
test set and metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).

Usage:
    python -m src.baseline_hog_svm
"""

import time
import joblib
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
)
import tensorflow as tf

from src.config import (
    TRAIN_DIR, TEST_DIR, CLASSES, MODEL_DIR, MODEL_SAVE_PATH, IMG_SIZE,
)

SVM_SAVE_PATH = MODEL_DIR / "hog_svm.joblib"
SVM_CM_PATH   = MODEL_DIR / "hog_svm_confusion_matrix.png"

# HOG parameters
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    channel_axis=-1,   # colour HOG (RGB)
)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_hog(img_bgr: np.ndarray) -> np.ndarray:
    """Resize to IMG_SIZE, convert to RGB, compute HOG descriptor."""
    img = cv2.resize(img_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return hog(img, **HOG_PARAMS)


def load_split(directory: Path, label: int) -> tuple[list, list]:
    images, labels = [], []
    for cls_idx, cls in enumerate(CLASSES):
        cls_dir = directory / cls
        paths = sorted(cls_dir.glob("*.*"))
        for p in tqdm(paths, desc=f"{directory.name}/{cls}", leave=False):
            img = cv2.imread(str(p))
            if img is None:
                continue
            images.append(extract_hog(img))
            labels.append(cls_idx)
    return images, labels


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    # ── Extract features ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  HOG + Linear SVM Baseline")
    print("=" * 60)

    print("\nExtracting HOG features from training set...")
    X_train, y_train = load_split(TRAIN_DIR, label=0)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"  Train: {X_train.shape[0]} images, {X_train.shape[1]} features each")

    print("\nExtracting HOG features from test set...")
    X_test, y_test = load_split(TEST_DIR, label=0)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(f"  Test : {X_test.shape[0]} images, {X_test.shape[1]} features each")

    # ── Train SVM ─────────────────────────────────────────────────────────
    print("\nTraining Linear SVM...")
    t0 = time.perf_counter()
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    svm.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.1f}s")

    joblib.dump(svm, SVM_SAVE_PATH)
    print(f"  SVM saved -> {SVM_SAVE_PATH}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    t0 = time.perf_counter()
    y_pred = svm.predict(X_test)
    inf_time = time.perf_counter() - t0
    fps = len(X_test) / inf_time

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    f1_with    = f1_score(y_test, y_pred, pos_label=0)
    f1_without = f1_score(y_test, y_pred, pos_label=1)

    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT — HOG + SVM")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro F1-Score: {macro_f1:.4f}")
    print(f"  Inference FPS : {fps:.1f}")
    print("=" * 55)

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
    )
    ax.set_title("Confusion Matrix — HOG + SVM", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(SVM_CM_PATH, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved -> {SVM_CM_PATH}")

    # ── Comparison table ──────────────────────────────────────────────────
    print("\nLoading MobileNetV2 results for comparison...")
    dl_model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))

    from src.dataset import build_generators
    _, _, test_gen_dl = build_generators(augment=False)
    y_prob_dl = dl_model.predict(test_gen_dl, verbose=0).flatten()
    y_pred_dl = (y_prob_dl > 0.5).astype(int)
    y_true_dl = test_gen_dl.classes

    # FPS for DL model
    _, _, test_gen_fps = build_generators(augment=False)
    t0 = time.perf_counter()
    for i, (bx, _) in enumerate(test_gen_fps):
        if i >= 10: break
        dl_model.predict(bx, verbose=0)
    dl_fps = (10 * 32) / (time.perf_counter() - t0)

    dl_acc = accuracy_score(y_true_dl, y_pred_dl)
    dl_f1  = f1_score(y_true_dl, y_pred_dl, average="macro")

    print("\n" + "=" * 65)
    print("  COMPARISON: MobileNetV2  vs  HOG + Linear SVM")
    print("=" * 65)
    print(f"  {'Metric':<22} {'MobileNetV2':>14} {'HOG + SVM':>12}")
    print("-" * 65)
    print(f"  {'Accuracy':<22} {dl_acc:>14.4f} {acc:>12.4f}")
    print(f"  {'Macro F1':<22} {dl_f1:>14.4f} {macro_f1:>12.4f}")
    print(f"  {'WithMask F1':<22} {f1_score(y_true_dl,y_pred_dl,pos_label=0):>14.4f} {f1_with:>12.4f}")
    print(f"  {'WithoutMask F1':<22} {f1_score(y_true_dl,y_pred_dl,pos_label=1):>14.4f} {f1_without:>12.4f}")
    print(f"  {'Inference FPS':<22} {dl_fps:>14.1f} {fps:>12.1f}")
    print("=" * 65)

    return acc, macro_f1, fps


if __name__ == "__main__":
    run()

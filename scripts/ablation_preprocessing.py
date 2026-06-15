"""
ablation_preprocessing.py — Preprocessing ablation study.

Evaluates the trained MobileNetV2 model on the test set under four
preprocessing conditions to measure the impact of each step.

Usage:
    python scripts/ablation_preprocessing.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

from src.config import MODEL_SAVE_PATH, MODEL_DIR, TEST_DIR, IMG_SIZE, CLASSES, BATCH_SIZE
from src.preprocessing import apply_preprocessing

ABLATION_CONFIGS = {
    "(a) No preprocessing":   {"gaussian": False, "median": False, "clahe": False, "contrast_stretch": False},
    "(b) Gaussian only":      {"gaussian": True,  "median": False, "clahe": False, "contrast_stretch": False},
    "(c) CLAHE only":         {"gaussian": False, "median": False, "clahe": True,  "contrast_stretch": False},
    "(d) Gaussian + CLAHE":   {"gaussian": True,  "median": False, "clahe": True,  "contrast_stretch": False},
}

OUT_PNG = MODEL_DIR / "ablation_preprocessing.png"
OUT_CSV = MODEL_DIR / "ablation_results.csv"


def load_test_images():
    """Load all test images as BGR arrays with their labels."""
    images, labels = [], []
    for cls_idx, cls in enumerate(CLASSES):
        folder = TEST_DIR / cls
        for p in sorted(folder.glob("*.*")):
            img = cv2.imread(str(p))
            if img is not None:
                images.append(img)
                labels.append(cls_idx)
    return images, np.array(labels)


def predict_with_config(model, images, labels, config):
    """Run inference on all images with a given preprocessing config."""
    y_pred, total_time = [], 0.0

    for img_bgr in images:
        preprocessed = apply_preprocessing(img_bgr, config)
        rgb     = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        arr     = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)

        t0   = time.perf_counter()
        prob = float(model.predict(arr, verbose=0)[0][0])
        total_time += time.perf_counter() - t0

        y_pred.append(int(prob > 0.5))

    y_pred = np.array(y_pred)
    acc    = accuracy_score(labels, y_pred)
    f1     = f1_score(labels, y_pred, average="macro")
    fps    = len(images) / total_time
    return acc, f1, fps


def run():
    print("\n" + "=" * 60)
    print("  PREPROCESSING ABLATION STUDY")
    print("=" * 60)

    print(f"\nLoading model from {MODEL_SAVE_PATH} ...")
    model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))

    print("Loading test images...")
    images, labels = load_test_images()
    print(f"  {len(images)} test images loaded.")

    results = {}
    print()
    for name, config in ABLATION_CONFIGS.items():
        print(f"Running {name} ...")
        acc, f1, fps = predict_with_config(model, images, labels, config)
        results[name] = {"Accuracy": acc, "Macro F1": f1, "FPS": fps}
        print(f"  Accuracy={acc:.4f}  Macro F1={f1:.4f}  FPS={fps:.1f}")

    # ── Print table ───────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  ABLATION RESULTS")
    print("=" * 68)
    print(f"  {'Variant':<28} {'Accuracy':>10} {'Macro F1':>10} {'FPS':>8}")
    print("-" * 68)
    baseline_acc = results[list(results.keys())[0]]["Accuracy"]
    baseline_f1  = results[list(results.keys())[0]]["Macro F1"]
    for name, r in results.items():
        diff_acc = r["Accuracy"] - baseline_acc
        diff_f1  = r["Macro F1"] - baseline_f1
        diff_str = f"  ({diff_acc:+.4f})" if name != list(results.keys())[0] else ""
        print(f"  {name:<28} {r['Accuracy']:>10.4f} {r['Macro F1']:>10.4f} {r['FPS']:>8.1f}{diff_str}")
    print("=" * 68)

    # Flag if preprocessing hurts
    for name, r in results.items():
        if name == list(results.keys())[0]:
            continue
        if r["Accuracy"] < baseline_acc - 0.001:
            print(f"\n  [!] {name} DECREASED accuracy by {baseline_acc - r['Accuracy']:.4f}")
        elif r["Accuracy"] > baseline_acc + 0.001:
            print(f"\n  [+] {name} IMPROVED accuracy by {r['Accuracy'] - baseline_acc:.4f}")

    # ── Bar chart ─────────────────────────────────────────────────────────
    names = list(results.keys())
    accs  = [results[n]["Accuracy"] for n in names]
    f1s   = [results[n]["Macro F1"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, f1s,  width, label="Macro F1", color="#DD8452", alpha=0.85)

    ax.set_ylim(min(accs + f1s) - 0.02, 1.005)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Preprocessing Ablation Study — MobileNetV2 on Test Set",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.4f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.4f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"\nBar chart saved -> {OUT_PNG}")

    # ── CSV ───────────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Variant", "Accuracy", "Macro F1", "FPS"])
        writer.writeheader()
        for name, r in results.items():
            writer.writerow({"Variant": name, "Accuracy": f"{r['Accuracy']:.4f}",
                             "Macro F1": f"{r['Macro F1']:.4f}", "FPS": f"{r['FPS']:.1f}"})
    print(f"CSV saved        -> {OUT_CSV}")

    return results


if __name__ == "__main__":
    run()

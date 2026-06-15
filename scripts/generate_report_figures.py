"""
generate_report_figures.py — Generate figures for the final report.

Produces three figures inside report/figures/:
  fig_detection.png      — side-by-side detection results (WithMask / WithoutMask)
  fig_gradcam.png        — Grad-CAM 3-panel grid for both classes
  fig_preprocessing.png  — before/after preprocessing strip (4 operations)

Run from project root:
    python scripts/generate_report_figures.py
"""

import sys
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODEL_SAVE_PATH, TEST_DIR, CLASSES, IMG_SIZE
from src.gradcam import compute_heatmap, overlay_heatmap
from src.preprocessing import (
    gaussian_blur, histogram_equalization,
    median_filter, contrast_stretching,
)

OUT_DIR = ROOT / "report" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_samples(n=3):
    """Load n random images from each class in the test set."""
    samples = {}
    for cls in CLASSES:
        paths = sorted((TEST_DIR / cls).glob("*.*"))
        chosen = random.sample(paths, min(n, len(paths)))
        imgs = []
        for p in chosen:
            img = cv2.imread(str(p))
            if img is not None:
                imgs.append(img)
        samples[cls] = imgs
    return samples


def classify(model, img_bgr):
    """Return (label, confidence) for a BGR face crop."""
    rgb = cv2.cvtColor(cv2.resize(img_bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)
    arr = np.expand_dims(rgb / 255.0, 0).astype(np.float32)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "WithMask" if prob < 0.5 else "WithoutMask"
    conf  = 1 - prob if prob < 0.5 else prob
    return label, conf, arr


def draw_label_box(ax, label, conf):
    color = "#2ecc71" if label == "WithMask" else "#e74c3c"
    symbol = "✓" if label == "WithMask" else "✗"
    ax.text(
        0.5, -0.08,
        f"{symbol}  {label}  {conf:.1%}",
        transform=ax.transAxes,
        ha="center", va="top", fontsize=9, fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"),
    )


# ── Figure 1: Detection Results ───────────────────────────────────────────────

def make_detection_figure(model, samples):
    """3+3 grid: top row WithMask, bottom row WithoutMask."""
    n = 3
    fig, axes = plt.subplots(2, n, figsize=(10, 7))
    fig.suptitle(
        "Face Mask Classification Results\n(MobileNetV2, test set images)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    for row_idx, cls in enumerate(CLASSES):
        imgs = samples[cls][:n]
        for col_idx, img in enumerate(imgs):
            ax = axes[row_idx][col_idx]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            ax.axis("off")

            label, conf, _ = classify(model, img)
            draw_label_box(ax, label, conf)

        # Row label
        axes[row_idx][0].set_ylabel(
            f"True: {cls}", fontsize=10, fontweight="bold",
            rotation=90, labelpad=8,
        )

    # Legend
    patch_mask    = mpatches.Patch(color="#2ecc71", label="WithMask")
    patch_nomask  = mpatches.Patch(color="#e74c3c", label="WithoutMask")
    fig.legend(handles=[patch_mask, patch_nomask],
               loc="lower center", ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    out = OUT_DIR / "fig_detection.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2: Grad-CAM ────────────────────────────────────────────────────────

def make_gradcam_figure(model, samples):
    """2 rows (one per class) x 3 cols (original / heatmap / overlay)."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 8))
    fig.suptitle(
        "Grad-CAM Explainability — Last Convolutional Layer (out_relu, 7×7×1280)",
        fontsize=12, fontweight="bold", y=0.99,
    )

    col_titles = ["Face Crop", "Grad-CAM Heatmap", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row_idx, cls in enumerate(CLASSES):
        img = samples[cls][0]
        label, conf, arr = classify(model, img)
        heatmap = compute_heatmap(arr, model)
        overlay, heatmap_colored = overlay_heatmap(img, heatmap, alpha=0.45)

        ax_orig  = axes[row_idx][0]
        ax_heat  = axes[row_idx][1]
        ax_over  = axes[row_idx][2]

        ax_orig.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax_heat.imshow(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
        ax_over.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        for ax in (ax_orig, ax_heat, ax_over):
            ax.axis("off")

        color  = "#2ecc71" if label == "WithMask" else "#e74c3c"
        symbol = "✓" if label == "WithMask" else "✗"
        axes[row_idx][0].set_ylabel(
            f"True: {cls}\nPred: {symbol} {label} ({conf:.1%})",
            fontsize=9, fontweight="bold", color=color,
            rotation=0, labelpad=6, ha="right", va="center",
        )

    # Colorbar annotation
    fig.text(
        0.5, 0.01,
        "Heatmap colour scale:  Blue = low activation   →   Red = high activation",
        ha="center", fontsize=9, style="italic", color="grey",
    )

    plt.tight_layout(rect=[0.12, 0.04, 1, 0.97])
    out = OUT_DIR / "fig_gradcam.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: Preprocessing ───────────────────────────────────────────────────

def make_preprocessing_figure(samples):
    """
    2 rows (WithMask, WithoutMask) x 5 cols
    (Original | Gaussian | Median | CLAHE | Contrast stretch)
    """
    ops = [
        ("Original",           lambda x: x),
        ("Gaussian Blur",      lambda x: gaussian_blur(x, ksize=5)),
        ("Median Filter",      lambda x: median_filter(x, ksize=5)),
        ("CLAHE",              histogram_equalization),
        ("Contrast Stretching",contrast_stretching),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(
        "Classical DIP Preprocessing — Effect on Face Crops",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for col_idx, (title, _) in enumerate(ops):
        axes[0][col_idx].set_title(title, fontsize=10, fontweight="bold", pad=5)

    for row_idx, cls in enumerate(CLASSES):
        img = samples[cls][1] if len(samples[cls]) > 1 else samples[cls][0]
        axes[row_idx][0].set_ylabel(cls, fontsize=10, fontweight="bold",
                                    rotation=0, labelpad=6, ha="right", va="center")

        for col_idx, (_, fn) in enumerate(ops):
            processed = fn(img.copy())
            ax = axes[row_idx][col_idx]
            ax.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            ax.axis("off")

            if col_idx == 0:
                ax.patch.set_edgecolor("black")
                ax.patch.set_linewidth(2)

    plt.tight_layout()
    out = OUT_DIR / "fig_preprocessing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model from {MODEL_SAVE_PATH} ...")
    model = tf.keras.models.load_model(str(MODEL_SAVE_PATH))

    print("Loading test samples ...")
    samples = load_samples(n=3)

    print("\nGenerating figures ...")
    make_detection_figure(model, samples)
    make_gradcam_figure(model, samples)
    make_preprocessing_figure(samples)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

"""
gradcam.py — Grad-CAM explainability for the face mask classifier.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the image
regions that most influenced the model's prediction. For a mask detector,
a well-trained model should activate over the nose and mouth region.

Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization." ICCV 2017.
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Streamlit
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import IMG_SIZE, MODEL_DIR

# Names of the classification head layers (must match model.py)
_HEAD_LAYER_NAMES = ["gap", "dropout_1", "dense_128", "dropout_2", "output"]

# Last conv layer name in MobileNetV2 (include_top=False)
# Verified: output shape (batch, 7, 7, 1280) for 224×224 input
_LAST_CONV_LAYER = "out_relu"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_base(model: tf.keras.Model) -> tf.keras.Model:
    """Return the MobileNetV2 sub-model from the full model."""
    return next(l for l in model.layers if "mobilenetv2" in l.name)


def _get_head(model: tf.keras.Model) -> tf.keras.Sequential:
    """Return the custom classification head as a Sequential model."""
    return tf.keras.Sequential(
        [model.get_layer(name) for name in _HEAD_LAYER_NAMES],
        name="classification_head",
    )


# ── Core Grad-CAM ─────────────────────────────────────────────────────────────

def compute_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a single preprocessed face image.

    Args:
        img_array: Float32 array of shape (1, 224, 224, 3), values in [0, 1].
        model:     Trained Keras model (MaskDetector).

    Returns:
        heatmap: np.ndarray of shape (7, 7), values in [0, 1].
                 High values = regions the model focused on.
    """
    base = _get_base(model)
    head = _get_head(model)

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        # Forward pass — watch the feature maps from the last conv block
        conv_outputs = base(img_tensor, training=False)   # (1, 7, 7, 1280)
        tape.watch(conv_outputs)
        predictions  = head(conv_outputs, training=False) # (1, 1)

    # Gradients of the prediction score w.r.t. each feature map pixel
    grads = tape.gradient(predictions, conv_outputs)      # (1, 7, 7, 1280)

    # Importance weight for each channel = mean gradient over spatial dims
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # (1280,)

    # Weighted sum of feature maps → heatmap
    conv_outputs = conv_outputs[0]                        # (7, 7, 1280)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]# (7, 7, 1)
    heatmap = tf.squeeze(heatmap)                         # (7, 7)

    # ReLU — keep only positive contributions
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize to [0, 1]
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


# ── Overlay ───────────────────────────────────────────────────────────────────

def overlay_heatmap(
    bgr_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Superimpose the Grad-CAM heatmap on a BGR face image.

    Args:
        bgr_img: BGR image (H, W, 3).
        heatmap: Grad-CAM heatmap (h, w), values in [0, 1].
        alpha:   Heatmap opacity (0 = invisible, 1 = full overlay).

    Returns:
        overlay:        BGR image with heatmap blended in.
        heatmap_colored: Colorized (JET) heatmap as BGR image.
    """
    h, w = bgr_img.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(bgr_img, 1.0 - alpha, heatmap_colored, alpha, 0)
    return overlay, heatmap_colored


# ── Visualization ─────────────────────────────────────────────────────────────

def make_gradcam_figure(
    face_crops: list[np.ndarray],
    heatmaps: list[np.ndarray],
    labels: list[str],
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Build a matplotlib figure comparing original crops with Grad-CAM overlays.

    Layout (per face):
        [ Original crop ]  |  [ Grad-CAM overlay ]

    Args:
        face_crops: List of BGR face images.
        heatmaps:   Corresponding Grad-CAM heatmaps.
        labels:     Predicted class label for each face.
        save_path:  If provided, save the figure here.

    Returns:
        Matplotlib Figure object.
    """
    n = len(face_crops)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.5 * n), squeeze=False)

    for i, (crop, heatmap, label) in enumerate(zip(face_crops, heatmaps, labels)):
        overlay, _ = overlay_heatmap(crop, heatmap)

        axes[i][0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[i][0].set_title(f"Face {i + 1} — {label}", fontsize=10, fontweight="bold")
        axes[i][0].axis("off")

        axes[i][1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i][1].set_title("Grad-CAM", fontsize=10)
        axes[i][1].axis("off")

    plt.suptitle(
        "Grad-CAM: regions driving the mask prediction",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM figure saved → {save_path}")

    return fig

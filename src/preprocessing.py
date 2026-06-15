"""
preprocessing.py — Classical Digital Image Processing pipeline.

All functions accept and return BGR numpy arrays (uint8).
Designed for face crops before mask classification.
"""

import cv2
import numpy as np


# ── Individual filters ────────────────────────────────────────────────────────

def gaussian_blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Gaussian low-pass filter for noise reduction."""
    if ksize % 2 == 0:
        ksize += 1  # kernel must be odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)


def median_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median filter — effective against salt-and-pepper noise."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(img, ksize)


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel
    of LAB color space. Improves local contrast without over-amplifying noise.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def contrast_stretching(img: np.ndarray) -> np.ndarray:
    """
    Min-max contrast stretching per channel.
    Maps [min, max] → [0, 255] independently for each BGR channel.
    """
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            out[:, :, c] = (ch - lo) / (hi - lo) * 255.0
        else:
            out[:, :, c] = ch
    return out.astype(np.uint8)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def apply_preprocessing(img: np.ndarray, config: dict) -> np.ndarray:
    """
    Apply a configurable preprocessing pipeline to a BGR image.

    Order: noise reduction first, then contrast enhancement.

    Args:
        img:    BGR uint8 numpy array.
        config: Dict with boolean flags:
                  gaussian         — Gaussian blur
                  median           — Median filter
                  clahe            — CLAHE histogram equalization
                  contrast_stretch — Min-max contrast stretching

    Returns:
        Preprocessed BGR uint8 numpy array, same shape as input.
    """
    out = img.copy()

    # 1. Noise reduction
    if config.get("gaussian", False):
        out = gaussian_blur(out)
    if config.get("median", False):
        out = median_filter(out)

    # 2. Contrast / brightness enhancement
    if config.get("clahe", False):
        out = histogram_equalization(out)
    if config.get("contrast_stretch", False):
        out = contrast_stretching(out)

    return out

"""
dataset.py — Data loading and augmentation using Keras ImageDataGenerator.

Dataset folder structure expected:
    data/raw/
    ├── Train/
    │   ├── WithMask/
    │   └── WithoutMask/
    ├── Validation/
    │   ├── WithMask/
    │   └── WithoutMask/
    └── Test/
        ├── WithMask/
        └── WithoutMask/
"""

import cv2
import numpy as np
import tensorflow as tf
from src.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE
)
from src.preprocessing import apply_preprocessing


def _make_preprocessing_fn(config: dict):
    """Return a preprocessing_function compatible with ImageDataGenerator."""
    def fn(img_rgb: np.ndarray) -> np.ndarray:
        # ImageDataGenerator passes float32 RGB [0,255] — convert to BGR uint8
        bgr = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        bgr = apply_preprocessing(bgr, config)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return fn


def build_generators(augment: bool = True, preprocessing_config: dict | None = None):
    """
    Build train, validation, and test generators.

    Args:
        augment:               Apply data augmentation on training set.
        preprocessing_config:  If provided, apply classical preprocessing
                               inside the generator (default OFF to preserve
                               existing training results).

    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    pre_fn = _make_preprocessing_fn(preprocessing_config) if preprocessing_config else None

    # ── Training generator (with optional augmentation) ────────────────────
    if augment:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            horizontal_flip=True,
            rotation_range=15,
            brightness_range=[0.8, 1.2],
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode="nearest",
            preprocessing_function=pre_fn,
        )
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            preprocessing_function=pre_fn,
        )

    # ── Val / Test generator (no augmentation, only rescale) ──────────────
    eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=pre_fn,
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",    # 0 = WithMask, 1 = WithoutMask
        shuffle=True,
        seed=42,
    )

    val_gen = eval_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    print(f"Train samples   : {train_gen.samples}")
    print(f"Val samples     : {val_gen.samples}")
    print(f"Test samples    : {test_gen.samples}")
    print(f"Class indices   : {train_gen.class_indices}")

    return train_gen, val_gen, test_gen

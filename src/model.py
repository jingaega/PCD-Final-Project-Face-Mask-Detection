"""
model.py — MobileNetV2 architecture with custom binary classification head.

Two build modes:
  - trainable_base=False  →  Phase 1: only the new head is trained
  - trainable_base=True   →  Phase 2: upper MobileNetV2 layers are unfrozen
"""

import tensorflow as tf
from src.config import IMG_SHAPE, UNFREEZE_FROM, LR_HEAD, LR_FINETUNE


def build_model(trainable_base: bool = False) -> tf.keras.Model:
    """
    Build MobileNetV2 with a custom binary classification head.

    Args:
        trainable_base: If True, unfreeze layers from UNFREEZE_FROM onward.

    Returns:
        Compiled Keras model.
    """
    # ── Base model ────────────────────────────────────────────────────────
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,        # remove ImageNet classifier
        weights="imagenet",       # pretrained weights
    )

    # Freeze / partially unfreeze
    base.trainable = trainable_base
    if trainable_base:
        for layer in base.layers[:UNFREEZE_FROM]:
            layer.trainable = False
        unfrozen = sum(1 for l in base.layers if l.trainable)
        print(f"Unfrozen layers in base: {unfrozen}")

    # ── Custom head ───────────────────────────────────────────────────────
    inputs  = tf.keras.Input(shape=IMG_SHAPE, name="input_image")
    x       = base(inputs, training=trainable_base)
    x       = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x       = tf.keras.layers.Dropout(0.3, name="dropout_1")(x)
    x       = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x       = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="output"
    )(x)   # sigmoid → binary prob; WithMask if < 0.5

    model = tf.keras.Model(inputs, outputs, name="MaskDetector")

    # ── Compile ───────────────────────────────────────────────────────────
    lr = LR_FINETUNE if trainable_base else LR_HEAD
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

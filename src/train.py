"""
train.py — Two-phase training pipeline.

Phase 1: Train only the classification head (base frozen).
Phase 2: Fine-tune upper MobileNetV2 layers with a lower learning rate.

Usage:
    python -m src.train
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import (
    MODEL_SAVE_PATH, MODEL_DIR,
    EPOCHS_HEAD, EPOCHS_TUNE,
    LR_FINETUNE,
)
from src.dataset import build_generators
from src.model import build_model


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(monitor_metric: str = "val_accuracy") -> list:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor=monitor_metric,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_history(history: tf.keras.callbacks.History, title: str = "Training") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    save_path = MODEL_DIR / f"{title.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved -> {save_path}")


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> tf.keras.Model:
    train_gen, val_gen, _ = build_generators(augment=True)

    # ── Phase 1: head only ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1 — Training classification head (base frozen)")
    print("=" * 60)

    model = build_model(trainable_base=False)
    model.summary()

    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks(),
    )
    plot_history(h1, title="Phase 1 Head Training")

    # ── Phase 2: fine-tune upper layers ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2 — Fine-tuning upper MobileNetV2 layers")
    print("=" * 60)

    model = build_model(trainable_base=True)

    # Load best weights from Phase 1 before continuing
    model.load_weights(str(MODEL_SAVE_PATH))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINETUNE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_TUNE,
        callbacks=get_callbacks(),
    )
    plot_history(h2, title="Phase 2 Fine Tuning")

    print(f"\nBest model saved -> {MODEL_SAVE_PATH}")
    return model


if __name__ == "__main__":
    train()

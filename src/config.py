"""
config.py — Central configuration for Face Mask Detection project.
Edit values here; everything else imports from this file.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data" / "raw"
TRAIN_DIR = DATA_DIR / "Train"
VAL_DIR   = DATA_DIR / "Validation"
TEST_DIR  = DATA_DIR / "Test"

MODEL_DIR       = BASE_DIR / "models" / "saved"
FACE_DET_DIR    = BASE_DIR / "models" / "face_detector"
MODEL_SAVE_PATH = MODEL_DIR / "mobilenetv2_mask.keras"

# Create output dirs if they don't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FACE_DET_DIR.mkdir(parents=True, exist_ok=True)

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES     = ["WithMask", "WithoutMask"]   # must match folder names in dataset
NUM_CLASSES = len(CLASSES)                  # 2 → binary classification

# ── Image ─────────────────────────────────────────────────────────────────────
IMG_SIZE  = (224, 224)       # MobileNetV2 default input
IMG_SHAPE = (224, 224, 3)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS_HEAD   = 20    # Phase 1: train head only
EPOCHS_TUNE   = 10    # Phase 2: fine-tune upper layers
LR_HEAD       = 1e-3  # higher LR for fresh head
LR_FINETUNE   = 1e-5  # lower LR for fine-tuning

# ── MobileNetV2 Fine-tuning ───────────────────────────────────────────────────
# MobileNetV2 has 155 layers total.
# Freeze everything below this index; unfreeze from here onward.
UNFREEZE_FROM = 100

# ── Inference ─────────────────────────────────────────────────────────────────
FACE_CONF_THRESHOLD = 0.5   # minimum confidence for SSD face detections
MASK_THRESHOLD      = 0.5   # sigmoid threshold → WithMask if prob > 0.5

# ── Classical preprocessing pipeline ─────────────────────────────────────────
# Applied to face crops at inference time (predict.py).
# Does NOT affect the trained model weights — only inference preprocessing.
PREPROCESSING_CONFIG = {
    "gaussian":          True,   # Gaussian blur for noise reduction
    "median":            False,  # Median filter (salt-and-pepper noise)
    "clahe":             True,   # CLAHE adaptive histogram equalization
    "contrast_stretch":  False,  # Min-max contrast stretching
}

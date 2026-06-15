# Face Mask Detection

Binary face mask classification using **MobileNetV2 + OpenCV SSD**.  
Built for **Digital Image Processing**, Telkom University.

---

## Project Structure

```
face_mask_detection/
├── data/
│   └── raw/                   ← put Kaggle dataset here
│       ├── Train/
│       │   ├── WithMask/
│       │   └── WithoutMask/
│       ├── Validation/
│       │   ├── WithMask/
│       │   └── WithoutMask/
│       └── Test/
│           ├── WithMask/
│           └── WithoutMask/
├── models/
│   ├── saved/                 ← trained model + plots saved here
│   └── face_detector/         ← OpenCV SSD weights (auto-downloaded)
├── src/
│   ├── config.py              ← all settings in one place
│   ├── dataset.py             ← data loading + augmentation
│   ├── model.py               ← MobileNetV2 architecture
│   ├── train.py               ← two-phase training loop
│   ├── evaluate.py            ← metrics + confusion matrix
│   └── predict.py             ← inference + webcam
├── app/
│   └── app.py                 ← Streamlit UI
├── scripts/
│   └── download_face_detector.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
1. Go to: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset
2. Download and extract into `data/raw/` so that the folder structure matches above.

### 4. Download the OpenCV face detector
```bash
python scripts/download_face_detector.py
```

---

## Usage

### Train
```bash
python -m src.train
```
Runs Phase 1 (head only) then Phase 2 (fine-tuning).  
Best model is saved automatically to `models/saved/mobilenetv2_mask.keras`.

### Evaluate
```bash
python -m src.evaluate
```
Prints classification report and saves `confusion_matrix.png`.

### Webcam (live detection)
```bash
python -m src.predict
```
Opens an OpenCV window with real-time detection. Press `q` to quit.

### Streamlit app (image upload)
```bash
streamlit run app/app.py
```

---

## Configuration

All hyperparameters and paths live in `src/config.py`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS_HEAD` | 20 | Phase 1 epochs |
| `EPOCHS_TUNE` | 10 | Phase 2 epochs |
| `LR_HEAD` | 1e-3 | Phase 1 learning rate |
| `LR_FINETUNE` | 1e-5 | Phase 2 learning rate |
| `UNFREEZE_FROM` | 100 | MobileNetV2 layer index to start unfreezing |
| `FACE_CONF_THRESHOLD` | 0.5 | SSD face detection confidence |

"""
download_face_detector.py — Download OpenCV SSD face detector weights.

Downloads two files into models/face_detector/:
  - deploy.prototxt
  - res10_300x300_ssd_iter_140000.caffemodel

Usage:
    python scripts/download_face_detector.py
"""

import urllib.request
from pathlib import Path

DETECTOR_DIR = Path(__file__).resolve().parent.parent / "models" / "face_detector"
DETECTOR_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}


def download_all() -> None:
    for filename, url in FILES.items():
        dest = DETECTOR_DIR / filename
        if dest.exists():
            print(f"  ✓ Already exists: {filename}")
            continue
        print(f"  Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1_048_576
        print(f"    Saved ({size_mb:.1f} MB) → {dest}")
    print("\nDone. Face detector ready.")


if __name__ == "__main__":
    download_all()

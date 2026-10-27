"""
download_face_detector.py
--------------------------
Downloads the OpenCV Caffe SSD face detector files into face_detector/.

These are the two files needed:
  • deploy.prototxt
  • res10_300x300_ssd_iter_140000.caffemodel

Run once before detect_mask_webcam.py:
    python download_face_detector.py
"""

import os
import urllib.request

FACE_DIR = os.path.join(os.path.dirname(__file__), "face_detector")
os.makedirs(FACE_DIR, exist_ok=True)

FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar = "#" * int(pct // 2)
        print(f"\r    [{bar:<50}] {pct:5.1f}%", end="", flush=True)


def main():
    print("=" * 60)
    print("  Downloading OpenCV Face Detector (Caffe SSD)")
    print("=" * 60)

    for fname, url in FILES.items():
        dest = os.path.join(FACE_DIR, fname)
        if os.path.isfile(dest):
            print(f"[SKIP] {fname} already exists.")
            continue
        print(f"[INFO] Downloading {fname} …")
        urllib.request.urlretrieve(url, dest, show_progress)
        print()   # newline after progress bar
        print(f"[OK]  Saved to: {dest}")

    print("\n[DONE] Face detector files ready in face_detector/")


if __name__ == "__main__":
    main()

# 😷 Face Mask Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.6%2B-5C3EE8?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Accuracy-~98%25-brightgreen" />
  <img src="https://img.shields.io/badge/Real--Time-Webcam-red?logo=youtube&logoColor=white" />
</p>

<p align="center">
  A real-time face mask detection system built with <strong>Deep Learning</strong> using <strong>MobileNetV2 Transfer Learning</strong> and <strong>OpenCV</strong>. Detects whether a person is wearing a face mask via laptop webcam — built as a Final Year CSE PBL Project.
</p>

---

## 📸 Demo

| With Mask ✅ | Without Mask ❌ |
|:---:|:---:|
| 🟢 Green bounding box | 🔴 Red bounding box |
| Confidence % displayed | Confidence % displayed |

---

## 🧠 How It Works

This project uses a **two-stage detection pipeline**:

```
📷 Webcam Frame
       │
       ▼
┌─────────────────────────────────┐
│  Stage 1: Face Detection        │
│  OpenCV DNN — Caffe SSD         │
│  ResNet-10 (pre-trained)        │
│  → Finds WHERE faces are        │
└──────────────┬──────────────────┘
               │  face bounding boxes
               ▼
┌─────────────────────────────────┐
│  Stage 2: Mask Classification   │
│  MobileNetV2 (Transfer Learn.)  │
│  Custom Head (trained by you)   │
│  → Mask ✓ or No Mask ✗          │
└──────────────┬──────────────────┘
               │
               ▼
     Annotated Live Feed
     (bounding box + label + %)
```

### Model Architecture

```
MobileNetV2 (ImageNet pretrained — FROZEN)
          │
    AveragePooling2D (7×7)
          │
        Flatten
          │
    Dense(128, ReLU)
          │
      Dropout(0.5)
          │
    Dense(2, Softmax)
          │
  [with_mask, without_mask]
```

---

## 📁 Project Structure

```
face_mask_detection/
│
├── 📂 dataset/
│   ├── with_mask/           ← training images (masked faces)
│   └── without_mask/        ← training images (unmasked faces)
│
├── 📂 face_detector/        ← OpenCV DNN files (auto-downloaded)
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── 📂 model/                ← saved model outputs after training
│   ├── mask_detector.h5
│   ├── training_plot.png
│   └── confusion_matrix.png
│
├── 📂 screenshots/          ← webcam screenshots (press S during demo)
│
├── 📄 train_model.py            ← train the DL model
├── 📄 detect_mask_webcam.py     ← MAIN: real-time webcam detection
├── 📄 evaluate_model.py         ← confusion matrix + metrics
├── 📄 download_dataset.py       ← auto-download training images
├── 📄 download_face_detector.py ← auto-download face detector weights
├── 📄 requirements.txt
├── 📄 run_all.bat               ← one-click Windows setup
└── 📄 README.md
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow 2.x / Keras |
| Base Model | MobileNetV2 (Transfer Learning, ImageNet) |
| Face Detection | OpenCV DNN — Caffe SSD ResNet-10 |
| Computer Vision | OpenCV 4.x |
| ML Utilities | scikit-learn |
| Visualization | Matplotlib, Seaborn |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or above → [Download](https://www.python.org/downloads/)
- A working webcam on your laptop
- ~500 MB free disk space

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/face-mask-detection.git
cd face-mask-detection
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3 — Download the face detector (one time, ~10 MB)

```bash
python download_face_detector.py
```

---

### Step 4 — Download training dataset

```bash
# Option A: Free GitHub mirror (~660 images, no login needed)
python download_dataset.py --github

# Option B: Full Kaggle dataset (~7,500 images, better accuracy)
# First: pip install kaggle  →  add kaggle.json to ~/.kaggle/
python download_dataset.py --kaggle
```

> Dataset source: [Face Mask Dataset — Kaggle (Omkar Gurav)](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

---

### Step 5 — Train the model

```bash
python train_model.py
```

- ⏳ Takes ~5–15 minutes on CPU (faster with GPU)
- Saves `model/mask_detector.h5` when complete
- Generates `model/training_plot.png` (accuracy & loss curves)

---

### Step 6 — Run real-time webcam detection 🎥

```bash
python detect_mask_webcam.py
```

**Webcam Controls:**

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot to `screenshots/` |

---

### Step 7 — Evaluate the model (optional, for report)

```bash
python evaluate_model.py
```

Generates `model/confusion_matrix.png` and prints Precision / Recall / F1-Score.

---

### ⚡ Windows One-Click Setup

Just double-click `run_all.bat` — it does all the above steps automatically.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~98–99% |
| Validation Accuracy | ~96–98% |
| Real-time FPS (CPU) | 10–20 fps |
| Model Size | ~10 MB |

---

## 📦 Dataset

- **Name**: Face Mask Dataset
- **Author**: Omkar Gurav
- **Source**: [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- **Size**: ~7,553 images (3,725 with mask + 3,828 without mask)
- **Classes**: `with_mask` | `without_mask`

> **Note**: The `dataset/` folder is empty in this repo (images are not committed to Git due to size). Run `download_dataset.py` to populate it.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named tensorflow` | `pip install tensorflow` |
| Webcam not opening | Change `cv2.VideoCapture(0)` → `(1)` in `detect_mask_webcam.py` |
| Model not found error | Run `train_model.py` before `detect_mask_webcam.py` |
| Training very slow | Normal on CPU. Try reducing `EPOCHS = 10` in `train_model.py` |
| Low detection accuracy | Use full Kaggle dataset instead of GitHub mirror |

---

## 📝 Project Report Notes

- **Two-stage pipeline**: face detection and mask classification are independent models
- **Transfer Learning**: MobileNetV2 base frozen; only custom head trained on mask data
- **Data Augmentation**: rotation, zoom, flip → prevents overfitting
- **MobileNetV2**: lightweight (3.4M params), ideal for real-time CPU inference
- **Binary Cross-Entropy** loss used (two-class problem)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Adrian Rosebrock — PyImageSearch](https://www.pyimagesearch.com/) — inspiration for the pipeline
- [OpenCV DNN Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
- [Omkar Gurav — Kaggle Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- [MobileNetV2 — Google Research](https://arxiv.org/abs/1801.04381)

---

<p align="center">Made with ❤️ for Final Year CSE PBL | Deep Learning Project</p>

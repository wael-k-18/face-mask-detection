"""
evaluate_model.py
-----------------
Detailed evaluation of the trained model on the test set.
Produces:
  • Confusion matrix (model/confusion_matrix.png)
  • Classification report printed to console

Usage:
    python evaluate_model.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH  = os.path.join(BASE_DIR, "model", "mask_detector.h5")
CM_PATH     = os.path.join(BASE_DIR, "model", "confusion_matrix.png")

CATEGORIES  = ["with_mask", "without_mask"]
IMG_SIZE    = (224, 224)


def load_dataset():
    data, labels = [], []
    for cat in CATEGORIES:
        folder = os.path.join(DATASET_DIR, cat)
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = load_img(os.path.join(folder, fname), target_size=IMG_SIZE)
            img = img_to_array(img)
            img = preprocess_input(img)
            data.append(img)
            labels.append(cat)
    return np.array(data, dtype="float32"), np.array(labels)


def main():
    print("[INFO] Loading dataset …")
    data, raw_labels = load_dataset()

    lb     = LabelBinarizer()
    labels = lb.fit_transform(raw_labels)
    labels = to_categorical(labels)

    _, test_x, _, test_y = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42
    )

    print("[INFO] Loading model …")
    model = load_model(MODEL_PATH)

    print("[INFO] Predicting …")
    preds     = model.predict(test_x, batch_size=32)
    pred_idxs = np.argmax(preds, axis=1)
    true_idxs = np.argmax(test_y, axis=1)

    # ── Classification report ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Classification Report")
    print("=" * 50)
    print(classification_report(true_idxs, pred_idxs, target_names=CATEGORIES))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(true_idxs, pred_idxs)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CATEGORIES, yticklabels=CATEGORIES,
    )
    plt.title("Confusion Matrix – Face Mask Detector")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(CM_PATH)
    print(f"[OK]  Confusion matrix saved to: {CM_PATH}")


if __name__ == "__main__":
    main()

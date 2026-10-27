"""
train_model.py
--------------
Trains a MobileNetV2-based face mask classifier using Transfer Learning.

Usage:
    python train_model.py

Output:
    model/mask_detector.h5  – saved Keras model
    model/training_plot.png – accuracy / loss curves
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend (no display needed)
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    AveragePooling2D, Dropout, Flatten, Dense, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_DIR     = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH    = os.path.join(MODEL_DIR, "mask_detector.h5")
PLOT_PATH     = os.path.join(MODEL_DIR, "training_plot.png")

CATEGORIES    = ["with_mask", "without_mask"]   # class names
IMG_SIZE      = (224, 224)   # MobileNetV2 default input
INIT_LR       = 1e-4         # initial learning rate
EPOCHS        = 20
BATCH_SIZE    = 32

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Load & pre-process images
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset():
    data, labels = [], []
    for category in CATEGORIES:
        folder = os.path.join(DATASET_DIR, category)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"[ERROR] Folder not found: {folder}\n"
                f"        Run 'python download_dataset.py --github' first."
            )
        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        print(f"[INFO] Loading {len(files):>4} images from '{category}' …")
        for fname in files:
            path  = os.path.join(folder, fname)
            image = load_img(path, target_size=IMG_SIZE)
            image = img_to_array(image)
            image = preprocess_input(image)   # MobileNetV2 preprocessing
            data.append(image)
            labels.append(category)

    data   = np.array(data,   dtype="float32")
    labels = np.array(labels)
    return data, labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Encode labels & split dataset
# ─────────────────────────────────────────────────────────────────────────────
def prepare_labels(labels):
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    return labels


def split_data(data, labels):
    return train_test_split(
        data, labels,
        test_size=0.20,
        stratify=labels,
        random_state=42,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Build model (MobileNetV2 + custom head)
# ─────────────────────────────────────────────────────────────────────────────
def build_model():
    # Load MobileNetV2 without top classification layers
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3)),
    )
    # Freeze all base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Build custom head
    head = base_model.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten()(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(2, activation="softmax")(head)

    model = Model(inputs=base_model.input, outputs=head)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Data augmentation generator
# ─────────────────────────────────────────────────────────────────────────────
def get_augmentor():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Train & save
# ─────────────────────────────────────────────────────────────────────────────
def train_and_save(model, train_x, train_y, test_x, test_y):
    aug = get_augmentor()
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=["accuracy"],
    )

    print("[INFO] Training model …")
    H = model.fit(
        aug.flow(train_x, train_y, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, len(train_x) // BATCH_SIZE),
        validation_data=(test_x, test_y),
        validation_steps=max(1, len(test_x) // BATCH_SIZE),
        epochs=EPOCHS,
    )

    # ── Evaluation ──────────────────────────────────────────────────────────
    print("[INFO] Evaluating model …")
    preds = model.predict(test_x, batch_size=BATCH_SIZE)
    preds = np.argmax(preds, axis=1)
    true  = np.argmax(test_y, axis=1)
    print(classification_report(true, preds, target_names=CATEGORIES))

    # ── Save model ──────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"[OK]  Model saved to: {MODEL_PATH}")

    # ── Plot ────────────────────────────────────────────────────────────────
    N = EPOCHS
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch #")
    axes[0].set_ylabel("Loss")
    axes[0].plot(np.arange(N), H.history["loss"],     label="train_loss")
    axes[0].plot(np.arange(N), H.history["val_loss"], label="val_loss")
    axes[0].legend()

    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch #")
    axes[1].set_ylabel("Accuracy")
    axes[1].plot(np.arange(N), H.history["accuracy"],     label="train_acc")
    axes[1].plot(np.arange(N), H.history["val_accuracy"], label="val_acc")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"[OK]  Training plot saved to: {PLOT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Face Mask Detector – Model Training")
    print("=" * 60)

    data, labels = load_dataset()
    labels = prepare_labels(labels)
    train_x, test_x, train_y, test_y = split_data(data, labels)

    print(f"[INFO] Train samples : {len(train_x)}")
    print(f"[INFO] Test  samples : {len(test_x)}")

    model = build_model()
    train_and_save(model, train_x, train_y, test_x, test_y)

    print("\n[DONE] Training complete. Run 'python detect_mask_webcam.py' to start detection.")

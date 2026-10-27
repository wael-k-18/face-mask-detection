"""
download_dataset.py
-------------------
Helper script to download the Face Mask Dataset from Kaggle or a public mirror.

OPTION 1 (Recommended) – Kaggle API:
    pip install kaggle
    Place your kaggle.json in ~/.kaggle/
    Then run: python download_dataset.py --kaggle

OPTION 2 – Manual download:
    1. Go to https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
    2. Download and unzip into the dataset/ folder so the structure looks like:
        dataset/
            with_mask/    ← images of people wearing masks
            without_mask/ ← images of people without masks

OPTION 3 – Use the public GitHub mirror (no account needed):
    python download_dataset.py --github
"""

import argparse
import os
import sys
import zipfile
import urllib.request

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
WITH_MASK_DIR = os.path.join(DATASET_DIR, "with_mask")
WITHOUT_MASK_DIR = os.path.join(DATASET_DIR, "without_mask")

# Public mirror (small subset ~400 images, enough to test training)
GITHUB_ZIP_URL = (
    "https://github.com/balajisrinivas/Face-Mask-Detection/archive/refs/heads/master.zip"
)


def _count_images(folder: str) -> int:
    exts = {".jpg", ".jpeg", ".png"}
    return sum(
        1
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )


def download_via_github():
    """Download the Face-Mask-Detection GitHub repo and copy dataset images."""
    zip_path = os.path.join(DATASET_DIR, "repo.zip")
    extract_dir = os.path.join(DATASET_DIR, "_tmp_repo")

    print("[INFO] Downloading from GitHub …")
    urllib.request.urlretrieve(GITHUB_ZIP_URL, zip_path)
    print("[INFO] Extracting …")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    import shutil

    # Locate images inside extracted repo
    repo_root = os.path.join(extract_dir, "Face-Mask-Detection-master")
    src_with    = os.path.join(repo_root, "dataset", "with_mask")
    src_without = os.path.join(repo_root, "dataset", "without_mask")

    if not os.path.isdir(src_with) or not os.path.isdir(src_without):
        print("[ERROR] Could not find dataset folders inside the downloaded repo.")
        print("        Please download manually from Kaggle.")
        sys.exit(1)

    for fname in os.listdir(src_with):
        shutil.copy2(os.path.join(src_with, fname), os.path.join(WITH_MASK_DIR, fname))
    for fname in os.listdir(src_without):
        shutil.copy2(os.path.join(src_without, fname), os.path.join(WITHOUT_MASK_DIR, fname))

    # Cleanup
    shutil.rmtree(extract_dir)
    os.remove(zip_path)

    print(f"[INFO] with_mask   : {_count_images(WITH_MASK_DIR):>5} images")
    print(f"[INFO] without_mask: {_count_images(WITHOUT_MASK_DIR):>5} images")
    print("[OK]  Dataset ready in ./dataset/")


def download_via_kaggle():
    """Use the Kaggle API to download the dataset."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[ERROR] kaggle package not found. Run: pip install kaggle")
        sys.exit(1)

    import subprocess

    print("[INFO] Downloading from Kaggle …")
    subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "omkargurav/face-mask-dataset",
            "-p", DATASET_DIR, "--unzip",
        ],
        check=True,
    )
    print("[OK]  Dataset downloaded.")


def main():
    parser = argparse.ArgumentParser(description="Download face-mask dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--github", action="store_true", help="Download from GitHub mirror")
    group.add_argument("--kaggle", action="store_true", help="Download via Kaggle API")
    args = parser.parse_args()

    os.makedirs(WITH_MASK_DIR, exist_ok=True)
    os.makedirs(WITHOUT_MASK_DIR, exist_ok=True)

    if args.github:
        download_via_github()
    else:
        download_via_kaggle()


if __name__ == "__main__":
    main()

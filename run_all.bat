@echo off
title Face Mask Detection – Setup & Run
echo ============================================================
echo   Face Mask Detection System - Quick Start
echo ============================================================

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Install from https://python.org
    pause
    exit /b 1
)

echo [STEP 1] Installing dependencies ...
pip install -r requirements.txt --quiet

echo [STEP 2] Downloading face detector ...
python download_face_detector.py

echo [STEP 3] Downloading dataset (GitHub mirror) ...
python download_dataset.py --github

echo [STEP 4] Training model (this takes a few minutes) ...
python train_model.py

echo [STEP 5] Starting webcam detection ...
python detect_mask_webcam.py

pause

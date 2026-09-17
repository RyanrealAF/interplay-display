#!/usr/bin/env bash
set -e

echo "=== Interplay-Display Setup Script ==="

# 1. Install System Dependencies
echo "[1/3] Checking and installing system dependencies (ffmpeg, libsndfile1)..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y ffmpeg libsndfile1
else
    echo "Warning: apt-get not found. Please ensure ffmpeg and libsndfile1 are installed on your system."
fi

# 2. Install Python Dependencies
echo "[2/3] Installing Python dependencies from requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 3. Create Required Output Directories
echo "[3/3] Creating output directory for audio stems..."
mkdir -p /tmp/stemsplitter_outputs
chmod 777 /tmp/stemsplitter_outputs || true

echo "=== Setup Complete! ==="
echo "To run the API server:"
echo "  uvicorn api:app --host 0.0.0.0 --port 7860"

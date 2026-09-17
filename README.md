---
title: Interplay Display
emoji: 👀
colorFrom: pink
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: Backend API for audio stem separation and DSP interplay analytics
---

# Interplay-Display SaaS Engine API

Backend Python/PyTorch API hosted on Hugging Face Spaces that separates audio into 6 stems (Vocals, Drums, Bass, Guitar, Piano, Other) using Demucs and performs DSP analytics on instrument interplay (rhythmic lock, vocal masking, harmonic clashing).

## Requirements

### System Dependencies
- `ffmpeg`
- `libsndfile1`

On Ubuntu/Debian:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1
```

### Python Dependencies
Install dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Running the API

```bash
uvicorn api:app --host 0.0.0.0 --port 7860
```

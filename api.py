import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import librosa
import scipy.signal
import scipy.stats
import torch
import demucs.api

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Hardware Acceleration & Model Pre-loading
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "htdemucs_6s"

# Pre-load Demucs separator into RAM/VRAM
separator = demucs.api.Separator(model=MODEL_NAME, device=DEVICE)

app = FastAPI(title="Interplay-Display SaaS Engine API")

# --- CORS CONFIGURATION ---
# This allows your custom React/Next.js frontend to talk to this HF Space.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your actual frontend URL (e.g., https://your-vercel-app.com)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the temporary directory so the frontend can download the audio files
TEMP_DIR = Path("/tmp/stemsplitter_outputs")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=TEMP_DIR), name="outputs")

# ---------------------------------------------------------------------------
# DSP Logic: The Second Pass & Interplay Analytics
# ---------------------------------------------------------------------------
def multi_pass_bleed_reduction(separated_stems: dict, n_fft=2048, hop_length=512, isolation_strength=1.5):
    """
    THE SECOND PASS: Compares all stems against each other simultaneously.
    Forces mutual exclusivity in the frequency domain to massively reduce bleed.
    isolation_strength > 1.0 increases bleed reduction.
    """
    print("Initiating Second Pass: Global Cross-Stem Spectral Analysis...")

    # Get the hardware device (CPU/CUDA) from the first tensor
    device = next(iter(separated_stems.values())).device
    window = torch.hann_window(n_fft).to(device)

    stfts = {}
    mags = {}
    phases = {}

    # 1. Convert ALL stems into the frequency domain simultaneously
    for name, waveform in separated_stems.items():
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        stfts[name] = stft
        mags[name] = torch.abs(stft)
        phases[name] = torch.angle(stft)

    # 2. Map the total Global Energy of the entire mix at every frequency bin
    eps = 1e-9
    total_energy = sum(mags.values()) + eps

    refined_stems = {}

    # 3. Analyze each stem against the global total and aggressively cut bleed
    for name in separated_stems.keys():
        # Calculate the ratio of THIS stem's energy vs ALL stems' energy combined
        ratio = mags[name] / total_energy

        # Apply the isolation exponent to squash bleed while preserving dominant frequencies
        mask = ratio ** isolation_strength

        # Reconstruct the cleaned, isolated audio
        cleaned_mag = mags[name] * mask
        cleaned_stft = torch.polar(cleaned_mag, phases[name])

        # Convert back to audio waveform
        refined_waveform = torch.istft(
            cleaned_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            length=separated_stems[name].shape[-1]
        )

        refined_stems[name] = refined_waveform

    return refined_stems


def analyze_granular_interplay(stems_mono: dict, sr: int):
    """Calculates DSP Analytics natively for JSON transport (No UI rendering)"""
    interplay_metrics = {}

    # 1. Rhythmic Lock (Drums vs Bass)
    if 'drums' in stems_mono and 'bass' in stems_mono:
        drum_onset = librosa.onset.onset_strength(y=stems_mono['drums'], sr=sr)
        bass_onset = librosa.onset.onset_strength(y=stems_mono['bass'], sr=sr)
        lock_score, _ = scipy.stats.pearsonr(drum_onset, bass_onset)
        if np.isnan(lock_score):
            lock_score = 0.0
        interplay_metrics['rhythmic_lock_score'] = round(float(lock_score), 4)

    # 2. Vocal Masking
    if 'vocals' in stems_mono:
        stft_vocals = np.abs(librosa.stft(stems_mono['vocals']))
        masking_pct = {}
        for inst, y_inst in stems_mono.items():
            if inst != 'vocals':
                stft_inst = np.abs(librosa.stft(y_inst))
                hits = np.sum(stft_inst > stft_vocals)
                total = stft_vocals.size
                pct = float((hits / total) * 100) if total > 0 else 0.0
                masking_pct[inst] = pct
        interplay_metrics['vocal_masking_pct'] = {k: round(float(v), 2) for k, v in masking_pct.items()}

    # 3. Harmonic Clashing Matrix (Averaged over time for JSON efficiency)
    if 'guitar' in stems_mono and 'piano' in stems_mono:
        chroma_g = librosa.feature.chroma_cqt(y=stems_mono['guitar'], sr=sr)
        chroma_p = librosa.feature.chroma_cqt(y=stems_mono['piano'], sr=sr)
        clash_matrix = np.abs(chroma_g - chroma_p)
        # Average the clash per pitch class (C, C#, D, etc.) over the whole song
        mean_clash_per_note = np.mean(clash_matrix, axis=1)

        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        interplay_metrics['harmonic_clash_profile'] = {
            note: round(float(val), 4) for note, val in zip(notes, mean_clash_per_note)
        }

    return interplay_metrics

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "online", "model": MODEL_NAME, "device": DEVICE, "service": "Interplay-Display SaaS Engine API"}

@app.post("/api/v1/analyze")
async def analyze_audio(request: Request, file: UploadFile = File(...)):
    """
    Main processing endpoint. Accepts audio, runs demixing & 2nd Pass DSP, and returns static URLs & JSON analytics.
    """
    try:
        # Generate base URL for static file serving dynamically
        base_url = str(request.base_url)

        input_path = TEMP_DIR / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Processing '{file.filename}' through Demucs...")
        # 1. Neural Demixing (First Pass)
        origin, separated = separator.separate_audio_file(input_path)
        sr = separator._samplerate

        # 2. The Second Pass: Global Cross-Stem Analysis & Bleed Reduction
        # Compares all 6 stems against each other to surgically eliminate frequency bleed.
        separated = multi_pass_bleed_reduction(separated, isolation_strength=1.5)

        stems_mono = {}
        download_urls = {}

        print("Extracting Audio and Calculating Analytics...")
        # 3. Export Stems to Static Directory
        for stem_name, source_tensor in separated.items():
            stem_file_name = f"{input_path.stem}_{stem_name}.wav"
            stem_file_path = TEMP_DIR / stem_file_name

            # Save audio to disk for the frontend to download
            demucs.api.save_audio(source_tensor, str(stem_file_path), samplerate=sr)
            download_urls[stem_name] = f"{base_url}outputs/{stem_file_name}"

            # Convert to mono for DSP analytics math
            audio_np = source_tensor.cpu().numpy()
            y_mono = np.mean(audio_np, axis=0) if audio_np.ndim > 1 else audio_np
            stems_mono[stem_name] = y_mono

        # 4. DSP Analytics
        interplay_metrics = analyze_granular_interplay(stems_mono, sr)

        # 5. Zip Creation (Bundle the pristine stems together)
        zip_filename = f"{input_path.stem}_bundle.zip"
        zip_path = TEMP_DIR / zip_filename
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for stem_name in download_urls.keys():
                file_to_zip = TEMP_DIR / f"{input_path.stem}_{stem_name}.wav"
                zf.write(file_to_zip, arcname=f"stems/{stem_name}.wav")

        download_urls["production_bundle_zip"] = f"{base_url}outputs/{zip_filename}"

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        print("Pipeline Complete! Handing payload back to Frontend.")

        # 6. JSON Response Handoff to Custom Frontend
        return JSONResponse(content={
            "status": "success",
            "source_file": file.filename,
            "downloads": download_urls,
            "analytics": {
                "interplay_analytics": interplay_metrics
            }
        })

    except Exception as e:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces strictly listens on port 7860 for Docker environments
    uvicorn.run(app, host="0.0.0.0", port=7860)

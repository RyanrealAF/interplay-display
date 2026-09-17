import io
import pytest
import numpy as np
import torch
import soundfile as sf
from fastapi.testclient import TestClient

from api import app, multi_pass_bleed_reduction, analyze_granular_interplay

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "htdemucs" in data["model"]

def test_multi_pass_bleed_reduction():
    sr = 44100
    duration = 1.0  # seconds
    n_samples = int(sr * duration)

    # Generate dummy stereo stems (channels=2, samples=n_samples)
    stems = {
        "vocals": torch.randn(2, n_samples),
        "drums": torch.randn(2, n_samples),
        "bass": torch.randn(2, n_samples),
        "guitar": torch.randn(2, n_samples),
        "piano": torch.randn(2, n_samples),
        "other": torch.randn(2, n_samples),
    }

    refined = multi_pass_bleed_reduction(stems, isolation_strength=1.5)
    assert set(refined.keys()) == set(stems.keys())
    for k, tensor in refined.items():
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, n_samples)

def test_analyze_granular_interplay():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Generate synthetic audio signals
    stems_mono = {
        "vocals": np.sin(2 * np.pi * 440 * t),
        "drums": np.random.randn(len(t)),
        "bass": np.sin(2 * np.pi * 110 * t),
        "guitar": np.sin(2 * np.pi * 330 * t),
        "piano": np.sin(2 * np.pi * 554.37 * t),
        "other": np.zeros(len(t)),
    }

    metrics = analyze_granular_interplay(stems_mono, sr)
    assert "rhythmic_lock_score" in metrics
    assert "vocal_masking_pct" in metrics
    assert "harmonic_clash_profile" in metrics
    assert isinstance(metrics["rhythmic_lock_score"], float)
    assert isinstance(metrics["vocal_masking_pct"], dict)
    assert isinstance(metrics["harmonic_clash_profile"], dict)

def test_analyze_endpoint_mock(tmp_path):
    # Create a small valid WAV file
    sr = 44100
    audio_data = (np.random.randn(sr * 2) * 0.1).astype(np.float32)
    wav_path = tmp_path / "test_song.wav"
    sf.write(str(wav_path), audio_data, sr)

    with open(wav_path, "rb") as f:
        response = client.post(
            "/api/v1/analyze",
            files={"file": ("test_song.wav", f, "audio/wav")}
        )

    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "success"
    assert json_resp["source_file"] == "test_song.wav"
    assert "downloads" in json_resp
    assert "analytics" in json_resp
    assert "interplay_analytics" in json_resp["analytics"]
    assert "vocals" in json_resp["downloads"]
    assert "production_bundle_zip" in json_resp["downloads"]

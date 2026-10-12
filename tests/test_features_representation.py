"""Test biểu diễn mel mới (log-magnitude + chuẩn hoá toàn cục) và F0 pyworld.

Xem `docs/REVISION_PLAN_V2.md` §2 — bước 2.
"""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vifasttts.data.features import estimate_f0_pyworld, extract_features, extract_mel_energy

MEL = {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20,
       "f_min": 0, "f_max": 12000, "sample_rate": 24000}


def _write_sine(path: Path, freq: float = 150.0, seconds: float = 0.5, sample_rate: int = 24000) -> None:
    time_axis = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * freq * time_axis)
    pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def test_log_magnitude_khac_log_power():
    torch.manual_seed(0)
    wav = torch.randn(4096) * 0.2
    power_mel, _ = extract_mel_energy(wav, MEL)
    mag_mel, _ = extract_mel_energy(wav, {**MEL, "log_magnitude": True})
    assert mag_mel.shape == power_mel.shape
    assert not torch.allclose(mag_mel, power_mel)
    # Im lặng hoàn toàn → mel chạm floor: log(1e-5) cho log-magnitude, log(1e-6) cho log-power.
    silence = torch.zeros(4096)
    mag_silence, _ = extract_mel_energy(silence, {**MEL, "log_magnitude": True})
    power_silence, _ = extract_mel_energy(silence, MEL)
    assert torch.allclose(mag_silence, torch.full_like(mag_silence, math.log(1e-5)), atol=1e-6)
    assert torch.allclose(power_silence, torch.full_like(power_silence, math.log(1e-6)), atol=1e-6)


def test_chuan_hoa_toan_cuc_theo_stats(tmp_path: Path):
    torch.manual_seed(0)
    wav = torch.randn(4096) * 0.2
    raw, _ = extract_mel_energy(wav, {**MEL, "log_magnitude": True})
    mean = raw.double().mean(dim=0)
    std = raw.double().std(dim=0).clamp_min(1e-4)
    stats = tmp_path / "mel_stats.json"
    stats.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}), encoding="utf-8")

    normalized, _ = extract_mel_energy(
        wav, {**MEL, "log_magnitude": True, "normalize_path": str(stats)}
    )
    assert torch.allclose(normalized.double().mean(dim=0), torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(normalized.double().std(dim=0), torch.ones_like(std), atol=1e-3)
    # Dùng mean=0, std=1 → không đổi giá trị (chỉ công thức (x-mean)/std).
    neutral = tmp_path / "neutral.json"
    neutral.write_text(json.dumps({"mean": [0.0] * 20, "std": [1.0] * 20}), encoding="utf-8")
    neutral_mel, _ = extract_mel_energy(
        wav, {**MEL, "log_magnitude": True, "normalize_path": str(neutral)}
    )
    assert torch.allclose(neutral_mel, raw, atol=1e-5)


def test_pyworld_f0_sine():
    pytest.importorskip("pyworld")
    sample_rate, hop = 24000, 256
    time_axis = torch.arange(sample_rate) / sample_rate
    # Thêm nhiễu nhẹ (thực tế hơn sine thuần — harvest bỏ qua sine thuần tuyệt đối).
    generator = torch.Generator().manual_seed(0)
    wav = 0.3 * torch.sin(2 * torch.pi * 150.0 * time_axis) + 0.01 * torch.randn(
        sample_rate, generator=generator
    )
    f0, vuv = estimate_f0_pyworld(wav, sample_rate, hop)
    assert f0.numel() == 1 + wav.numel() // hop
    assert vuv.numel() == f0.numel()
    voiced = f0[vuv > 0.5]
    assert voiced.numel() > f0.numel() // 2
    assert abs(float(voiced.median()) - 150.0) < 3.0


def test_extract_features_pyworld_du_do_dai(tmp_path: Path):
    pytest.importorskip("pyworld")
    wav_path = tmp_path / "a.wav"
    _write_sine(wav_path)
    config = {
        "sample_rate": 24000,
        "mel": {**MEL, "log_magnitude": True},
        "f0": {"method": "pyworld", "f_min": 70, "f_max": 400, "interpolate": True},
    }
    features = extract_features(wav_path, config)
    frames = features["mel"].size(0)
    assert frames > 10
    for name in ("mel", "energy", "f0", "vuv"):
        assert features[name].size(0) == frames
    assert float((features["f0"] > 0).float().mean()) > 0.5

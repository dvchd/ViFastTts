"""Test DCT4 (`data/features.py`) + `tools/compute_f0_stats.py`."""
from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vifasttts.data.features import dct4_frames, idct4_frames


def test_dct4_full_round_trip():
    values = torch.randn(16, dtype=torch.float64)
    coeffs = dct4_frames(values, keep=16)
    restored = idct4_frames(coeffs, 16)
    assert torch.allclose(restored, values, atol=1e-6)


def test_dct4_khong_co_cheo_truncated_van_trung_smooth():
    # x_t = cos(pi k (t+0.5)/n) với k=2 là ĐÚNG một basis DCT → 4 hệ số dựng lại khớp tuyệt đối.
    n, k = 24, 2
    positions = torch.arange(n, dtype=torch.float64)
    values = torch.cos(math.pi * k * (positions + 0.5) / n)
    coeffs = dct4_frames(values, keep=4)
    assert coeffs.numel() == 4
    restored = idct4_frames(coeffs, n)
    assert torch.allclose(restored, values, atol=1e-6)
    assert abs(float(coeffs[k]) - math.sqrt(n / 2.0)) < 1e-6  # trực giao: hệ số = sqrt(n/2)
    assert abs(float(coeffs[0])) < 1e-6


def test_dct4_ngan_hon_4_frame():
    values = torch.tensor([1.0, 2.0, 3.0])
    coeffs = dct4_frames(values)
    assert coeffs.numel() == 3
    restored = idct4_frames(coeffs, 3)
    assert torch.allclose(restored, values, atol=1e-6)


def _wav(path: Path, freq: float = 150.0, seconds: float = 1.0, silence: bool = False) -> None:
    time_axis = np.arange(int(seconds * 24000)) / 24000
    tone = np.zeros_like(time_axis) if silence else 0.3 * np.sin(2 * np.pi * freq * time_axis)
    pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24000)
        fh.writeframes(pcm.tobytes())


def test_compute_stats_tren_audio_tong_hop(tmp_path: Path):
    from tools.compute_f0_stats import compute_stats

    wavs = []
    for index in range(2):
        path = tmp_path / f"s{index}.wav"
        _wav(path, freq=140.0 + 20 * index)
        wavs.append(path)
    silence_wav = tmp_path / "sil.wav"
    _wav(silence_wav, silence=True)

    config = {
        "sample_rate": 24000,
        "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20, "f_min": 0, "f_max": 12000},
        "f0": {"method": "autocorr", "f_min": 70, "f_max": 400},
    }
    frames = 1 + 24000 // 64
    rows = [
        {"id": f"s{i}", "audio_path": str(wavs[i]), "text": "a b",
         "syllables": [{"raw": "a"}, {"raw": "b"}], "durations": [frames // 2, frames - frames // 2]}
        for i in range(2)
    ]
    rows.append({"id": "sil", "audio_path": str(silence_wav), "text": "a",
                 "syllables": [{"raw": "a"}], "durations": [frames]})
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    stats = compute_stats([str(manifest)], config, sample=10, workers=1)
    assert stats["clips"] == 3
    assert stats["low_voiced_clips"] == 1          # clip im lặng bị loại khỏi thống kê
    assert stats["voiced_frames"] > 0
    assert stats["syllables"] >= 4                  # 2 clip × 2 âm tiết (clip im lặng không tính)
    assert stats["f0_log_std"] > 0.0
    assert len(stats["dct_std"]) == 4 and stats["dct_std"][0] > 0.0
    assert stats["energy_count"] >= 4
    expected_mean = math.log(140.0)
    assert abs(stats["f0_log_mean"] - expected_mean) < 0.3  # ~ log(140..160)

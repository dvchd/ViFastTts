"""Test lọc rẻ B4 (`tools/clip_quality.py`): metric, ngưỡng, áp danh sách loại."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from tools.clip_quality import apply_drop, clip_metrics, quality_flags


def _wav(path: Path, samples: np.ndarray, sample_rate: int = 24000) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def _tone(seconds: float, amplitude: float = 0.3) -> np.ndarray:
    time_axis = np.arange(int(seconds * 24000)) / 24000
    return (amplitude * np.sin(2 * np.pi * 180.0 * time_axis)).astype(np.float32)


def test_metric_tone_giữ_và_im_lặng_loại():
    tone_metrics = clip_metrics(_tone(1.0).astype(np.float64), 24000)
    assert tone_metrics["speech_ratio"] > 0.8
    assert quality_flags(tone_metrics) == []

    quiet = _tone(1.0, amplitude=1e-4).astype(np.float64)
    quiet_metrics = clip_metrics(quiet, 24000)
    flags = quality_flags(quiet_metrics)
    assert "rms_thap" in flags


def test_clipping_bị_gắn_cờ():
    clipped = np.clip(_tone(1.0, amplitude=2.0), -1.0, 1.0).astype(np.float64)
    metrics = clip_metrics(clipped, 24000)
    assert metrics["clipping_ratio"] > 0.01
    assert "clipping" in quality_flags(metrics)


def test_apply_drop_ghi_đè_và_lưu_excluded(tmp_path: Path):
    manifest = tmp_path / "train.jsonl"
    rows = [
        {"id": "a_0001", "audio_path": str(tmp_path / "a.wav")},
        {"id": "b_0001", "audio_path": str(tmp_path / "b.wav")},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    drop = tmp_path / "drop.txt"
    drop.write_text("b_0001\n", encoding="utf-8")
    excluded = tmp_path / "excluded.jsonl"
    counts = apply_drop([str(manifest)], str(drop), str(excluded))
    assert counts["train.jsonl"] == 1
    kept = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in kept] == ["a_0001"]
    assert "b_0001" in excluded.read_text(encoding="utf-8")

"""Test cổng nghiệm thu vocoder (`tools/eval_vocoder.py`) — dùng chung cho mọi mốc R7-1."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.eval_vocoder import evaluate
from vifasttts.models.istftnet import build_generator

CONFIG = {
    "sample_rate": 24000,
    "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20,
            "f_min": 0, "f_max": 12000},
    "vocoder_training": {"generator": "lite", "hidden": 16, "upsample_ratio": 4},
}


def _wav(path: Path, seconds: float = 0.6) -> None:
    """Giọng giả có hài âm (pyworld harvest nhận là hữu thanh, khác sine thuần)."""
    time_axis = np.arange(int(seconds * 24000)) / 24000
    tone = sum((0.3 / k) * np.sin(2 * np.pi * 150.0 * k * time_axis + 0.1 * k) for k in range(1, 8))
    pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24000)
        fh.writeframes(pcm.tobytes())


def _checkpoint(tmp_path: Path) -> Path:
    generator = build_generator(CONFIG["vocoder_training"], {"n_mels": 20, "n_fft": 256, "hop_length": 64})
    path = tmp_path / "last.pt"
    torch.save({"vocoder": generator.state_dict(), "step": 0, "clean": True}, path)
    return path


def test_evaluate_xuat_wav_va_metrics(tmp_path: Path):
    rows = []
    for index in range(2):
        wav = tmp_path / f"s{index}.wav"
        _wav(wav)
        rows.append({"id": f"s{index}", "audio_path": str(wav)})
    output = tmp_path / "eval"
    aggregate = evaluate(CONFIG, _checkpoint(tmp_path), rows, output, clips=2, device="cpu")

    assert aggregate["clips"] == 2
    assert (output / "metrics.json").exists()
    for row in rows:
        assert (output / "wavs" / f"{row['id']}.wav").exists()
    assert aggregate["metrics"]["mel_l1"]["mean"] >= 0.0
    assert aggregate["metrics"]["length_ratio"]["mean"] > 0.0
    assert aggregate["per_clip"][0]["frames"] > 0
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8"))["clips"] == 2

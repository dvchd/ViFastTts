"""Test công cụ demo theo mốc (`tools/vocoder_milestones.py`)."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.vocoder_milestones import _read_step, run_milestone, select_clips
from vifasttts.models.istftnet import build_generator


def _wav(path: Path, seconds: float = 0.4) -> None:
    time_axis = np.arange(int(seconds * 24000)) / 24000
    tone = 0.3 * np.sin(2 * np.pi * 160.0 * time_axis)
    pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24000)
        fh.writeframes(pcm.tobytes())


CONFIG = {
    "sample_rate": 24000,
    "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20,
            "f_min": 0, "f_max": 12000},
    "vocoder_training": {"generator": "lite", "hidden": 16, "upsample_ratio": 4},
}


def _checkpoint(tmp_path: Path) -> Path:
    mel_config = {"n_mels": 20, "n_fft": 256, "hop_length": 64}
    generator = build_generator(CONFIG["vocoder_training"], mel_config)
    path = tmp_path / "last.pt"
    torch.save({"vocoder": generator.state_dict(), "step": 5000, "clean": True, "last_loss": 1.0}, path)
    return path


def test_select_clips_trải_đều_và_tất_định():
    rows = [{"id": f"x_{i:04d}"} for i in range(100)]
    picked = select_clips(rows, 10)
    assert len(picked) == 10
    assert picked == select_clips(rows, 10)
    assert picked[0]["id"] == "x_0000" and picked[1]["id"] == "x_0010"


def test_run_milestone_xuất_wav_và_metric(tmp_path: Path):
    rows = []
    for index in range(2):
        wav = tmp_path / f"s{index}.wav"
        _wav(wav)
        rows.append({"id": f"s{index}", "audio_path": str(wav)})
    checkpoint = _checkpoint(tmp_path)
    aggregate = run_milestone(CONFIG, rows, checkpoint, 5000, tmp_path / "demo", clips=2)
    assert aggregate["clips"] == 2 and aggregate["mel_l1_mean"] >= 0.0
    for row in rows:
        assert (tmp_path / "demo" / "step_5000" / f"{row['id']}.wav").exists()
    payload = json.loads((tmp_path / "demo" / "step_5000" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["step"] == 5000 and len(payload["per_clip"]) == 2
    assert 0.5 < payload["per_clip"][0]["length_ratio"] < 1.5


def test_read_step_từ_chối_checkpoint_bẩn(tmp_path: Path):
    clean = tmp_path / "clean.pt"
    torch.save({"step": 700, "clean": True}, clean)
    assert _read_step(clean) == 700
    dirty = tmp_path / "dirty.pt"
    torch.save({"step": 800, "clean": False}, dirty)
    assert _read_step(dirty) is None
    assert _read_step(tmp_path / "missing.pt") is None

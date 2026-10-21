"""Test chuẩn hoá độ to theo chunk trên log-mel (A0 v10, Việc 4/6d)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vifasttts.runtime.chunk_loudness import mel_level_db, normalize_chunk, voiced_frames


def _stats():
    stats = json.loads(Path("data/codelung/mel_stats.json").read_text(encoding="utf-8"))
    return np.asarray(stats["mean"], dtype=np.float64), np.asarray(stats["std"], dtype=np.float64)


def test_normalize_chunk_ve_dung_target() -> None:
    mean, std = _stats()
    rng = np.random.default_rng(0)
    mel = (rng.normal(0, 1, (80, 100)) * 0.5).astype(np.float32)
    target = 24.0
    shifted, info = normalize_chunk(mel, mean, std, target)
    level = mel_level_db(shifted, mean, std)
    achieved = float(np.median(level[voiced_frames(level)]))
    assert abs(achieved - target) < 0.05
    assert abs(info["shift_db"]) <= 4.0


def test_gioi_han_dich_4db() -> None:
    mean, std = _stats()
    mel = np.full((40, 100), -3.0, dtype=np.float32)  # rất nhỏ
    _, info = normalize_chunk(mel, mean, std, 24.0)
    assert info["shift_db"] == 4.0

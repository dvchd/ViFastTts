"""Test VAD nới biên clip (B3 — `_adjust_boundary`, `_clip_level_db`)."""
from __future__ import annotations

import numpy as np

from vifasttts.data.youtube import (
    BOUNDARY_DROP_DB,
    _adjust_boundary,
    _clip_level_db,
)

SR = 24000


def _tone(seconds: float, amplitude: float = 0.3, freq: float = 180.0) -> np.ndarray:
    time_axis = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * time_axis)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _threshold(samples: np.ndarray) -> float:
    return _clip_level_db(samples, SR) - BOUNDARY_DROP_DB


def test_nới_tới_mép_khoảng_lặng():
    # [lặng 0.5s][tone 1.0s] — biên đặt trong lặng, cách tiếng nói 0.1s
    samples = np.concatenate([_silence(0.5), _tone(1.0)])
    threshold = _threshold(_tone(1.0))
    boundary = 0.4  # trong lặng (tone bắt đầu 0.5)
    new_start, flag = _adjust_boundary(samples, SR, boundary, -1, threshold)
    assert flag is None
    # nới hết khoảng lặng về phía trước tới 0 (giới hạn 200 ms từ mốc 0.4 → 0.2)
    assert abs(new_start - 0.2) < 0.02


def test_băng_qua_âm_tiết_ngắn_rồi_tới_lặng():
    # [lặng 0.3][tone 0.5..0.62 ngắn][lặng 0.62..1.0][tone chính]
    tail = _tone(0.12)
    samples = np.concatenate([_silence(0.3), tail, _silence(0.38), _tone(1.0)])
    threshold = _threshold(_tone(1.0))
    # biên cắt vào đuôi âm tiết ngắn (0.3+0.12 = 0.42); đặt biên 0.40
    new_start, flag = _adjust_boundary(samples, SR, 0.40, -1, threshold)
    assert flag == "tight"
    # nới qua đuôi âm tiết (0.30..0.42) rồi đi hết khoảng lặng, chạm giới hạn 200 ms → 0.20
    assert abs(new_start - 0.20) < 0.02


def test_tiếng_nói_liền_giữ_nguyên_và_đánh_cờ():
    # tone liên tục 2s — biên giữa, phía ngoài toàn tiếng nói
    samples = np.concatenate([_silence(0.05), _tone(2.0), _silence(0.05)])
    threshold = _threshold(_tone(2.0))
    new_start, flag = _adjust_boundary(samples, SR, 1.0, -1, threshold)
    assert flag == "speech"
    assert abs(new_start - 1.0) < 1e-6


def test_biên_phải_mở_về_sau():
    samples = np.concatenate([_tone(1.0), _silence(0.6), _tone(0.5)])
    threshold = _threshold(_tone(1.0))
    new_end, flag = _adjust_boundary(samples, SR, 1.1, 1, threshold)
    assert flag is None
    # nới hết khoảng lặng phía sau (1.0..1.6), giới hạn 200 ms → 1.3
    assert abs(new_end - 1.3) < 0.02


def test_clip_level_db_ổn_định():
    level = _clip_level_db(_tone(1.0), SR)
    assert -20.0 < level < 0.0
    # lặng hoàn toàn → mức rất thấp (không NaN)
    quiet = _clip_level_db(_silence(1.0), SR)
    assert quiet < -100.0

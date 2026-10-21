"""Test chuẩn hoá loudness cả file (ITU-R BS.1770) — A0 v10, Việc 2/6f."""

from __future__ import annotations

import numpy as np
import pytest

from vifasttts.runtime.loudness import measure_lufs, normalize_loudness, true_peak_dbtp

SR = 24000


def _sine(amplitude: float = 0.1, seconds: float = 3.0, freq: float = 1000.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_measure_lufs_sine_khoang_23_dbfs() -> None:
    # Sine biên độ 0,1 → RMS −23 dBFS; K-weighting tại 1 kHz ≈ 0 dB.
    assert measure_lufs(_sine(0.1), SR) == pytest.approx(-23.0, abs=0.5)


def test_normalize_ve_target_va_tat_duoc() -> None:
    x = _sine(0.1)
    y, info = normalize_loudness(x, SR, target_lufs=-16.0)
    assert measure_lufs(y, SR) == pytest.approx(-16.0, abs=0.2)
    assert info["gain_db"] > 5.0
    off, info_off = normalize_loudness(x, SR, target_lufs=0)
    assert info_off["gain_db"] == 0.0
    assert np.array_equal(off, x)


def test_normalize_tat_dinh() -> None:
    x = _sine(0.05)
    a, _ = normalize_loudness(x, SR, target_lufs=-16.0)
    b, _ = normalize_loudness(x, SR, target_lufs=-16.0)
    assert np.array_equal(a, b)


def test_limiter_giu_true_peak_duoi_1_dbtp() -> None:
    x = np.zeros(SR, dtype=np.float32)
    x[1000] = 1.0  # xung đỉnh 0 dBFS, loudness thấp
    x[2000:2400] = 0.05
    y, info = normalize_loudness(x, SR, target_lufs=-16.0)
    assert true_peak_dbtp(y, SR) <= -1.0 + 1e-6
    assert info["tp_out"] <= -1.0 + 1e-6
    assert np.max(np.abs(y)) <= 1.0

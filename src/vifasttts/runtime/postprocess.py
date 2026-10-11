"""Hậu xử lý waveform thuần numpy (fade, chuẩn hoá đỉnh).

Dùng chung cho cả đường PyTorch (`infer.py`) và runtime ONNX (`runtime/`).
"""

from __future__ import annotations

import numpy as np


def peak_normalize(wav: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """Chuẩn hoá đỉnh về `target_db` (dBFS). Im lặng → giữ nguyên."""
    array = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak <= 1e-8:
        return array
    target = float(10.0 ** (target_db / 20.0))
    return (array * (target / peak)).astype(np.float32)


def fade(wav: np.ndarray, sample_rate: int, fade_in_ms: float = 10.0, fade_out_ms: float = 20.0) -> np.ndarray:
    """Fade tuyến tính đầu/cuối để tránh tiếng "cụp" khi phát."""
    array = np.asarray(wav, dtype=np.float32).copy()
    length = array.size
    if length == 0:
        return array
    fade_in = min(length, max(0, int(sample_rate * fade_in_ms / 1000.0)))
    fade_out = min(length, max(0, int(sample_rate * fade_out_ms / 1000.0)))
    if fade_in > 1:
        array[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 1:
        array[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return array


def postprocess(wav: np.ndarray, sample_rate: int, *, target_db: float | None = None) -> np.ndarray:
    """Fade mặc định; chuẩn hoá đỉnh nếu `target_db` được đặt."""
    output = fade(wav, sample_rate)
    if target_db is not None:
        output = peak_normalize(output, target_db)
    return output

"""iSTFT overlap-add thuần numpy, khớp `torch.istft(center=True, hann)`.

Dùng cho runtime ONNX (không cần PyTorch): vocoder trả về phổ (real/imag),
bước cuối ghép frame về waveform cần thực hiện ngoài graph.
"""

from __future__ import annotations

import numpy as np


def hann_window(length: int) -> np.ndarray:
    """Cửa sổ Hann đối xứng như torch.hann_window (dùng cho iSTFT numpy)."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)).astype(np.float64)


def istft(
    real: np.ndarray,
    imag: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
    center: bool = True,
) -> np.ndarray:
    """Ghép phổ [bins, T] về waveform 1 chiều.

    Quy ước khớp `torch.istft`: cửa sổ hann, chuẩn hoá bằng tổng bình phương
    cửa sổ chồng lấn, và khi `center=True` cắt `n_fft//2` mẫu mỗi đầu; độ dài
    trả về là `(T-1) * hop_length`.
    """
    spectrum = np.asarray(real) + 1j * np.asarray(imag)
    if spectrum.ndim != 2:
        raise ValueError("istft nhận phổ 2 chiều [bins, frames]")
    win_length = win_length or n_fft
    frames = np.fft.irfft(spectrum, n=n_fft, axis=0)  # [n_fft, T]

    window = hann_window(win_length)
    padded = np.zeros(n_fft, dtype=np.float64)
    offset = (n_fft - win_length) // 2
    padded[offset : offset + win_length] = window

    total = (spectrum.shape[1] - 1) * hop_length + n_fft
    output = np.zeros(total, dtype=np.float64)
    envelope = np.zeros(total, dtype=np.float64)
    window_sq = padded**2
    for index in range(spectrum.shape[1]):
        start = index * hop_length
        output[start : start + n_fft] += frames[:, index] * padded
        envelope[start : start + n_fft] += window_sq
    output = output / np.maximum(envelope, 1e-11)
    if center:
        trim = n_fft // 2
        output = output[trim : total - trim]
    return output.astype(np.float32)

"""Đặc trưng phonation (HNR, spectral tilt, glottalization) cấp frame.

Bổ sung cho f0/vuv/energy khi cần điều kiện hoá chất giọng (breathy/creaky,
kiểu thanh hỏi–ngã) — xem docs/TTS_ARCHITECTURE_RESEARCH.md §9.7. Tất cả
hàm chạy thuần torch, cùng độ dài frame với mel (`center=True`:

    frames = 1 + len(wav) // hop_length)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vifasttts.data.features import load_wav


def _frame_count(num_samples: int, hop_length: int) -> int:
    return 1 + num_samples // hop_length


def estimate_hnr(
    wav: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    frame_length: int = 1024,
    *,
    f_min: float = 60.0,
    f_max: float = 400.0,
    center: bool = True,
) -> torch.Tensor:
    """Harmonics-to-Noise Ratio thô theo autocorrelation, đơn vị dB, [T].

    `center=True` đệm nửa cửa sổ hai đầu để số frame khớp STFT `center=True`
    (`1 + len(wav)//hop`).
    """
    if center and frame_length > 1:
        pad = frame_length // 2
        wav = torch.nn.functional.pad(wav, (pad, pad))
    frames = wav.unfold(0, frame_length, hop_length)
    if frames.numel() == 0:
        return torch.zeros(0)
    frames = frames - frames.mean(dim=1, keepdim=True)
    size = 1
    while size < 2 * frame_length:
        size <<= 1
    spectrum = torch.fft.rfft(frames, n=size, dim=1)
    autocorr = torch.fft.irfft(spectrum * spectrum.conj(), n=size, dim=1)[:, :frame_length]
    energy = autocorr[:, 0].clamp_min(1e-8)
    min_lag, max_lag = max(1, int(sample_rate / f_max)), int(sample_rate / f_min)
    lags = torch.arange(min_lag, max_lag + 1)
    periodicity = (autocorr[:, lags] / energy.unsqueeze(1)).clamp(min=1e-4, max=1 - 1e-4).max(dim=1).values
    return (10.0 * torch.log10(periodicity / (1.0 - periodicity))).clamp(min=-20.0, max=40.0)


def spectral_tilt(
    wav: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    n_fft: int = 1024,
    *,
    f_min: float = 80.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """Độ dốc phổ log-magnitude theo octave (dB/oct), [T]; âm thường âm."""
    if f_max is None or f_max <= 0:
        f_max = sample_rate / 2
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                          window=window, return_complex=True, center=True)
    magnitude = spectrum.abs()  # [bins, T]
    freqs = torch.arange(magnitude.size(0), dtype=torch.float32) * (sample_rate / n_fft)
    mask = (freqs >= f_min) & (freqs <= f_max)
    x = torch.log2(freqs[mask].clamp_min(1.0))
    y = 20.0 * torch.log10(magnitude[mask].clamp_min(1e-8))
    x_centered = x - x.mean()
    slope = (x_centered.unsqueeze(1) * (y - y.mean(dim=0, keepdim=True))).sum(dim=0) / (x_centered ** 2).sum().clamp_min(1e-6)
    return slope


def glottalization_ratio(
    wav: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    n_fft: int = 1024,
    *,
    cutoff: float = 80.0,
) -> torch.Tensor:
    """Tỉ lệ năng lượng dải trầm (< cutoff Hz) trên tổng — proxy cho creak."""
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                          window=window, return_complex=True, center=True)
    power = spectrum.abs() ** 2
    freqs = torch.arange(power.size(0), dtype=torch.float32) * (sample_rate / n_fft)
    low = power[(freqs <= cutoff)].sum(dim=0)
    total = power.sum(dim=0).clamp_min(1e-8)
    return low / total


def extract_phonation_frames(
    wav: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Ba kênh phonation cấp frame cho một waveform mono."""
    sample_rate = int(config.get("sample_rate", 24000))
    mel_config = config.get("mel", {})
    hop_length = int(mel_config.get("hop_length", 256))
    n_fft = int(mel_config.get("n_fft", 1024))
    return {
        "hnr": estimate_hnr(wav, sample_rate, hop_length, frame_length=max(n_fft, 512)),
        "tilt": spectral_tilt(wav, sample_rate, hop_length, n_fft=n_fft),
        "glottalization": glottalization_ratio(wav, sample_rate, hop_length, n_fft=n_fft),
    }


def extract_phonation(wav_path: str | Path, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Đọc WAV rồi trích phonation; độ dài = `1 + len(wav)//hop` như mel."""
    wav = load_wav(wav_path, int(config.get("sample_rate", 24000)))
    return extract_phonation_frames(wav, config)


def phonation_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("phonation", {}).get("enabled", False))

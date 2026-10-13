"""Đặc trưng phonation cấp frame: `hnr` · `tilt` · `glottalization` · `jitter` · `aper` (D4C).

Bổ sung cho f0/vuv/energy khi cần điều kiện hoá chất giọng (breathy/creaky, kiểu thanh
hỏi–ngã) — xem docs/TTS_ARCHITECTURE_RESEARCH.md §9.7. Chọn kênh theo R6
(docs/ACOUSTIC_V2.md §20): AUC ngã/nặng-vs-ngang — `hnr` 0,869 · `jitter` 0,864 ·
`glottalization` 0,763 · `aper` 0,576 · `tilt` 0,521 → **cache v3 chỉ dùng 3 kênh đạt cổng**
(`hnr`, `glottalization`, `jitter`); `tilt`/`aper` giữ trong code làm chẩn đoán.
Tất cả hàm chạy torch/numpy, cùng độ dài frame với mel (`center=True`:

    frames = 1 + len(wav) // hop_length)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vifasttts.data.features import load_wav


def _frame_count(num_samples: int, hop_length: int) -> int:
    """Số frame của STFT center=True: `1 + len//hop`."""
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


def estimate_cycle_jitter(wav_f0: torch.Tensor, vuv: torch.Tensor) -> torch.Tensor:
    """Độ bất ổn chu kỳ: |Δ log2 f0| giữa 2 frame kề nhau đều hữu thanh, [T] (R6 — §20).

    `jitter[t] = |log2 f0[t+1] − log2 f0[t]|` khi `vuv[t] & vuv[t+1]`, ngược lại 0.
    Dùng cho creak (ngã/nặng): AUC 0,864 trên tập probe R6 (mạnh nhất cùng `hnr`).
    """
    f0 = wav_f0.detach().to(torch.float64)
    voiced = vuv.detach().to(torch.float64) > 0.5
    output = torch.zeros(f0.size(0), dtype=torch.float32)
    if f0.size(0) < 2:
        return output
    pair = voiced[:-1] & voiced[1:]
    delta = (torch.log2(f0[1:].clamp_min(1e-6)) - torch.log2(f0[:-1].clamp_min(1e-6))).abs()
    output[:-1] = torch.where(pair, delta, torch.zeros_like(delta)).float()
    return output


def estimate_aperiodicity(
    wav: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    f0: torch.Tensor,
    vuv: torch.Tensor,
    *,
    low_hz: float = 1000.0,
    high_hz: float = 4000.0,
) -> torch.Tensor:
    """Aperiodicity D4C (pyworld `cheaptrick`) trung bình dải `[low, high]`, [T] (R6 — §20).

    Dùng lại `f0`/`vuv` đã có (không harvest lần hai): f0 thô = f0·(vuv>0,5) — đúng vì
    `estimate_f0_pyworld` chỉ nội suy chỗ vô thanh. Dải mặc định **1–4 kHz**: dưới 1 kHz
    cheaptrick thoái hoá với tín hiệu ít hài (sine 150 Hz thuần: 0–1 kHz = 1,12 vs 1–4 kHz = 0,00)
    — xem test `test_phonation_r6.py`. AUC 0,704 (R6); lưu cache v3 để dùng thêm như cổng
    `f0_valid` khả dĩ.
    """
    import numpy as np
    import pyworld

    samples = np.ascontiguousarray(wav.detach().to(torch.float64).numpy())
    raw = (f0.detach().to(torch.float64).numpy() * (vuv.detach().numpy() > 0.5)).astype(np.float64)
    times = np.arange(raw.size, dtype=np.float64) * (hop_length / sample_rate)
    aperiodicity = pyworld.cheaptrick(samples, raw, times, sample_rate)
    bins = aperiodicity.shape[1]
    freqs = np.arange(bins, dtype=np.float64) * (sample_rate / 2.0) / (bins - 1)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    return torch.from_numpy(aperiodicity[:, band].mean(axis=1)).float()


def extract_phonation_frames(
    wav: torch.Tensor,
    config: dict[str, Any],
    *,
    f0: torch.Tensor | None = None,
    vuv: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Các kênh phonation cấp frame cho một waveform mono.

    Mặc định (config `phonation.channels`): `hnr`, `tilt`, `glottalization`; thêm `jitter`
    và `aper` khi được yêu cầu (cần `f0`/`vuv` — `extract_features` truyền sẵn, không harvest lại).
    """
    sample_rate = int(config.get("sample_rate", 24000))
    mel_config = config.get("mel", {})
    hop_length = int(mel_config.get("hop_length", 256))
    n_fft = int(mel_config.get("n_fft", 1024))
    channels = tuple(config.get("phonation", {}).get("channels")
                     or ("hnr", "tilt", "glottalization"))
    output: dict[str, torch.Tensor] = {}
    if "hnr" in channels:
        output["hnr"] = estimate_hnr(wav, sample_rate, hop_length, frame_length=max(n_fft, 512))
    if "tilt" in channels:
        output["tilt"] = spectral_tilt(wav, sample_rate, hop_length, n_fft=n_fft)
    if "glottalization" in channels:
        output["glottalization"] = glottalization_ratio(wav, sample_rate, hop_length, n_fft=n_fft)
    if "jitter" in channels:
        if f0 is None or vuv is None:
            raise ValueError("kênh `jitter` cần f0/vuv truyền vào (xem extract_features)")
        output["jitter"] = estimate_cycle_jitter(f0, vuv)
    if "aper" in channels:
        if f0 is None or vuv is None:
            raise ValueError("kênh `aper` cần f0/vuv truyền vào (xem extract_features)")
        output["aper"] = estimate_aperiodicity(wav, sample_rate, hop_length, f0, vuv)
    return output


def extract_phonation(wav_path: str | Path, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Đọc WAV rồi trích phonation; độ dài = `1 + len(wav)//hop` như mel.

    Kênh mặc định (học từ R6 — docs/ACOUSTIC_V2.md §20): `hnr`, `tilt`, `glottalization`;
    cấu hình `phonation.channels` có thể thêm `jitter`, `aper` (tự tính f0/vuv ở đây).
    """
    from vifasttts.data.features import estimate_f0_pyworld

    sample_rate = int(config.get("sample_rate", 24000))
    hop_length = int(config.get("mel", {}).get("hop_length", 256))
    f0_config = dict(config.get("f0", {}) or {})
    wav = load_wav(wav_path, sample_rate)
    f0, vuv = estimate_f0_pyworld(wav, sample_rate, hop_length,
                                  f_min=float(f0_config.get("f_min", 70.0)),
                                  f_max=float(f0_config.get("f_max", 400.0)),
                                  interpolate=bool(f0_config.get("interpolate", True)))
    return extract_phonation_frames(wav, config, f0=f0, vuv=vuv)


def phonation_enabled(config: dict[str, Any]) -> bool:
    """Cờ bật trích kênh phonation trong config (`phonation.enabled`)."""
    return bool(config.get("phonation", {}).get("enabled", False))

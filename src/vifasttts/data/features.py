"""Trích feature âm thanh: mel, năng lượng, F0/VUV (+ phonation tùy chọn).

Mọi hàm ở đây chạy thuần torch/numpy trên CPU; mel dùng STFT center=True nên số
frame = `1 + len(wav)//hop_length`. Xem `docs/MODEL_FLOW.md` §4.1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


def _hz_to_mel(freq: float) -> float:
    """Đổi Hz sang thang mel (công thức HTK)."""
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Đổi mel về Hz (nghịch đảo `_hz_to_mel`)."""
    return float(700.0 * (10.0 ** (mel / 2595.0) - 1.0))


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, f_min: float = 0.0, f_max: float | None = None) -> torch.Tensor:
    """Bộ lọc mel kiểu HTK/Slaney, trả về [n_mels, n_fft // 2 + 1]."""
    if f_max is None or f_max <= 0:
        f_max = sample_rate / 2
    bins = n_fft // 2 + 1
    mel_points = torch.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_points = torch.tensor([_mel_to_hz(float(m)) for m in mel_points])
    frequencies = torch.arange(bins, dtype=torch.float) * (sample_rate / n_fft)
    bank = torch.zeros(n_mels, bins)
    for m in range(n_mels):
        lower, center, upper = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        left = (frequencies - lower) / (center - lower).clamp_min(1e-6)
        right = (upper - frequencies) / (upper - center).clamp_min(1e-6)
        bank[m] = torch.clamp(torch.minimum(left, right), min=0.0) * (2.0 / (upper - lower).clamp_min(1e-6))
    return bank


def load_wav(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Đọc WAV mono. Ưu tiên soundfile nếu có, fallback stdlib wave (PCM)."""
    try:
        import soundfile as sf  # type: ignore
        data, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
        wav = torch.from_numpy(data if data.ndim == 1 else data.mean(axis=1))
    except ImportError:
        import wave
        with wave.open(str(path), "rb") as fh:
            if fh.getsampwidth() == 2:
                import numpy as np
                raw = np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype("float32") / 32768.0
            else:
                raise ValueError("Chỉ hỗ trợ WAV PCM 16-bit khi không có soundfile")
            wav = torch.from_numpy(raw)
            source_rate = fh.getframerate()
            if fh.getnchannels() > 1:
                wav = wav.view(-1, fh.getnchannels()).mean(dim=1)
    if source_rate != sample_rate:
        target = max(1, round(wav.numel() * sample_rate / source_rate))
        wav = torch.nn.functional.interpolate(wav.view(1, 1, -1), size=target, mode="linear", align_corners=False).view(-1)
    return wav


def extract_mel_energy(wav: torch.Tensor, mel_config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """WAV mono -> mel log [T, M] và log-energy [T]."""
    mel, energy = extract_mel_energy_batched(wav.unsqueeze(0), mel_config)
    return mel[0], energy[0]


def extract_mel_energy_batched(wav: torch.Tensor, mel_config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """WAV [B, L] -> mel log [B, T, M] và log-energy [B, T]."""
    n_fft = int(mel_config["n_fft"])
    hop = int(mel_config["hop_length"])
    win = int(mel_config.get("win_length", n_fft))
    sample_rate = int(mel_config.get("sample_rate", 24000))
    window = torch.hann_window(win)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True, center=True)
    power = spec.abs() ** 2  # [B, bins, T]
    bank = mel_filterbank(sample_rate, n_fft, int(mel_config["n_mels"]), float(mel_config.get("f_min", 0)), float(mel_config.get("f_max", 0)))
    mel = torch.log(torch.einsum("mc,bct->btm", bank, power) + 1e-6)  # [B, T, M]
    energy = torch.log(power.mean(dim=1) + 1e-6)  # [B, T]
    return mel, energy


def estimate_f0(wav: torch.Tensor, sample_rate: int, hop_length: int, frame_length: int = 1024,
                f_min: float = 70.0, f_max: float = 400.0, voicing_threshold: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    """Ước lượng F0 bằng autocorrelation đơn giản (đủ cho pipeline tham chiếu)."""
    frames = wav.unfold(0, frame_length, hop_length)
    if frames.numel() == 0:
        return torch.zeros(0), torch.zeros(0)
    frames = frames - frames.mean(dim=1, keepdim=True)
    n = 1
    while n < 2 * frame_length:
        n <<= 1
    spectrum = torch.fft.rfft(frames, n=n, dim=1)
    autocorr = torch.fft.irfft(spectrum * spectrum.conj(), n=n, dim=1)[:, :frame_length]
    energy = autocorr[:, 0].clamp_min(1e-8)
    min_lag, max_lag = int(sample_rate / f_max), int(sample_rate / f_min)
    lags = torch.arange(min_lag, max_lag + 1)
    normalized = autocorr[:, lags] / energy.unsqueeze(1)
    best = normalized.argmax(dim=1)
    peak = normalized.gather(1, best.unsqueeze(1)).squeeze(1)
    voiced = peak > voicing_threshold
    f0 = sample_rate / lags[best].float()
    f0 = torch.where(voiced, f0, torch.zeros_like(f0))
    return f0, voiced.float()


def extract_features(wav_path: str | Path, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """WAV -> mel, energy, f0, vuv; độ dài các chuỗi frame được đồng bộ.

    Khi `config["phonation"]["enabled"]`, thêm `hnr`, `tilt`, `glottalization`
    (xem `data/phonation.py`).
    """
    sample_rate = int(config.get("sample_rate", 24000))
    mel_config = dict(config["mel"])
    mel_config.setdefault("sample_rate", sample_rate)
    wav = load_wav(wav_path, sample_rate)
    mel, energy = extract_mel_energy(wav, mel_config)
    f0, vuv = estimate_f0(wav, sample_rate, int(mel_config["hop_length"]))
    frames = min(mel.size(0), energy.size(0), f0.size(0))
    features = {
        "mel": mel[:frames],
        "energy": energy[:frames],
        "f0": f0[:frames],
        "vuv": vuv[:frames],
    }
    if config.get("phonation", {}).get("enabled", False):
        from vifasttts.data.phonation import extract_phonation_frames

        for name, values in extract_phonation_frames(wav, config).items():
            features[name] = values[:frames]
    return features

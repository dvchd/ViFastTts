"""Test kênh phonation v2 (R6): jitter chu kỳ · D4C aperiodicity · tách khoá cache."""

from __future__ import annotations

import torch

from vifasttts.data.cache import feature_cache_key
from vifasttts.data.phonation import (
    estimate_aperiodicity,
    estimate_cycle_jitter,
    extract_phonation_frames,
)


def _sine(frames: int = 9600, sample_rate: int = 24000, frequency: float = 150.0) -> torch.Tensor:
    time = torch.arange(frames, dtype=torch.float32) / sample_rate
    return 0.3 * torch.sin(2.0 * torch.pi * frequency * time)


def _config(channels: list[str]) -> dict:
    return {
        "sample_rate": 24000,
        "mel": {"hop_length": 256, "n_fft": 1024},
        "f0": {"f_min": 70, "f_max": 400},
        "phonation": {"enabled": True, "channels": channels},
    }


def test_cycle_jitter_bien_doi_gap_doi():
    f0 = torch.tensor([100.0, 200.0, 200.0, 200.0])
    vuv = torch.ones(4)
    jitter = estimate_cycle_jitter(f0, vuv)
    assert jitter.shape == (4,)
    assert abs(float(jitter[0]) - 1.0) < 1e-6  # 100 → 200 Hz = +1 octave
    assert abs(float(jitter[1])) < 1e-6
    assert float(jitter[3]) == 0.0  # frame cuối không có cặp


def test_cycle_jitter_mask_vuv():
    f0 = torch.tensor([150.0, 300.0, 150.0])
    vuv = torch.tensor([1.0, 0.0, 1.0])
    jitter = estimate_cycle_jitter(f0, vuv)
    assert float(jitter[0]) == 0.0 and float(jitter[1]) == 0.0


def test_aperiodicity_sine_thap_hon_noise():
    sample_rate, hop = 24000, 256
    sine = _sine(9600, sample_rate)
    f0 = torch.full((1 + sine.numel() // hop,), 150.0)
    vuv = torch.ones_like(f0)
    # dải 1–4 kHz: tín hiệu tuần hoàn → ~0; nhiễu trắng → cao (band 0–1 kHz thoái hoá, xem docstring)
    aper_sine = estimate_aperiodicity(sine, sample_rate, hop, f0, vuv, low_hz=1000.0, high_hz=4000.0)
    noise = 0.3 * torch.randn(sine.numel(), generator=torch.Generator().manual_seed(0))
    aper_noise = estimate_aperiodicity(noise, sample_rate, hop, f0, vuv, low_hz=1000.0, high_hz=4000.0)
    assert aper_sine.shape == f0.shape
    assert float(aper_sine.mean()) < 0.05 < float(aper_noise.mean())


def test_extract_phonation_frames_channels_va_do_dai():
    sample_rate, hop = 24000, 256
    sine = _sine(9600, sample_rate)
    f0 = torch.full((1 + sine.numel() // hop,), 150.0)
    vuv = torch.ones_like(f0)
    output = extract_phonation_frames(sine, _config(["hnr", "glottalization", "jitter", "aper"]),
                                      f0=f0, vuv=vuv)
    assert set(output) == {"hnr", "glottalization", "jitter", "aper"}
    for value in output.values():
        assert value.numel() == f0.numel()


def test_extract_phonation_frames_jitter_can_f0():
    config = _config(["jitter"])
    try:
        extract_phonation_frames(_sine(), config)
        raise AssertionError("phải raise vì thiếu f0/vuv")
    except ValueError as error:
        assert "f0/vuv" in str(error)


def test_cache_key_tach_theo_phonation():
    row = {"audio_path": "/khong/ton/tai.wav"}
    base = {"sample_rate": 24000, "mel": {"hop_length": 256}, "f0": {"method": "pyworld"}}
    off = {**base, "phonation": {"enabled": False}}
    on = {**base, "phonation": {"enabled": True, "channels": ["hnr", "aper"]}}
    assert feature_cache_key(row, off) != feature_cache_key(row, on)
    assert feature_cache_key(row, off) == feature_cache_key(row, {**base, "phonation": {"enabled": False}})

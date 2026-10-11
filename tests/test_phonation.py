"""Test đặc trưng phonation (HNR, tilt, glottalization)."""
from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vifasttts.data.features import extract_features
from vifasttts.data.phonation import (
    estimate_hnr,
    extract_phonation,
    extract_phonation_frames,
    glottalization_ratio,
    phonation_enabled,
    spectral_tilt,
)

SAMPLE_RATE = 24000


def _sine(freq: float = 200.0, seconds: float = 0.4) -> torch.Tensor:
    time_axis = torch.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return 0.3 * torch.sin(2 * math.pi * freq * time_axis)


def test_hnr_high_for_periodic_low_for_noise():
    sine = _sine(200.0)
    noise = 0.1 * torch.randn_like(sine)
    hnr_sine = estimate_hnr(sine, SAMPLE_RATE, 256).mean()
    hnr_noise = estimate_hnr(noise, SAMPLE_RATE, 256).mean()
    assert hnr_sine > 6.0
    assert hnr_noise < 0.0
    assert hnr_sine > hnr_noise


def test_spectral_tilt_negative_for_voiced():
    tilt = spectral_tilt(_sine(150.0), SAMPLE_RATE, 256, n_fft=512).mean()
    assert tilt < 0.0


def test_glottalization_ratio_in_range():
    ratios = glottalization_ratio(_sine(60.0), SAMPLE_RATE, 256, n_fft=1024)
    assert ratios.numel() > 0
    assert ((ratios >= 0.0) & (ratios <= 1.0)).all()
    # Tín hiệu 60 Hz nằm sát ngưỡng trầm: tỉ lệ năng lượng trầm phải dương.
    assert ratios.mean() > 0.0


def test_frames_align_with_mel_center():
    wav = _sine(seconds=0.3)
    config = {"sample_rate": SAMPLE_RATE, "mel": {"hop_length": 256, "n_fft": 512}}
    phonation = extract_phonation_frames(wav, config)
    expected = 1 + wav.numel() // 256
    for name, values in phonation.items():
        assert values.numel() == expected, name


def test_phonation_enabled_flag():
    assert not phonation_enabled({})
    assert phonation_enabled({"phonation": {"enabled": True}})


def test_extract_features_optional_phonation(tmp_path: Path):
    path = tmp_path / "tone.wav"
    wav = _sine(220.0)
    pcm = (wav.clamp(-1, 1) * 32767).to(torch.int16)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes(pcm.numpy().tobytes())

    base_config = {
        "sample_rate": SAMPLE_RATE,
        "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20, "f_min": 0, "f_max": 12000},
    }
    plain = extract_features(path, base_config)
    assert "hnr" not in plain

    enriched = extract_features(path, {**base_config, "phonation": {"enabled": True}})
    for name in ("hnr", "tilt", "glottalization"):
        assert name in enriched
        assert enriched[name].numel() == enriched["mel"].size(0)

    direct = extract_phonation(path, base_config)
    assert direct["hnr"].numel() > 0


def test_dataset_and_collate_include_phonation(tmp_path: Path):
    import csv

    from vifasttts.data.dataset import TtsDataset, collate_batch
    from vifasttts.data.manifest import build_manifest

    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    pcm = (_sine(220.0).clamp(-1, 1) * 32767).to(torch.int16)
    with wave.open(str(wav_dir / "s1.wav"), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SAMPLE_RATE)
        fh.writeframes(pcm.numpy().tobytes())
    metadata = tmp_path / "meta.csv"
    with metadata.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["s1", "Chào bạn"])
    manifest = tmp_path / "m.jsonl"
    build_manifest(str(metadata), str(wav_dir), str(manifest))

    config = {
        "sample_rate": SAMPLE_RATE,
        "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20, "f_min": 0, "f_max": 12000},
        "phonation": {"enabled": True},
    }
    item = TtsDataset(str(manifest), config)[0]
    assert item["phonation"].shape == (item["features"].size(0), 3)
    batch = collate_batch([item, item])
    assert batch["phonation"].shape[2] == 3

    plain = TtsDataset(str(manifest), {**config, "phonation": {"enabled": False}})[0]
    assert "phonation" not in plain

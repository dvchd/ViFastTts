"""Test cache feature/duration."""
from __future__ import annotations

import csv
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vifasttts.data.cache import FeatureCache, feature_cache_key, precompute_features
from vifasttts.data.dataset import TtsDataset
from vifasttts.data.manifest import build_manifest

CONFIG = {
    "sample_rate": 24000,
    "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20, "f_min": 0, "f_max": 12000},
}


def _write_wav(path: Path, freq: float = 220.0, seconds: float = 0.3) -> None:
    import numpy as np

    sample_rate = 24000
    time_axis = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * freq * time_axis)
    pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def _manifest(tmp_path: Path, texts: list[tuple[str, str]]) -> Path:
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir(exist_ok=True)
    for sample_id, _ in texts:
        _write_wav(wav_dir / f"{sample_id}.wav")
    metadata = tmp_path / "meta.csv"
    with metadata.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        for row in texts:
            writer.writerow(row)
    manifest = tmp_path / "manifest.jsonl"
    build_manifest(str(metadata), str(wav_dir), str(manifest))
    return manifest


def test_cache_key_stable_and_config_sensitive(tmp_path: Path):
    _manifest(tmp_path, [("s1", "ba")])
    row = {"audio_path": str(tmp_path / "wavs" / "s1.wav")}
    assert feature_cache_key(row, CONFIG) == feature_cache_key(row, CONFIG)
    changed = {**CONFIG, "mel": {**CONFIG["mel"], "n_mels": 21}}
    assert feature_cache_key(row, CONFIG) != feature_cache_key(row, changed)
    with_phonation = {**CONFIG, "phonation": {"enabled": True}}
    assert feature_cache_key(row, CONFIG) != feature_cache_key(row, with_phonation)


def test_dataset_uses_cache_without_recompute(tmp_path: Path, monkeypatch):
    manifest = _manifest(tmp_path, [("s1", "Chào bạn")])
    cache_dir = tmp_path / "cache"

    import vifasttts.data.dataset as dataset_module

    calls = {"count": 0}
    real_extract = dataset_module.extract_features

    def counting_extract(*args, **kwargs):
        calls["count"] += 1
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(dataset_module, "extract_features", counting_extract)
    dataset = TtsDataset(str(manifest), CONFIG, cache_dir=cache_dir)
    dataset[0]
    assert calls["count"] == 1
    dataset[0]
    assert calls["count"] == 1  # lần hai đọc cache
    assert list(cache_dir.glob("*.pt"))


def test_precompute_features(tmp_path: Path):
    manifest = _manifest(tmp_path, [("s1", "ba"), ("s2", "mẹ")])
    cache_dir = tmp_path / "cache"
    done = precompute_features(manifest, CONFIG, cache_dir)
    assert done == 2
    assert len(list(cache_dir.glob("*.pt"))) == 2
    # Chạy lại chỉ đếm, không tạo thêm file (đã có cache).
    assert precompute_features(manifest, CONFIG, cache_dir) == 2
    assert len(list(cache_dir.glob("*.pt"))) == 2


def test_feature_cache_get_put_roundtrip(tmp_path: Path):
    cache = FeatureCache(tmp_path / "c", CONFIG)
    row = {"audio_path": str(tmp_path / "missing.wav")}
    assert cache.get(row) is None
    cache.put(row, {"mel": torch.zeros(3, 20)})
    loaded = cache.get(row)
    assert loaded is not None and loaded["mel"].shape == (3, 20)

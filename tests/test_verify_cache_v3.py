"""Test tool kiểm tra cache v3 (tools/verify_cache_v3.py)."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tools.verify_cache_v3 import REQUIRED, verify_entry, verify_manifest
from vifasttts.data.cache import feature_cache_key

CONFIG = {
    "sample_rate": 24000,
    "mel": {"hop_length": 256, "n_fft": 1024},
    "f0": {"method": "pyworld"},
    "phonation": {"enabled": True, "channels": ["hnr", "glottalization", "jitter"]},
}


def _valid(n: int = 8) -> dict:
    return {
        "mel": torch.zeros(n, 10),
        "energy": torch.zeros(n),
        "f0": torch.full((n,), 120.0),
        "vuv": torch.ones(n),
        "hnr": torch.zeros(n),
        "glottalization": torch.full((n,), 0.5),
        "jitter": torch.zeros(n),
    }


def test_entry_hop_le_khong_loi():
    assert verify_entry(_valid()) == []


def test_entry_thieu_khoa_neu_ten_khoa():
    errors = verify_entry({k: v for k, v in _valid().items() if k != "hnr"}, REQUIRED)
    assert any("hnr" in e for e in errors)


def test_entry_lech_do_dai_bao_loi():
    bad = _valid()
    bad["f0"] = torch.zeros(5)
    assert any("f0" in e for e in verify_entry(bad, REQUIRED))


def test_entry_nan_va_jitter_am_bao_loi():
    bad = _valid()
    bad["energy"][0] = float("nan")
    bad["jitter"][1] = -0.5
    errors = verify_entry(bad, REQUIRED)
    assert any("nan" in e and "energy" in e for e in errors)
    assert any("jitter" in e for e in errors)


def test_entry_glottalization_ngoai_mien_bao_loi():
    bad = _valid()
    bad["glottalization"][0] = 1.5
    assert any("glottalization" in e for e in verify_entry(bad, REQUIRED))


def test_manifest_hit_roi_missing(tmp_path: Path):
    row = {"audio_path": "/khong/ton/tai.wav", "id": "x"}
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    torch.save(_valid(), cache_dir / f"{feature_cache_key(row, CONFIG)}.pt")
    got = verify_manifest([row], CONFIG, cache_dir, sample=0, seed=42)
    assert got["hits"] == 1 and got["missing"] == 0 and not got["issue_counts"]
    (cache_dir / f"{feature_cache_key(row, CONFIG)}.pt").unlink()
    got = verify_manifest([row], CONFIG, cache_dir, sample=0, seed=42)
    assert got["hits"] == 0 and got["missing"] == 1

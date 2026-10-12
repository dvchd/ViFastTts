"""Test đóng băng biểu diễn mel: hash cấu hình + stats, lưu và kiểm trong checkpoint.

Xem `docs/REVISION_PLAN_V2.md` rev.5 §1a.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vifasttts.data.features import canonical_mel_config, mel_config_hash, verify_mel_hash

MEL = {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20,
       "f_min": 0, "f_max": 12000}
CONFIG = {"sample_rate": 24000, "mel": MEL}


def test_hash_ổn_định_và_nhạy_với_cấu_hình():
    assert mel_config_hash(canonical_mel_config(CONFIG)) == mel_config_hash(canonical_mel_config(CONFIG))
    for key, value in (("hop_length", 128), ("n_fft", 512), ("f_max", 8000), ("n_mels", 21)):
        changed = {**CONFIG, "mel": {**MEL, key: value}}
        assert mel_config_hash(canonical_mel_config(changed)) != mel_config_hash(canonical_mel_config(CONFIG)), key


def test_hash_đổi_khi_sửa_file_stats(tmp_path: Path):
    stats = tmp_path / "mel_stats.json"
    stats.write_text(json.dumps({"mean": [0.0] * 20, "std": [1.0] * 20}), encoding="utf-8")
    config = {**CONFIG, "mel": {**MEL, "normalize_path": str(stats)}}
    before = mel_config_hash(canonical_mel_config(config))
    stats.write_text(json.dumps({"mean": [0.5] * 20, "std": [1.0] * 20}), encoding="utf-8")
    assert mel_config_hash(canonical_mel_config(config)) != before


def test_verify_mel_hash_báo_lỗi_khi_lệch():
    mel_config = canonical_mel_config(CONFIG)
    good = {"mel_hash": mel_config_hash(mel_config)}
    verify_mel_hash(good, mel_config, "ck.pt")  # khớp → không lỗi
    with pytest.raises(ValueError, match="không khớp"):
        verify_mel_hash({"mel_hash": "sai"}, mel_config, "ck.pt")
    verify_mel_hash({}, mel_config, "checkpoint-cu.pt")  # checkpoint cũ: bỏ qua

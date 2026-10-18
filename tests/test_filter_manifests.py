"""Test tools/filter_manifests.py — lọc row theo clip bị loại (chuẩn hoá khoá)."""
from __future__ import annotations

import json

from tools.filter_manifests import clip_id, filter_manifest, normalize_drop


def _row(parent: str, path: str, tone: str = "NGANG") -> dict:
    return {
        "parent": parent,
        "audio_path": path,
        "syllables": [{"abstract": {"tone": tone, "coda": "NG"}}],
    }


def test_normalize_drop_nhan_ca_id_va_path():
    assert normalize_drop(["abc_0001", "data/x/abc_0002.wav"]) == {"abc_0001", "abc_0002"}


def test_filter_manifest_bo_dung_clip(tmp_path):
    manifest = tmp_path / "m.jsonl"
    rows = [
        _row("p1", "data/audio_v2/p1_0001.wav"),
        _row("p2", "data/audio_v2/p2_0001.wav"),
        {"audio_path": "data/audio_v2/p1_0002.wav", "syllables": [{"abstract": {"tone": "NGA", "coda": "P"}}]},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    kept, stats = filter_manifest(manifest, normalize_drop(["p1_0001", "data/audio_v2/p1_0002.wav"]))
    assert stats["rows_before"] == 3 and stats["rows_after"] == 1
    assert clip_id(kept[0]) == "p2_0001"
    assert stats["clips_dropped"] == 2
    assert stats["tones"] == {"NGANG": 1}

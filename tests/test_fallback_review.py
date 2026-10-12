"""Test gói rà soát fallback + merge overrides (`tools/fallback_review.py`)."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from tools.fallback_review import (
    build_review,
    collect_fallback_words,
    merge_overrides,
    suggest_reading,
)


def _write_wav(path: Path) -> None:
    time_axis = np.arange(2400) / 24000
    pcm = (0.2 * np.sin(2 * np.pi * 200 * time_axis) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24000)
        fh.writeframes(pcm.tobytes())


def _manifest(tmp_path: Path) -> Path:
    wav = tmp_path / "s1.wav"
    _write_wav(wav)
    rows = [
        {"id": "s1", "audio_path": str(wav), "syllables": [
            {"normalized": "dust", "status": "FALLBACK_REQUIRED"},
            {"normalized": "của", "status": "VALID"},
        ]},
        {"id": "s2", "audio_path": str(wav), "syllables": [
            {"normalized": "dust", "status": "FALLBACK_REQUIRED"},
            {"normalized": "ordering", "status": "FALLBACK_REQUIRED"},
        ]},
        {"id": "s3", "audio_path": str(wav), "syllables": [
            {"normalized": "lạ", "status": "FALLBACK_REQUIRED"},  # chỉ 1 lần → bị bỏ qua
            {"normalized": "ordering", "status": "FALLBACK_REQUIRED"},
        ]},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                        encoding="utf-8")
    return manifest


def test_suggest_reading():
    assert suggest_reading("API") == "ây pi ai"
    assert suggest_reading("ordering") == "o đờ ring"
    assert suggest_reading("chffix") is None


def test_collect_fallback_words(tmp_path: Path):
    counter, samples = collect_fallback_words([str(_manifest(tmp_path))])
    assert counter["dust"] == 2 and counter["lạ"] == 1
    assert samples["dust"] == ["s1", "s2"]


def test_build_review_and_merge(tmp_path: Path):
    manifest = _manifest(tmp_path)
    out = tmp_path / "review"
    stats = build_review([str(manifest)], out, top=10, min_count=2)
    assert stats["words"] == 2  # dust + ordering (lạ bị loại vì 1 lần)
    candidates = [json.loads(line) for line in
                  (out / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["word"] for row in candidates} == {"dust", "ordering"}
    assert any(path.name.startswith("dust__") for path in (out / "clips").glob("*.wav"))
    filled = (out / "filled.txt").read_text(encoding="utf-8")
    assert "dust|" in filled and "ordering|o đờ ring" in filled

    # Merge: điền "đớt" cho dust, bỏ trống ordering, thêm dòng lỗi.
    fill = tmp_path / "filled_edit.txt"
    fill.write_text("dust|đớt\nordering|\nqqq|xyz\n", encoding="utf-8")
    overrides = tmp_path / "overrides.json"
    overrides.write_text('{"cũ": "cũ"}', encoding="utf-8")
    result = merge_overrides(fill, overrides)
    assert result["added"] == 1
    payload = json.loads(overrides.read_text(encoding="utf-8"))
    assert payload["dust"] == "đớt" and payload["cũ"] == "cũ"
    assert result["invalid"] and result["invalid"][0][0] == "qqq"

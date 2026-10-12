"""Test parse lại text sau overrides (`tools/reparse_overrides.py`)."""
from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np

from tools.reparse_overrides import reparse_dataset


def _wav(path: Path) -> None:
    time_axis = np.arange(2400) / 24000
    pcm = (0.2 * np.sin(2 * np.pi * 180 * time_axis) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24000)
        fh.writeframes(pcm.tobytes())


def test_reparse_áp_overrides_và_rebuild_manifest(tmp_path: Path):
    root = tmp_path / "ds"
    (root / "audio").mkdir(parents=True)
    _wav(root / "audio" / "v1_0001.wav")
    _wav(root / "audio" / "v2_0001.wav")
    # Text gốc có token OOV "window" chưa có cách đọc.
    with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["v1_0001", "mở window lên"])
        writer.writerow(["v2_0001", "đóng window lại"])

    from vifasttts.data.manifest import build_manifest, coverage_report

    build_manifest(str(root / "metadata.csv"), str(root / "audio"), str(root / "manifest.jsonl"))
    before = coverage_report(root / "manifest.jsonl")
    assert before["fallback"] > 0  # "window" là OOV

    stats = reparse_dataset(root, {"window": "uyn đâu"}, val_videos=1, seed=13)
    assert stats["fallback_after"] == 0
    assert stats["fallback_before"] == before["fallback"]
    assert stats["changed_texts"] == 2
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "uyn" in rows[0]["text"] and "window" not in rows[0]["text"]
    assert (root / "train.jsonl").exists() and (root / "val.jsonl").exists()
    assert stats["train"] + stats["val"] == 2
    assert (root / "metadata.pre_reparse.csv").exists()

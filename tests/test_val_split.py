"""V11.A6: val_calib / val_lock chia THEO CLIP CHA — không giao nhau, không giao val300."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data/codelung/val_clean_v2_dur_seg_slot4_fixtail.jsonl"
CALIB = ROOT / "data/codelung/val_clean_v2_dur_seg_slot4_fixtail_calib.jsonl"
LOCK = ROOT / "data/codelung/val_clean_v2_dur_seg_slot4_fixtail_lock.jsonl"

pytestmark = pytest.mark.skipif(not (VAL.exists() and CALIB.exists() and LOCK.exists()),
                                reason="manifest chia tập val là dữ liệu local (data/)")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_chia_theo_clip_cha_khong_giao_va_khong_giao_val300():
    val, calib, lock = _rows(VAL), _rows(CALIB), _rows(LOCK)
    p300 = {row["parent"] for row in val[:300]}
    parents_calib = {row["parent"] for row in calib}
    parents_lock = {row["parent"] for row in lock}
    assert not (parents_calib & parents_lock), "clip cha xuất hiện ở CẢ HAI tập"
    assert not (parents_calib & p300), "val_calib giao val300"
    assert not (parents_lock & p300), "val_lock giao val300"
    assert not ({row["id"] for row in calib} & {row["id"] for row in lock}), "hàng trùng hai tập"
    pool_ids = {row["id"] for row in val[300:1773]}
    assert {row["id"] for row in calib} <= pool_ids
    assert {row["id"] for row in lock} <= pool_ids
    ratio = len(calib) / (len(calib) + len(lock))
    assert 0.4 <= ratio <= 0.6, f"chia lệch: {ratio:.2f}"

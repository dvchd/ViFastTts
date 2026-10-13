"""Test `tools/verify_manifest_gaps.py` (A2/§43.1) — hàm thuần, không cần dữ liệu thật."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import verify_manifest_gaps as vm


def _row(row_id: str, *, durations: list[int], gaps: list[int], span: int | None = None,
         lead: int = 0, parent: str | None = None) -> dict:
    """Row manifest giả; `span=None` ⇒ không có `frame_range` (clip đầy đủ)."""
    row = {"id": row_id, "durations": durations, "gaps": gaps, "lead_frames": lead}
    if span is not None:
        row["frame_range"] = [100, 100 + span]
        row["parent"] = parent or "clip"
    return row


def test_check_rows_dat_bat_bien():
    """Manifest A2 đúng: `T == Σdur + Σgap`, `len(gaps) == N`, lead = 0 ⇒ ok."""
    rows = [_row("a", durations=[10, 10], gaps=[5, 30], span=55),
            _row("b", durations=[8], gaps=[0], span=8)]
    report = vm.check_rows(rows)
    assert report["ok"] is True
    assert all(entry["count"] == 0 for entry in report["failures"].values())
    assert report["gap_sum_per_row"] == 17.5      # (35 + 0) / 2
    assert report["labeled_positions"] == 3 and report["positive_rate"] == 0.6667
    assert report["rows_with_tail_pause"] == 1    # row "a": gap cuối 30 ≥ 3
    assert report["span"]["max"] == 55


def test_check_rows_bat_loi_tung_loai():
    """Vỡ bất biến / sai số gap / lead ≠ 0 / quá dài ⇒ ghi đúng id, `ok=False`."""
    rows = [_row("thieu", durations=[10, 10], gaps=[5], span=26),          # T ≠ Σdur+Σgap (25)
            _row("lead", durations=[10], gaps=[2], span=12, lead=7),        # row segment lead ≠ 0
            _row("dai", durations=[10], gaps=[vm.MAX_POSITIONS], span=vm.MAX_POSITIONS + 10),
            _row("la", durations=[10], gaps=[3], span=13)]                  # hợp lệ
    report = vm.check_rows(rows)
    assert report["ok"] is False
    assert report["failures"]["invariant"]["ids"] == ["thieu"]
    assert report["failures"]["lead"]["ids"] == ["lead"]
    assert report["failures"]["too_long"]["ids"] == ["dai"]
    assert report["failures"]["gaps_length"]["count"] == 0
    # row cũ (`len(gaps) == N-1`) vẫn hợp lệ về số lượng gap
    old = vm.check_rows([_row("cu", durations=[10, 10], gaps=[4], span=24)])
    assert old["ok"] is True and old["rows_with_tail_pause"] == 0


def test_parent_coverage():
    """Độ phủ khối lượng pause của segment so với clip cha (A2: 46,5% → ~100%)."""
    parents = [{"id": "clip", "durations": [10, 10], "gaps": [5, 30], "lead_frames": 0}]
    rows = [_row("clip__s00", durations=[10, 10], gaps=[5, 30], span=55, parent="clip")]
    coverage = vm.parent_coverage(rows, parents)
    assert coverage["parent_gap_mass"] == 35 and coverage["segment_gap_mass"] == 35
    assert coverage["coverage"] == 1.0 and coverage["parents_missing"] == 0
    # row cũ (bỏ gap cuối) ⇒ phủ 5/35
    old_rows = [_row("clip__s00", durations=[10, 10], gaps=[5], span=25, parent="clip")]
    assert vm.parent_coverage(old_rows, parents)["coverage"] == 0.1429
    # thiếu clip cha ⇒ đếm vào `parents_missing`, không vỡ
    missing = vm.parent_coverage([_row("x__s00", durations=[10], gaps=[1], span=11, parent="khac")],
                                 parents)
    assert missing["parents_missing"] == 1

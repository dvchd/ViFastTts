"""tools/verify_manifest_gaps.py — A2 (§43.1): kiểm **bất biến gap** của manifest segment.

Lỗi A2 (§43.1): `cut_segments.py` cũ ghi `gaps[start:end-1]` ⇒ 46,5% khối lượng pause của clip
(pause ở đúng chỗ cắt + lặng đuôi) **không có nhãn**; `v2_targets.pause_targets` còn bỏ luôn vị trí
cuối row. Manifest sinh lại phải thoả:

1. `T == Σdur + Σgap` với `T = frame_range[1] - frame_range[0]` (row clip đầy đủ: `T` = độ dài audio
   nếu truyền `--audio-check`) — nếu vỡ thì nhãn gate/log_gap trỏ ra ngoài row ("pause bịa");
2. `len(gaps) == len(durations)` (row A2) — row cũ `N-1` vẫn hợp lệ nhưng vị trí cuối không có nhãn;
3. row segment có `lead_frames == 0` (toạ độ tính từ âm tiết đầu của segment);
4. `T ≤ MAX_POSITIONS` (4096 frame ≈ 43,7 s — chia câu trước model);
5. (tuỳ chọn `--parent`) độ phủ khối lượng pause so với clip cha: `Σgap nhãn / Σgap clip`.

Chạy:
    python tools/verify_manifest_gaps.py \
        --manifest data/codelung/train_clean_v2_dur_seg_slot4_fixgap.jsonl \
        --parent data/codelung/train_clean_v2_dur.jsonl \
        --out artifacts/dataset_v2/verify_gaps_train_fixgap.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

#: Trần frame của model (`AGENTS.md`: `MAX_POSITIONS = 4096`).
MAX_POSITIONS = 4096
#: Ngưỡng "đặt pause" — khớp `data/v2_targets.GATE_MIN_FRAMES`.
GATE_MIN_FRAMES = 3


def load_rows(path: str | Path) -> list[dict]:
    """Đọc manifest JSONL (bỏ dòng trống)."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def row_span(row: dict) -> int | None:
    """`T` của row: độ dài `frame_range`, hoặc None nếu row không có (clip đầy đủ)."""
    frame_range = row.get("frame_range")
    return (int(frame_range[1]) - int(frame_range[0])) if frame_range else None


def check_rows(rows: list[dict], *, max_ids: int = 10) -> dict:
    """Kiểm bất biến trên danh sách row → dict thống kê (danh sách id lỗi bị cắt ở `max_ids`)."""
    failures: dict[str, list[str]] = {name: [] for name in
                                      ("invariant", "gaps_length", "lead", "too_long")}
    gap_mass = 0
    labeled_mass = 0
    positions = positive = tail_positive = 0
    spans: list[int] = []
    for row in rows:
        row_id = str(row.get("id", "?"))
        durations = [int(value) for value in row.get("durations") or []]
        gaps = [int(value) for value in row.get("gaps") or []]
        span = row_span(row)
        total = sum(durations) + sum(gaps)
        if span is not None and span != total:
            failures["invariant"].append(row_id)
        if len(gaps) not in (len(durations), max(len(durations) - 1, 0)):
            failures["gaps_length"].append(row_id)
        if span is not None and int(row.get("lead_frames") or 0) != 0:
            failures["lead"].append(row_id)
        if span is not None and span > MAX_POSITIONS:
            failures["too_long"].append(row_id)
        gap_mass += sum(gaps)
        labeled_mass += sum(gaps[:len(gaps)]) if len(gaps) == len(durations) else sum(gaps)
        positions += len(gaps)
        positive += sum(1 for gap in gaps if gap >= GATE_MIN_FRAMES)
        tail_positive += int(bool(gaps) and len(gaps) == len(durations) and gaps[-1] >= GATE_MIN_FRAMES)
        if span is not None:
            spans.append(span)
    report: dict = {
        "rows": len(rows),
        "failures": {name: {"count": len(ids), "ids": ids[:max_ids]} for name, ids in failures.items()},
        "gap_sum_per_row": round(statistics.mean([sum(int(v) for v in row.get("gaps") or [])
                                                  for row in rows]), 1) if rows else 0.0,
        "labeled_pause_mass": labeled_mass,
        "labeled_positions": positions,
        "positive_rate": round(positive / positions, 4) if positions else None,
        "rows_with_tail_pause": tail_positive,
        "span": {"n": len(spans), "min": min(spans) if spans else None,
                 "mean": round(statistics.mean(spans), 1) if spans else None,
                 "max": max(spans) if spans else None},
        "max_positions": MAX_POSITIONS,
    }
    report["ok"] = all(entry["count"] == 0 for name, entry in report["failures"].items()
                       if name != "too_long")
    return report


def parent_coverage(rows: list[dict], parents: list[dict]) -> dict:
    """Độ phủ khối lượng pause của các segment so với clip cha (`--parent`)."""
    by_id = {str(row.get("id")): row for row in parents}
    parent_mass = sum(sum(int(value) for value in row.get("gaps") or []) for row in parents)
    labeled = 0
    missing = 0
    for row in rows:
        if row.get("parent") is None:
            continue
        if str(row["parent"]) not in by_id:
            missing += 1
        labeled += sum(int(value) for value in row.get("gaps") or [])
    return {"parent_gap_mass": parent_mass, "segment_gap_mass": labeled,
            "coverage": round(labeled / parent_mass, 4) if parent_mass else None,
            "parents_missing": missing}


def main() -> None:
    """Kiểm một/nhiều manifest; in gọn + ghi JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--parent", default=None,
                        help="manifest clip cha (`*_dur.jsonl`) để đo độ phủ khối lượng pause")
    parser.add_argument("--out", default=None, help="ghi JSON thống kê (tuỳ chọn)")
    args = parser.parse_args()

    parents = load_rows(args.parent) if args.parent else None
    report: dict = {}
    for manifest in args.manifest:
        rows = load_rows(manifest)
        stats = check_rows(rows)
        if parents is not None:
            stats["coverage"] = parent_coverage(rows, parents)
        report[Path(manifest).stem] = stats
        failed = {name: entry["count"] for name, entry in stats["failures"].items() if entry["count"]}
        print(f"{Path(manifest).stem}: {stats['rows']} row · Σgap/row {stats['gap_sum_per_row']} · "
              f"dương tính {stats['positive_rate']} · pause đuôi {stats['rows_with_tail_pause']} · "
              f"T max {stats['span']['max']} · lỗi {failed or 'KHÔNG'} · ok={stats['ok']}")
        if "coverage" in stats:
            print(f"  phủ khối lượng pause: {stats['coverage']['coverage']} "
                  f"({stats['coverage']['segment_gap_mass']}/{stats['coverage']['parent_gap_mass']})")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Report: {out_path}")


if __name__ == "__main__":
    main()

"""Cắt clip dài thành segment 3–10s tại pause/ranh giới câu — REBUILD_PLAN_M6 §2.3.

- Input: manifest đã có `durations` (frames hữu thanh), `gaps` (frames lặng sau từng âm tiết,
  phần tử cuối = đuôi clip), `lead_frames` (lặng đầu clip) — từ `mfa_to_durations.py` — và
  `pauses` (pause ≥150 ms giữa hai từ, điểm cắt ưu tiên).
- **Cắt tại biên âm tiết** (không bao giờ cắt giữa âm tiết): ưu tiên điểm cắt là **pause**, nếu
  không có pause trong cửa sổ thì cắt tại biên âm tiết xa nhất còn trong `[min, max]` frame.
- Segment **không gồm** khoảng lặng ngoài rìa: `frame_range = [start_âm-tiết-đầu, end_âm-tiết-cuối]`
  (tự động cắt lặng đầu clip ở segment đầu và đuôi clip ở segment cuối khi rìa là pause/lặng).
- Row của segment giữ nguyên `audio_path` của clip gốc + `frame_range` [start, end) +
  `durations`/`gaps` đã cắt (gaps chỉ còn khoảng lặng **giữa** các âm tiết trong segment).
- Đuôi ngắn hơn `min` được gộp vào segment trước nếu tổng ≤ `max`.

Chạy:
    python tools/cut_segments.py --manifest data/codelung/train_clean_v2_dur.jsonl \
        --manifest data/codelung/val_clean_v2_dur.jsonl --out-dir data/codelung
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def cut_clip(spans: list[tuple[int, int]], pauses_after: set[int], *,
             min_frames: int, max_frames: int) -> list[tuple[int, int]]:
    """Trả danh sách (start, end) theo chỉ số âm tiết — greedy, ưu tiên pause trong cửa sổ.

    `spans[i] = (start_frame, end_frame)` tuyệt đối của âm tiết i (end không gồm gap sau nó).
    """
    count = len(spans)
    if count == 0:
        return []

    segments: list[tuple[int, int]] = []
    start = 0
    while start < count:
        candidates: list[int] = []
        for end in range(start + 1, count + 1):
            length = spans[end - 1][1] - spans[start][0]
            if length > max_frames and end > start + 1:
                break
            candidates.append(end)
        window = [end for end in candidates
                  if spans[end - 1][1] - spans[start][0] >= min_frames or end == count]
        if not window:  # phần còn lại ngắn hơn min → lấy hết
            window = [candidates[-1]]
        paused = [end for end in window if (end - 1) in pauses_after]
        chosen = paused[-1] if paused else window[-1]
        segments.append((start, chosen))
        start = chosen

    if len(segments) >= 2:
        merge_start, merge_end = segments[-1]
        if spans[merge_end - 1][1] - spans[merge_start][0] < min_frames:
            previous_start, _ = segments[-2]
            if spans[merge_end - 1][1] - spans[previous_start][0] <= max_frames:
                segments[-2] = (previous_start, merge_end)
                segments.pop()
    return segments


def cut_manifest(manifest: str | Path, out_path: str | Path, *,
                 min_frames: int, max_frames: int, floor_frames: int = 24) -> dict[str, Any]:
    """Cắt toàn bộ manifest thành segment; ghi manifest mới + trả thống kê.

    `floor_frames`: segment ngắn hơn ngưỡng này (mặc định 24 frame ≈ 0,26 s) bị bỏ —
    đuôi vụn sau khi cắt không đủ để thành mẫu huấn luyện.
    """
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_rows: list[dict[str, Any]] = []
    stats = {"clips": len(rows), "segments": 0, "pause_cuts": 0, "clips_split": 0,
             "min_seg": None, "max_seg": None, "dropped_tiny": 0}
    lengths: list[int] = []
    for row in rows:
        durations = row.get("durations")
        if not durations:
            continue
        gaps = list(row.get("gaps") or [])
        lead_frames = int(row.get("lead_frames") or 0)
        # Span tuyệt đối từng âm tiết trong clip (đã trừ lặng đầu/cuối).
        spans: list[tuple[int, int]] = []
        cursor = lead_frames
        for i, duration in enumerate(durations):
            spans.append((cursor, cursor + duration))
            cursor += duration + (gaps[i] if i < len(gaps) else 0)
        pauses_after = {int(p["after"]) for p in row.get("pauses", [])}
        cuts = cut_clip(spans, pauses_after, min_frames=min_frames, max_frames=max_frames)
        if len(cuts) > 1:
            stats["clips_split"] += 1
        for index, (start, end) in enumerate(cuts):
            frame_range = [spans[start][0], spans[end - 1][1]]
            if frame_range[1] - frame_range[0] < floor_frames:
                stats["dropped_tiny"] += 1
                continue
            segment = {
                "id": f"{row['id']}__s{index:02d}",
                "audio_path": row["audio_path"],
                "parent": row["id"],
                "frame_range": frame_range,
                "syllables": row["syllables"][start:end],
                "durations": durations[start:end],
                "gaps": gaps[start:end - 1] if gaps else [],
                "lead_frames": 0,
                "text": " ".join(str(s.get("raw", "")) for s in row["syllables"][start:end]),
            }
            out_rows.append(segment)
            lengths.append(frame_range[1] - frame_range[0])
            if (end - 1) in pauses_after and end < len(durations):
                stats["pause_cuts"] += 1
    stats["segments"] = len(out_rows)
    if lengths:
        stats["min_seg"] = min(lengths)
        stats["max_seg"] = max(lengths)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    """Cắt danh sách manifest; in thống kê + ghi report JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-seconds", type=float, default=3.0)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--min-segment-seconds", type=float, default=0.25,
                        help="Sàn độ dài segment giữ lại (mặc định 0,25 s; đuôi vụn bị bỏ)")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=256)
    args = parser.parse_args()

    min_frames = round(args.min_seconds * args.sample_rate / args.hop_length)
    max_frames = round(args.max_seconds * args.sample_rate / args.hop_length)
    floor_frames = round(args.min_segment_seconds * args.sample_rate / args.hop_length)
    report: dict[str, Any] = {"min_frames": min_frames, "max_frames": max_frames,
                              "floor_frames": floor_frames}
    for manifest in args.manifest:
        name = Path(manifest).stem
        stats = cut_manifest(manifest, Path(args.out_dir) / f"{name}_seg.jsonl",
                             min_frames=min_frames, max_frames=max_frames, floor_frames=floor_frames)
        report[name] = stats
        print(f"{name}: {stats['clips']} clip → {stats['segments']} segment "
              f"({stats['clips_split']} clip bị cắt · bỏ đuôi vụn {stats['dropped_tiny']}) · "
              f"cắt tại pause {stats['pause_cuts']} · dài {stats['min_seg']}–{stats['max_seg']} frame")
    report_path = Path(args.out_dir) / "cut_segments_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Report: {report_path} (min/max {min_frames}/{max_frames} frame ≈ "
          f"{args.min_seconds}–{args.max_seconds}s)")


if __name__ == "__main__":
    main()

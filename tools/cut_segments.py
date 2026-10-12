"""Cắt clip dài thành segment 3–10s tại pause/ranh giới câu — REBUILD_PLAN_M6 §2.3.

- Input: manifest đã có `durations` (từ `mfa_to_durations.py`) + `pauses` (pause ≥150ms ở
  ranh giới câu mà MFA xác nhận).
- **Cắt tại biên âm tiết** (không bao giờ cắt giữa âm tiết): ưu tiên điểm cắt là **pause**, nếu
  không có pause trong cửa sổ thì cắt tại biên âm tiết xa nhất còn trong `[min, max]` frame.
- Đuôi ngắn hơn `min` được gộp vào segment trước nếu tổng ≤ `max`.
- Row của segment giữ nguyên `audio_path` của clip gốc + thêm `frame_range` [start, end) —
  dataset cắt mel/f0/vuv theo frame từ clip đầy đủ (đúng spec, tránh hiệu ứng biên STFT).

Chạy:
    python tools/cut_segments.py --manifest data/codelung/train_clean_v2_dur.jsonl \
        --manifest data/codelung/val_clean_v2_dur.jsonl --out-dir data/codelung
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def cut_clip(durations: list[int], pauses_after: set[int], *,
             min_frames: int, max_frames: int) -> list[tuple[int, int]]:
    """Trả danh sách (start, end) theo chỉ số âm tiết — greedy, ưu tiên pause trong cửa sổ."""
    count = len(durations)
    if count == 0:
        return []
    cumulative = [0] * (count + 1)
    for index, duration in enumerate(durations):
        cumulative[index + 1] = cumulative[index] + duration

    segments: list[tuple[int, int]] = []
    start = 0
    while start < count:
        candidates: list[int] = []
        for end in range(start + 1, count + 1):
            length = cumulative[end] - cumulative[start]
            if length > max_frames and end > start + 1:
                break
            candidates.append(end)
        window = [end for end in candidates
                  if cumulative[end] - cumulative[start] >= min_frames or end == count]
        if not window:  # phần còn lại ngắn hơn min → lấy hết
            window = [candidates[-1]]
        paused = [end for end in window if (end - 1) in pauses_after]
        chosen = paused[-1] if paused else window[-1]
        segments.append((start, chosen))
        start = chosen

    if len(segments) >= 2:
        merge_start, merge_end = segments[-1]
        if cumulative[merge_end] - cumulative[merge_start] < min_frames:
            previous_start, _ = segments[-2]
            if cumulative[merge_end] - cumulative[previous_start] <= max_frames:
                segments[-2] = (previous_start, merge_end)
                segments.pop()
    return segments


def cut_manifest(manifest: str | Path, out_path: str | Path, *,
                 min_frames: int, max_frames: int) -> dict[str, Any]:
    """Cắt toàn bộ manifest thành segment; ghi manifest mới + trả thống kê."""
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_rows: list[dict[str, Any]] = []
    stats = {"clips": len(rows), "segments": 0, "pause_cuts": 0, "clips_split": 0,
             "min_seg": None, "max_seg": None}
    lengths: list[int] = []
    for row in rows:
        durations = row.get("durations")
        if not durations:
            continue
        pauses_after = {int(p["after"]) for p in row.get("pauses", [])}
        cuts = cut_clip(durations, pauses_after, min_frames=min_frames, max_frames=max_frames)
        if len(cuts) > 1:
            stats["clips_split"] += 1
        cumulative = [0]
        for duration in durations:
            cumulative.append(cumulative[-1] + duration)
        for index, (start, end) in enumerate(cuts):
            frame_range = [cumulative[start], cumulative[end]]
            segment = {
                "id": f"{row['id']}__s{index:02d}",
                "audio_path": row["audio_path"],
                "parent": row["id"],
                "frame_range": frame_range,
                "syllables": row["syllables"][start:end],
                "durations": durations[start:end],
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
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=256)
    args = parser.parse_args()

    min_frames = round(args.min_seconds * args.sample_rate / args.hop_length)
    max_frames = round(args.max_seconds * args.sample_rate / args.hop_length)
    report: dict[str, Any] = {"min_frames": min_frames, "max_frames": max_frames}
    for manifest in args.manifest:
        name = Path(manifest).stem
        stats = cut_manifest(manifest, Path(args.out_dir) / f"{name}_seg.jsonl",
                             min_frames=min_frames, max_frames=max_frames)
        report[name] = stats
        print(f"{name}: {stats['clips']} clip → {stats['segments']} segment "
              f"({stats['clips_split']} clip bị cắt) · cắt tại pause {stats['pause_cuts']} · "
              f"dài {stats['min_seg']}–{stats['max_seg']} frame")
    report_path = Path(args.out_dir) / "cut_segments_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Report: {report_path} (min/max {min_frames}/{max_frames} frame ≈ "
          f"{args.min_seconds}–{args.max_seconds}s)")


if __name__ == "__main__":
    main()

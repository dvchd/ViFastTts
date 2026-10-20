#!/usr/bin/env python
"""tools/v32_prep_clip_manifest.py — v3.2: manifest clip-level + `seg4` (KHÔNG sửa audio).

v3.2 = giữ nguyên audio/cache của b8, chỉ đổi chỗ cắt segment: pause dài (> `--max-gap`
frame, mặc định 60 ≈ 0,64 s) trở thành **ranh giới segment** thay vì bị cắt/nối trong audio.

Tool này chuẩn bị đầu vào cho `cut_segments.py`:
  - clip-level (train/val): `durations`/`gaps`/`lead_frames`/`pauses` giữ nguyên; `seg4` lấy từ
    manifest segment cũ (nối theo thứ tự frame). Clip mà manifest segment cũ chỉ phủ một **tiền tố**
    âm tiết (đuôi vụn bị bỏ khi cắt cũ) được **cắt cụt** đúng phần đó — không mất thêm dữ liệu so b8.
  - holdout: `seg_durations` đã có sẵn ở clip-level, chỉ đổi tên thành `seg4`.
  - Clip lệch nội dung âm tiết (raw khác) hoặc thiếu nhãn → bỏ (log).

Chạy: .venv/bin/python tools/v32_prep_clip_manifest.py \\
    --clip-manifest data/codelung/train_clean_v2_dur.jsonl \\
    --seg-manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \\
    --out data/codelung/v32/train_clean_v2_dur.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def group_segments(path: str) -> dict[str, list[dict]]:
    """Gom segment cũ theo clip, sắp theo `frame_range[0]` (thứ tự âm tiết)."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            parent = str(row.get("parent") or row["id"])
            grouped[parent].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["frame_range"][0]))
    return dict(grouped)


def prep_clips(clip_path: str, seg_path: str, out_path: str) -> dict:
    """Ghi manifest clip-level + `seg4` (cắt cụt tiền tố nếu manifest cũ phủ ngắn hơn)."""
    clips = [json.loads(line) for line in Path(clip_path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    grouped = group_segments(seg_path)
    stats = {"clips": len(clips), "kept": 0, "truncated": 0, "missing": 0, "mismatch": 0}
    out_rows: list[dict] = []
    for row in clips:
        segs = grouped.get(str(row["id"]), [])
        syllables = [s for seg in segs for s in seg["syllables"]]
        seg4 = [d for seg in segs for d in seg["seg_durations"]]
        if not syllables:
            stats["missing"] += 1
            continue
        if len(syllables) > len(row["syllables"]):
            stats["mismatch"] += 1
            continue
        ref = row["syllables"][: len(syllables)]
        if any(str(a.get("raw")) != str(b.get("raw")) for a, b in zip(syllables, ref)):
            stats["mismatch"] += 1
            continue
        count = len(syllables)
        if count < len(row["syllables"]):
            stats["truncated"] += 1
        durations = [int(d) for d in row["durations"][:count]]
        gaps = [int(g) for g in (row.get("gaps") or [])[:count]]
        out = {
            "id": row["id"],
            "audio_path": row["audio_path"],
            "text": " ".join(str(s.get("raw", "")) for s in ref),
            "syllables": ref,
            "durations": durations,
            "gaps": gaps,
            "lead_frames": int(row.get("lead_frames") or 0),
            "pauses": [p for p in row.get("pauses", []) if int(p["after"]) < count],
            "seg4": seg4,
            "seg_method": segs[0].get("seg_method"),
            "seg_method_version": segs[0].get("seg_method_version"),
            "seg_weights_hash": segs[0].get("seg_weights_hash"),
        }
        out_rows.append(out)
        stats["kept"] += 1
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    return stats


def prep_holdout(clip_path: str, out_path: str) -> dict:
    """Holdout: `seg_durations` đã ở clip-level → đổi tên `seg4`."""
    rows = [json.loads(line) for line in Path(clip_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    out_rows = []
    for row in rows:
        seg4 = row.get("seg4") or row.get("seg_durations")
        if not seg4 or len(seg4) != len(row["syllables"]):
            continue
        out = dict(row)
        out["seg4"] = seg4
        out_rows.append(out)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    return {"clips": len(rows), "kept": len(out_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--seg-manifest", default=None, help="Bỏ trống nếu dùng --holdout")
    parser.add_argument("--holdout", action="store_true",
                        help="Clip-level đã có `seg_durations` (holdout)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.holdout:
        stats = prep_holdout(args.clip_manifest, args.out)
    else:
        if not args.seg_manifest:
            raise SystemExit("Cần --seg-manifest (hoặc --holdout)")
        stats = prep_clips(args.clip_manifest, args.seg_manifest, args.out)
    print(f"{Path(args.clip_manifest).name} → {args.out}: {stats}")


if __name__ == "__main__":
    main()

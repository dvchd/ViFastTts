#!/usr/bin/env python
"""tools/v3_segments_slot4.py — Data v3 · B3b: ghép `seg_durations [N,4]` vào segment v3.

Thay `add_seg_durations.py` (cần TextGrid MFA) bằng cách **chuyển nhãn hybrid cũ**: manifest clip-level
v3 mang `seg4` (O/M/N/C mỗi âm tiết, từ nhãn hybrid MFA của corpus cũ — cap pause không đổi lòng âm
tiết nên nhãn chuyển nguyên vẹn). Với mỗi segment (từ `cut_segments.py`), cắt `seg4` theo đúng dải
âm tiết (đối chiếu `frame_range[0]` với spans của clip) → `seg_durations` + method/hash.

Chạy: .venv/bin/python tools/v3_segments_slot4.py \
    --clip-manifests data/codelung/v3/train_clean_v2_dur.jsonl,data/codelung/v3/val_clean_v2_dur.jsonl,data/codelung/v3/holdout_clean_v2_dur.jsonl \
    --seg-dir data/codelung/v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def spans_of(clip: dict) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = int(clip.get("lead_frames") or 0)
    durations = [int(d) for d in clip["durations"]]
    gaps = [int(g) for g in clip.get("gaps") or []]
    for i, duration in enumerate(durations):
        spans.append((cursor, cursor + duration))
        cursor += duration + (gaps[i] if i < len(gaps) else 0)
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-manifests", required=True)
    parser.add_argument("--seg-dir", default="data/codelung/v3")
    args = parser.parse_args()

    seg_dir = ROOT / args.seg_dir
    for manifest in args.clip_manifests.split(","):
        clip_rows = [json.loads(line) for line in (ROOT / manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {str(r["id"]): r for r in clip_rows}
        stem = Path(manifest).stem  # ví dụ train_clean_v2_dur
        seg_path = seg_dir / f"{stem}_seg.jsonl"
        out_path = seg_dir / f"{stem}_seg_slot4.jsonl"
        seg_rows = [json.loads(line) for line in seg_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out_rows = []
        for seg in seg_rows:
            clip = by_id[str(seg["parent"])]
            spans = spans_of(clip)
            target = int(seg["frame_range"][0])
            start = next((i for i, (a, _) in enumerate(spans) if a == target), None)
            if start is None:
                raise SystemExit(f"LỖI: không khớp frame_range {target} cho {seg['id']}")
            count = len(seg["syllables"])
            seg4 = [list(map(int, x)) for x in clip["seg4"][start:start + count]]
            if len(seg4) != count or sum(sum(x) for x in seg4) != sum(int(d) for d in seg["durations"]):
                raise SystemExit(f"LỖI: seg4 không khớp durations cho {seg['id']}")
            seg["seg_durations"] = seg4
            seg["seg_method"] = clip.get("seg_method")
            seg["seg_method_version"] = clip.get("seg_method_version")
            seg["seg_weights_hash"] = clip.get("seg_weights_hash")
            out_rows.append(seg)
        out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
        print(f"{seg_path.name}: {len(seg_rows)} segment → {out_path} ({len(out_rows)} row)", flush=True)


if __name__ == "__main__":
    main()

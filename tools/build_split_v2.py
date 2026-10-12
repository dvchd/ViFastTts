"""Chia split v2 + lock + `eval_clips_v1` (docs/REBUILD_PLAN_M6.md §3(4)).

Trên pool 25 video chủ kênh (sau lọc giọng + chuẩn hoá loudness):
- **12 video ít giờ nhất** = val/test videos; **val** = tối đa **150 clip/video** (mẫu ngẫu
  nhiên, seed cố định); phần clip còn lại của 12 video này = **holdout** (không vào train).
- **13 video còn lại** = train.
- Bản **clean** (`fallback_ratio == 0` trong clips.jsonl) cho acoustic + MFA;
  bản đầy đủ dùng cho vocoder (không cần text).
- `eval_clips_v1.jsonl`: **200 clip** lấy từ holdout, ưu tiên clip sạch fallback, phân tầng
  theo video, kèm `text_sha1` (bộ cố định để so CER/nghe giữa các mốc — không dùng train).
- `split_lock.json`: khóa video + seed + số clip (chạy lại cho cùng kết quả).

Chạy:
    python tools/build_split_v2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Đọc JSONL thành list dict."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _hours(rows: list[dict[str, Any]], durations: dict[str, float]) -> float:
    """Tổng thời lượng (giờ) của các clip."""
    return sum(durations.get(str(r["id"]), 0.0) for r in rows) / 3600.0


def main() -> None:
    """Chạy chia split và ghi kết quả."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/codelung/manifest_v2.jsonl")
    parser.add_argument("--clips", default="data/codelung/clips.jsonl")
    parser.add_argument("--out-dir", default="data/codelung")
    parser.add_argument("--val-videos", type=int, default=12, help="Số video làm val/test (ít giờ nhất)")
    parser.add_argument("--cap", type=int, default=150, help="Trần clip val mỗi video")
    parser.add_argument("--eval-clips", type=int, default=200, help="Số clip cho eval_clips_v1")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    out = Path(args.out_dir)
    rows = _load_jsonl(args.manifest)
    clips = {str(c["id"]): c for c in _load_jsonl(args.clips)}
    durations = {key: float(meta.get("duration", 0.0)) for key, meta in clips.items()}
    # Fallback suy từ chính manifest (status âm tiết) — cập nhật sau mỗi lần reparse overrides,
    # không phụ thuộc `clips.jsonl` (đóng băng từ lúc recut).
    fallback: dict[str, float] = {}
    for row in rows:
        if any(str(s.get("status", "VALID")) != "VALID" for s in row.get("syllables", [])):
            fallback[str(row["id"])] = 1.0
    video_of = {key: str(meta.get("video", "")) for key, meta in clips.items()}

    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_video.setdefault(video_of[str(row["id"])], []).append(row)

    videos = sorted(by_video, key=lambda name: (_hours(by_video[name], durations), name))
    val_videos = videos[: args.val_videos]
    train_videos = videos[args.val_videos:]
    rng = random.Random(args.seed)

    val_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    for video in val_videos:
        pool = sorted(by_video[video], key=lambda r: str(r["id"]))
        take = min(args.cap, len(pool))
        picked = rng.sample(pool, take)
        picked_ids = {str(r["id"]) for r in picked}
        val_rows.extend(picked)
        holdout_rows.extend(r for r in pool if str(r["id"]) not in picked_ids)
    train_rows = [row for video in train_videos for row in by_video[video]]
    # Sắp xếp lại cố định theo id cho mọi file (tái lập + dễ diff).
    for group in (train_rows, val_rows, holdout_rows):
        group.sort(key=lambda r: str(r["id"]))

    def write(name: str, group: list[dict[str, Any]], clean_only: bool = False) -> int:
        selected = [r for r in group if fallback.get(str(r["id"]), 0.0) == 0.0] if clean_only else group
        with (out / name).open("w", encoding="utf-8") as fh:
            for row in selected:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(selected)

    counts = {
        "train_v2": write("train_v2.jsonl", train_rows),
        "val_v2": write("val_v2.jsonl", val_rows),
        "holdout_v2": write("holdout_v2.jsonl", holdout_rows),
        "train_clean_v2": write("train_clean_v2.jsonl", train_rows, clean_only=True),
        "val_clean_v2": write("val_clean_v2.jsonl", val_rows, clean_only=True),
        "holdout_clean_v2": write("holdout_clean_v2.jsonl", holdout_rows, clean_only=True),
    }

    # eval_clips_v1: 200 clip từ holdout, ưu tiên clean, chia vòng tròn qua các video
    # (3 video nhỏ có thể không còn holdout → bù bằng các video khác; vẫn tất định theo seed).
    eval_rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for want_clean in (True, False):
        while len(eval_rows) < args.eval_clips:
            progressed = False
            for video in val_videos:
                if len(eval_rows) >= args.eval_clips:
                    break
                pool = [r for r in holdout_rows
                        if video_of[str(r["id"])] == video
                        and str(r["id"]) not in used
                        and (fallback.get(str(r["id"]), 0.0) == 0.0) == want_clean]
                if not pool:
                    continue
                row = rng.choice(pool)
                used.add(str(row["id"]))
                eval_rows.append(row)
                progressed = True
            if not progressed:
                break
    eval_rows.sort(key=lambda r: str(r["id"]))
    with (out / "eval_clips_v1.jsonl").open("w", encoding="utf-8") as fh:
        for row in eval_rows:
            payload = {
                "id": row["id"],
                "audio_path": row["audio_path"],
                "video": video_of[str(row["id"])],
                "text": row.get("text", ""),
                "text_sha1": hashlib.sha1(str(row.get("text", "")).encode("utf-8")).hexdigest(),
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    lock = {
        "seed": args.seed,
        "val_videos": val_videos,
        "train_videos": train_videos,
        "cap": args.cap,
        "counts": counts,
        "eval_clips": len(eval_rows),
        "hours": {
            "train_v2": round(_hours(train_rows, durations), 2),
            "val_v2": round(_hours(val_rows, durations), 2),
            "holdout_v2": round(_hours(holdout_rows, durations), 2),
            "train_clean_v2": round(_hours([r for r in train_rows if fallback.get(str(r["id"]), 0.0) == 0.0], durations), 2),
            "val_clean_v2": round(_hours([r for r in val_rows if fallback.get(str(r["id"]), 0.0) == 0.0], durations), 2),
            "holdout_clean_v2": round(_hours([r for r in holdout_rows if fallback.get(str(r["id"]), 0.0) == 0.0], durations), 2),
        },
    }
    (out / "split_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")

    print("Split v2 (25 video chủ kênh):")
    for key, value in counts.items():
        print(f"  {key:18s} {value:6d} clip · {lock['hours'].get(key, 0.0):6.2f} h")
    print(f"  eval_clips_v1      {len(eval_rows):6d} clip (holdout, ưu tiên clean)")
    print(f"  val videos ({len(val_videos)}): {', '.join(val_videos)}")
    print(f"Đã khóa → {out / 'split_lock.json'}")


if __name__ == "__main__":
    main()

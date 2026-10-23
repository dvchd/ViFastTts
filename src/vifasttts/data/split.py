"""Chia manifest train/val **theo video nguồn** để tránh rò rỉ dữ liệu.

Mỗi mẫu có id dạng `<video_id>_<NNNN>`; toàn bộ clip của một video chỉ nằm ở
train hoặc val, không bao giờ cả hai. Dùng cho `tools/split_manifest.py`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def video_id_of(sample_id: str) -> str:
    """Tách video id từ **clip id**: `lRojc8KDJ6k_0001` → `lRojc8KDJ6k`.

    Không dùng cho segment id (`...__s00`): id segment có `__sNN` ở cuối nên
    `rsplit("_", 1)` sẽ trả sai. Dùng `video_id_of_row` cho row manifest.
    """
    return sample_id.rsplit("_", 1)[0] if "_" in sample_id else sample_id


def video_id_of_row(row: dict[str, Any]) -> str:
    """Tách video id từ row manifest (ưu tiên `parent` của clip cha).

    `parent` có dạng `<video>_<NNNN>` kể cả khi video id chứa `_` (vd
    `4ITcvEa_Acc_0001` → `4ITcvEa_Acc`). Nếu row không có `parent`, fallback
    về `id` cấp clip; không dùng trực tiếp id segment.
    """
    parent = row.get("parent")
    if parent:
        return str(parent).rsplit("_", 1)[0]
    sample_id = str(row.get("id", ""))
    if "__s" in sample_id:  # segment id: bỏ hậu tố __sNN trước khi tách
        sample_id = sample_id.split("__s", 1)[0]
    return video_id_of(sample_id)


def pick_val_videos(
    video_ids: list[str], *, val_videos: int | None = None, val_ratio: float = 0.05, seed: int = 13,
) -> list[str]:
    """Chọn video cho tập validation, tất định theo `seed`.

    - `val_videos` được đặt: lấy đúng số đó (tối đa `len(video_ids) - 1` để còn train).
    - Ngược lại: lấy theo tỉ lệ `val_ratio`, tối thiểu 1 video (nếu có ≥ 2 video).
    """
    unique = sorted(set(video_ids))
    if len(unique) < 2:
        return []
    if val_videos is not None:
        count = max(0, min(val_videos, len(unique) - 1))
    else:
        count = max(1, round(val_ratio * len(unique)))
        count = min(count, len(unique) - 1)
    order = sorted(unique, key=lambda v: hashlib.sha1(f"{seed}:{v}".encode()).hexdigest())
    return sorted(order[:count])


def split_rows(
    rows: list[dict[str, Any]], *, val_videos: int | None = None, val_ratio: float = 0.05, seed: int = 13,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Chia rows thành (train, val, danh sách video val)."""
    videos = [video_id_of(str(row.get("id", ""))) for row in rows]
    val_set = set(pick_val_videos(videos, val_videos=val_videos, val_ratio=val_ratio, seed=seed))
    train = [row for row, video in zip(rows, videos, strict=True) if video not in val_set]
    val = [row for row, video in zip(rows, videos, strict=True) if video in val_set]
    return train, val, sorted(val_set)


def write_split(
    manifest: str | Path,
    out_dir: str | Path,
    *,
    val_videos: int | None = None,
    val_ratio: float = 0.05,
    seed: int = 13,
    clips: str | Path | None = None,
) -> dict[str, Any]:
    """Đọc manifest JSONL → ghi `train.jsonl`/`val.jsonl`, trả thống kê.

    Nếu có `clips.jsonl` (hoặc mặc định cạnh manifest), kèm số giờ mỗi tập.
    """
    manifest_path = Path(manifest)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train, val, val_set = split_rows(rows, val_videos=val_videos, val_ratio=val_ratio, seed=seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train.jsonl", train), ("val.jsonl", val)):
        (out / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in subset),
            encoding="utf-8",
        )

    clips_path = Path(clips) if clips is not None else manifest_path.parent / "clips.jsonl"
    durations: dict[str, float] = {}
    if clips_path.exists():
        for line in clips_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                durations[str(record.get("id"))] = float(record.get("duration", 0.0))

    def _hours(subset: list[dict[str, Any]]) -> float:
        return sum(durations.get(str(row.get("id")), 0.0) for row in subset) / 3600.0

    return {
        "train": len(train),
        "val": len(val),
        "train_hours": _hours(train),
        "val_hours": _hours(val),
        "val_videos": val_set,
    }

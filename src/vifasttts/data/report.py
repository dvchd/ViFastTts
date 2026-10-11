"""Tổng hợp & kiểm tra nhanh dataset YouTube (`clips.jsonl` + `stats.json`).

Dùng cho `tools/dataset_report.py`: xem per-video, tổng giờ, mức fallback và các
clip bất thường (quá ngắn/dài, fallback cao) trước khi chia train/val hoặc train.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_clips(dataset: str | Path) -> list[dict[str, Any]]:
    """Đọc `clips.jsonl` của dataset (rỗng nếu chưa có)."""
    path = Path(dataset) / "clips.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dataset_summary(
    dataset: str | Path,
    *,
    min_duration: float = 1.0,
    max_duration: float = 20.5,
    flag_fallback: float = 0.2,
) -> dict[str, Any]:
    """Tổng hợp dataset thành dict: totals, per-video và danh sách clip cần xem lại."""
    root = Path(dataset)
    records = load_clips(root)
    per_video: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"clips": 0, "seconds": 0.0, "zero_fallback": 0, "high_fallback": 0}
    )
    flagged: list[dict[str, Any]] = []
    for record in records:
        video = str(record.get("video", "?"))
        duration = float(record.get("duration", 0.0))
        ratio = float(record.get("fallback_ratio", 0.0))
        bucket = per_video[video]
        bucket["clips"] += 1
        bucket["seconds"] += duration
        if ratio == 0.0:
            bucket["zero_fallback"] += 1
        if ratio > flag_fallback:
            bucket["high_fallback"] += 1
            flagged.append({"id": record.get("id"), "why": "fallback cao", "value": ratio})
        if duration < min_duration or duration > max_duration:
            flagged.append({"id": record.get("id"), "why": "độ dài bất thường", "value": duration})

    total_seconds = sum(float(record.get("duration", 0.0)) for record in records)
    total_fallback = sum(float(record.get("fallback_ratio", 0.0)) for record in records)
    stats_path = root / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    return {
        "dataset": str(root),
        "videos": len(per_video),
        "clips": len(records),
        "seconds": total_seconds,
        "hours": total_seconds / 3600.0,
        "avg_duration": total_seconds / max(1, len(records)),
        "avg_fallback": total_fallback / max(1, len(records)),
        "zero_fallback_clips": sum(1 for r in records if float(r.get("fallback_ratio", 0.0)) == 0.0),
        "per_video": {video: dict(bucket) for video, bucket in sorted(per_video.items())},
        "flagged": flagged[:50],
        "license": stats.get("license"),
        "oov_top": stats.get("oov_top", []),
        "manifest_issues": stats.get("manifest_issues", []),
    }


def format_report(summary: dict[str, Any]) -> str:
    """Kết xuất báo cáo dạng chữ để in ra terminal."""
    lines = [
        f"Dataset: {summary['dataset']}",
        (
            f"Video: {summary['videos']} · Clip: {summary['clips']} · "
            f"Tổng: {summary['hours']:.2f} giờ · TB {summary['avg_duration']:.2f}s/clip"
        ),
        (
            f"Clip 0 fallback: {summary['zero_fallback_clips']}/{summary['clips']} · "
            f"fallback TB: {summary['avg_fallback'] * 100:.1f}%"
        ),
        (
            f"Giấy phép: {summary.get('license') or '—'} · "
            f"vấn đề manifest: {len(summary.get('manifest_issues') or [])}"
        ),
        "",
        f"{'video':<16} {'clip':>5} {'phút':>7} {'TB(s)':>6} {'0-fb':>5} {'fb-cao':>6}",
    ]
    for video, bucket in summary["per_video"].items():
        lines.append(
            f"{video:<16} {bucket['clips']:>5} {bucket['seconds'] / 60:>7.1f} "
            f"{bucket['seconds'] / max(1, bucket['clips']):>6.1f} "
            f"{bucket['zero_fallback']:>5} {bucket['high_fallback']:>6}"
        )
    if summary.get("oov_top"):
        preview = ", ".join(f"{token}×{count}" for token, count in summary["oov_top"][:10])
        lines += ["", f"Từ lạ nhiều nhất: {preview}"]
    if summary.get("flagged"):
        lines += ["", "Clip cần xem lại (tối đa 50):"]
        lines += [f"  {item['id']}: {item['why']} = {item['value']}" for item in summary["flagged"]]
    return "\n".join(lines)

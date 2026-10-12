"""Báo cáo chất lượng alignment từ TextGrid MFA (bổ trợ C4 khi score DB khó diễn giải).

Mỗi clip tính:
- `words`, `floor_rate` (tỉ lệ từ ≤ 55 ms — sát sàn min-duration HMM),
- `coverage` (tổng thời lượng word / span), `word_dur_median_s`,
- `long_phone_rate` (tỉ lệ phone **hữu thanh** > 0,5 s — vi phạm cứng C4).

Chạy:
    python tools/align_quality_report.py --textgrid-dir data/codelung/mfa_v2c_aligned \
        --out artifacts/dataset_v2/align_quality_v2c.json [--drop-list drop.txt]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from mfa_to_durations import (
    HARD_PHONE_SECONDS,
    SILENCE_PHONES,
    parse_long_textgrid,
)


def clip_metrics(grid_path: Path) -> dict[str, Any] | None:
    """Metric của một TextGrid; None nếu không có word."""
    tiers = parse_long_textgrid(grid_path)
    words = [(a, b) for a, b, t in tiers.get("words", []) if t.strip()]
    if not words:
        return None
    phones = [(a, b, label) for a, b, label in tiers.get("phones", [])
              if label.strip().lower() not in SILENCE_PHONES]
    durations = [b - a for a, b in words]
    span = words[-1][1] - words[0][0]
    return {
        "clip": grid_path.stem,
        "words": len(words),
        "floor_rate": round(sum(1 for d in durations if d <= 0.055) / len(durations), 3),
        "coverage": round(sum(durations) / span, 3) if span > 0 else 0.0,
        "word_dur_median_s": round(statistics.median(durations), 3),
        "long_phone_rate": round(
            sum(1 for a, b, _ in phones if b - a > HARD_PHONE_SECONDS) / max(1, len(phones)),
            4,
        ),
    }


def main() -> None:
    """Chạy báo cáo trên một thư mục TextGrid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrid-dir", required=True)
    parser.add_argument("--out", required=True, help="File JSON report")
    parser.add_argument("--floor-rate-threshold", type=float, default=0.15,
                        help="Ngưỡng floor_rate để vào drop-list (mặc định 0,15)")
    parser.add_argument("--drop-list", default=None, help="Ghi danh sách id vượt ngưỡng")
    args = parser.parse_args()

    grid_dir = Path(args.textgrid_dir)
    clips: list[dict[str, Any]] = []
    for grid in sorted(grid_dir.glob("*.TextGrid")):
        metrics = clip_metrics(grid)
        if metrics:
            clips.append(metrics)
    if not clips:
        print("Không có TextGrid nào")
        return

    floor_rates = sorted(c["floor_rate"] for c in clips)
    coverages = sorted(c["coverage"] for c in clips)
    medians = sorted(c["word_dur_median_s"] for c in clips)
    long_rates = sorted(c["long_phone_rate"] for c in clips)

    def percentile(values: list[float], q: float) -> float:
        return values[min(int(len(values) * q), len(values) - 1)]

    report = {
        "clips": len(clips),
        "floor_rate": {"p50": percentile(floor_rates, 0.5), "p90": percentile(floor_rates, 0.9),
                       "mean": round(statistics.mean(floor_rates), 4)},
        "coverage": {"p10": percentile(coverages, 0.1), "p50": percentile(coverages, 0.5)},
        "word_dur_median_s": {"p10": percentile(medians, 0.1), "p50": percentile(medians, 0.5)},
        "long_phone_rate": {"p50": percentile(long_rates, 0.5), "p99": percentile(long_rates, 0.99)},
    }
    print(f"{len(clips)} clip · floor_rate p50 {report['floor_rate']['p50']} "
          f"(mean {report['floor_rate']['mean']}) · coverage p10 {report['coverage']['p10']} "
          f"p50 {report['coverage']['p50']} · word dur p50 {report['word_dur_median_s']['p50']}s "
          f"· long-phone p99 {report['long_phone_rate']['p99']}")

    if args.drop_list:
        over = sorted(c["clip"] for c in clips if c["floor_rate"] > args.floor_rate_threshold)
        Path(args.drop_list).write_text("\n".join(over) + ("\n" if over else ""), encoding="utf-8")
        report["drop_count"] = len(over)
        print(f"drop-list (floor_rate > {args.floor_rate_threshold}): {len(over)} clip "
              f"({len(over) / len(clips):.1%}) → {args.drop_list}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()

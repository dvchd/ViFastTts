"""Tính stats cho target variance: log-F0 (scalar) + **4 hệ số DCT** (z-score riêng từng hệ số)
+ energy (scalar/âm tiết) — docs/AI_REVIEW_ANSWERS_R3 §C2, REBUILD_PLAN_M6 §2.2 (#2, #6).

Quy ước:
- Bỏ clip có voiced-ratio < `--voiced-ratio-min` (0,3) — như R3 §3.4 (tránh clip toàn vô thanh).
- Mỗi âm tiết: lấy frame theo `durations` (nếu manifest chưa có → chia đều); chỉ dùng frame
  hữu thanh (`f0 > 0`); < 2 frame hữu thanh → bỏ qua âm tiết (không xác định DCT).
- **DCT**: `dct4_frames(log f0 hữu thanh)` — âm tiết ngắn (n < 4) chỉ có k < n hệ số; stats
  từng hệ số chỉ tính trên các âm tiết có hệ số đó (loss mask phần thiếu, đúng R3 C2).
- Energy: trung bình frame trong span, đơn vị log-power (như `extract_features`).

Chạy:
    python tools/compute_f0_stats.py --manifest data/codelung/train_clean_v2_dur.jsonl \
        --config configs/small_codelung.yaml --out data/codelung/f0_stats.json \
        --cache-dir cache/codelung_v2 --sample 2000 --workers 8

Kết quả dùng cho: z-score từng hệ số DCT khi train (dataset/model #8) và `f0_stats_hash`
ghi vào checkpoint (REBUILD_PLAN_M6 §2.2 #6).
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def _worker(payload: tuple[dict[str, Any], dict[str, Any], str | None, float, str, float, float]) -> dict[str, Any]:
    """Một clip: trả tổng/thống kê tích luỹ (không trả dữ liệu thô).

    `mode="scalar"`: đếm log-F0 (mean/std scalar) + energy.
    `mode="dct"`: DCT4 trên log-F0 **đã z-score scalar** (f0_mean/f0_std từ pass 1) —
    đúng quy ước R3 §C2 (hệ số z-score riêng tính ở pass này).
    """
    import torch

    from vifasttts.data.cache import FeatureCache
    from vifasttts.data.dataset import allocate_durations
    from vifasttts.data.features import dct4_frames, extract_features

    row, config, cache_dir, voiced_ratio_min, mode, f0_mean, f0_std = payload
    cache = FeatureCache(cache_dir, config) if cache_dir else None
    extracted = cache.get(row) if cache is not None else None
    if extracted is None:
        extracted = extract_features(row["audio_path"], config)
        if cache is not None:
            cache.put(row, extracted)

    frame_f0 = extracted["f0"].double()
    frame_energy = extracted["energy"].double()
    frame_vuv = extracted["vuv"].double()
    frames = frame_f0.size(0)
    voiced_ratio = float(frame_vuv.mean()) if frames else 0.0
    voiced_positive = frame_f0 > 0

    sums = {
        "clips": 1.0,
        "voiced_frames": float(voiced_positive.sum()),
        "f0_sum": float(torch.log(frame_f0[voiced_positive]).sum()) if voiced_positive.any() else 0.0,
        "f0_sumsq": float((torch.log(frame_f0[voiced_positive]) ** 2).sum()) if voiced_positive.any() else 0.0,
        "f0_count": float(voiced_positive.sum()),
        "energy_sum": 0.0, "energy_sumsq": 0.0, "energy_count": 0.0,
        "syllables": 0.0, "skipped_syllables": 0.0, "low_voiced": 0.0,
    }
    for k in range(4):
        sums[f"dct{k}_sum"] = 0.0
        sums[f"dct{k}_sumsq"] = 0.0
        sums[f"dct{k}_count"] = 0.0

    if frames == 0 or voiced_ratio < voiced_ratio_min:  # clip quá ít hữu thanh (R3 §3.4)
        sums["low_voiced"] = 1.0
        return sums
    durations = row.get("durations")
    if not durations or sum(durations) != frames:
        durations = allocate_durations(frames, len(row["syllables"]))

    start = 0
    for duration in durations:
        end = min(frames, start + duration)
        if end <= start:
            start = end
            continue
        sums["syllables"] += 1.0
        span_f0 = frame_f0[start:end]
        voiced = span_f0 > 0
        if mode == "dct":
            if int(voiced.sum()) < 2:
                sums["skipped_syllables"] += 1.0
            else:
                z_values = (torch.log(span_f0[voiced]) - f0_mean) / max(f0_std, 1e-6)
                coefficients = dct4_frames(z_values)
                for k, value in enumerate(coefficients.tolist()):
                    sums[f"dct{k}_sum"] += value
                    sums[f"dct{k}_sumsq"] += value * value
                    sums[f"dct{k}_count"] += 1.0
        sums["energy_sum"] += float(frame_energy[start:end].mean())
        sums["energy_sumsq"] += float(frame_energy[start:end].mean() ** 2)
        sums["energy_count"] += 1.0
        start = end
    return sums


def compute_stats(manifests: list[str], config: dict[str, Any], *, sample: int = 2000,
                  seed: int = 42, workers: int = 8, cache_dir: str | None = None,
                  voiced_ratio_min: float = 0.3) -> dict[str, Any]:
    """Chạy thống kê trên mẫu clip; trả dict JSON (mean/std từng đại lượng)."""
    import random

    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        rows.extend(
            json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()
        )
    picked = random.Random(seed).sample(rows, min(sample, len(rows)))

    def run(mode: str, f0_mean: float, f0_std: float) -> dict[str, float]:
        payloads = [(row, config, cache_dir, voiced_ratio_min, mode, f0_mean, f0_std) for row in picked]
        totals: dict[str, float] = {}
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            for index, part in enumerate(pool.map(_worker, payloads), start=1):
                for key, value in part.items():
                    totals[key] = totals.get(key, 0.0) + value
                if index % 200 == 0:
                    print(f"  [{mode}] {index}/{len(payloads)} clip", flush=True)
        return totals

    def stats(totals: dict[str, float], prefix: str) -> tuple[float, float, float]:
        count = totals.get(f"{prefix}_count", 0.0)
        total = totals.get(f"{prefix}_sum", 0.0)
        total_sq = totals.get(f"{prefix}_sumsq", 0.0)
        if count <= 0:
            return 0.0, 0.0, 0.0
        mean = total / count
        variance = max(0.0, total_sq / count - mean * mean)
        return mean, math.sqrt(variance), count

    # Pass 1: log-F0 scalar + energy. Pass 2: DCT4 trên log-F0 đã z-score scalar (R3 §C2).
    totals = run("scalar", 0.0, 1.0)
    f0_mean, f0_std, f0_count = stats(totals, "f0")
    energy_mean, energy_std, energy_count = stats(totals, "energy")
    totals_dct = run("dct", f0_mean, f0_std)

    dct_mean, dct_std, dct_count = [], [], []
    for k in range(4):
        mean, std, count = stats(totals_dct, f"dct{k}")
        dct_mean.append(mean)
        dct_std.append(std)
        dct_count.append(count)
    return {
        "clips": int(totals.get("clips", 0.0)),
        "low_voiced_clips": int(totals.get("low_voiced", 0.0)),
        "syllables": int(totals.get("syllables", 0.0)),
        "skipped_syllables": int(totals_dct.get("skipped_syllables", 0.0)),
        "voiced_frames": int(f0_count),
        "f0_log_mean": f0_mean,
        "f0_log_std": f0_std,
        "dct_mean": dct_mean,
        "dct_std": dct_std,
        "dct_count": [int(value) for value in dct_count],
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "energy_count": int(energy_count),
    }


def main() -> None:
    """Đọc tham số, tính stats, ghi JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="Manifest JSONL (nên là train, đã có durations)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="File JSON đầu ra")
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default=None, help="Cache feature (nếu có) để khỏi trích lại")
    parser.add_argument("--voiced-ratio-min", type=float, default=0.3)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stats = compute_stats(args.manifest, config, sample=args.sample, seed=args.seed,
                          workers=args.workers, cache_dir=args.cache_dir,
                          voiced_ratio_min=args.voiced_ratio_min)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {out}")
    print(f"  clips {stats['clips']} · syllables {stats['syllables']} (bỏ {stats['skipped_syllables']}) "
          f"· voiced frames {stats['voiced_frames']}")
    print(f"  log-F0: mean {stats['f0_log_mean']:.3f} · std {stats['f0_log_std']:.3f}")
    print(f"  DCT std: {[round(v, 3) for v in stats['dct_std']]}")
    print(f"  energy: mean {stats['energy_mean']:.3f} · std {stats['energy_std']:.3f}")


if __name__ == "__main__":
    main()

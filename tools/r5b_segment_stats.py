"""tools/r5b_segment_stats.py — R5b: thống kê cấp segment cho acoustic v2 (docs/ACOUSTIC_V2.md §6).

Sinh hai file:
- `artifacts/dataset_v2/r5b_stats.json` — báo cáo đầy đủ (shape theo role/thanh, voiced ratio,
  energy theo segment, phân bố gap theo boundary cho ngưỡng gate ≥ 3 frame).
- `data/codelung/f0_stats_v2.json` — bản cho model: z-score log-F0 + `s_k` (nucleus, mask `vuv`)
  + energy + **prior mềm theo thanh** (T3.2), đóng băng `f0_stats_hash` vào checkpoint.

Quy ước:
- Segment O/M/N/C lấy từ `seg_durations` (T2.1); cửa sổ frame của segment =
  `start_hữu_thanh + Σ seg[i]` với `start_hữu_thanh = frame_range[0] + Σ(d + g)`; gap nằm ngoài.
- Mask hữu thanh = `vuv > 0.5` (F4 — không dùng `f0 > 0` vì f0 đã nội suy qua vùng vô thanh).
- `s_k = c_k/√n` với `c = dct4_frames(z)`, `z` = log-F0 z-score trên chuỗi frame hữu thanh
  (2 pass: pass 1 nucleus → mean/std; pass 2 → s_k).
- Row có voiced ratio < `--voiced-ratio-min` → bỏ (đếm riêng).

Chạy:
    python tools/r5b_segment_stats.py \
        --manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
        --config configs/small_codelung.yaml --cache-dir cache/codelung_v2 \
        --sample 2500 --workers 8 \
        --report artifacts/dataset_v2/r5b_stats.json \
        --f0-out data/codelung/f0_stats_v2.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROLES = ("O", "M", "N", "C")
TONES = ["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]


def segment_windows(row: dict[str, Any]) -> list[list[tuple[str, int, int]]]:
    """Cửa sổ frame tuyệt đối `[(role, start, end)]` cho từng âm tiết (inactive bỏ trống).

    Dùng `seg_durations` + `durations`/`gaps`/`frame_range`/`lead_frames` của row.
    """
    count = len(row["durations"])
    gaps = list(row.get("gaps") or [])
    if len(gaps) == count - 1:
        gaps = gaps + [0]
    seg_durations = row.get("seg_durations")
    cursor = int(row["frame_range"][0]) if "frame_range" in row else int(row.get("lead_frames") or 0)
    windows: list[list[tuple[str, int, int]]] = []
    for index in range(count):
        segment = seg_durations[index] if seg_durations is not None else None
        inner = cursor
        syllable: list[tuple[str, int, int]] = []
        for role_index, role in enumerate(ROLES):
            length = int(segment[role_index]) if segment is not None else 0
            if length > 0:
                syllable.append((role, inner, inner + length))
            inner += length
        windows.append(syllable)
        cursor += int(row["durations"][index]) + (int(gaps[index]) if index < len(gaps) else 0)
    return windows


def segment_shape(f0: NDArray[np.float64], vuv: NDArray[np.float64], f0_mean: float,
                  f0_std: float) -> tuple[NDArray[np.float64], int]:
    """`s_k = c_k/√n` trên frame hữu thanh của một segment; trả (s, n_hữu_thanh).

    `s` có `min(4, n)` phần tử (segment ngắn chỉ tính hệ số đủ) — B1, loss mask phần thiếu.
    """
    import torch

    from vifasttts.data.features import dct4_frames

    voiced = (vuv > 0.5) & (f0 > 1e-3)
    n_voiced = int(voiced.sum())
    if n_voiced < 2:
        return np.zeros(0), n_voiced
    z = (np.log(f0[voiced]) - f0_mean) / max(f0_std, 1e-6)
    coefficients = dct4_frames(torch.as_tensor(z, dtype=torch.float64))
    return coefficients.numpy() / math.sqrt(n_voiced), n_voiced


def _worker(payload: tuple[dict[str, Any], dict[str, Any], str | None, float, str, float, float]) -> dict[str, float]:
    """Một row: trả tổng tích luỹ (chỉ scalar).

    `mode="scalar"`: log-F0 (nucleus + toàn bộ row), energy theo role, voiced ratio, scalar theo thanh.
    `mode="dct"`: `s_k` theo role + theo thanh (nucleus), số frame hữu thanh mỗi segment.
    """
    from vifasttts.data.cache import FeatureCache
    from vifasttts.data.features import extract_features

    row, config, cache_dir, voiced_ratio_min, mode, f0_mean, f0_std = payload
    cache = FeatureCache(cache_dir, config) if cache_dir else None
    extracted = cache.get(row) if cache is not None else None
    if extracted is None:
        extracted = extract_features(row["audio_path"], config)
        if cache is not None:
            cache.put(row, extracted)

    frame_f0 = extracted["f0"].numpy().astype(np.float64)
    frame_vuv = extracted["vuv"].numpy().astype(np.float64)
    frame_energy = extracted["energy"].numpy().astype(np.float64)
    frame_range = row.get("frame_range")
    start = int(frame_range[0]) if frame_range else 0
    end = int(frame_range[1]) if frame_range else frame_f0.size
    span_f0 = frame_f0[start:end]
    span_vuv = frame_vuv[start:end]
    span_voiced = (span_vuv > 0.5) & (span_f0 > 1e-3)
    voiced_ratio = float(span_voiced.mean()) if span_voiced.size else 0.0

    totals: dict[str, float] = {"clips": 1.0, "low_voiced": 0.0}
    if span_voiced.size == 0 or voiced_ratio < voiced_ratio_min:
        totals["low_voiced"] = 1.0
        return totals

    if mode == "scalar":
        values = np.log(np.maximum(span_f0[span_voiced], 1e-6))
        totals["voiced_frames_all"] = float(span_voiced.sum())
        totals["f0_sum_all"] = float(values.sum())
        totals["f0_sumsq_all"] = float((values ** 2).sum())

    for syllable, windows in zip(row["syllables"], segment_windows(row)):
        tone = syllable["abstract"]["tone"]
        for role, low, high in windows:
            low = max(low, 0)
            high = min(high, frame_f0.size)
            if high <= low:
                continue
            window_f0 = frame_f0[low:high]
            window_vuv = frame_vuv[low:high]
            voiced = (window_vuv > 0.5) & (window_f0 > 1e-3)
            ratio = float(voiced.mean())
            if mode == "scalar":
                totals[f"vr_{role}_sum"] = totals.get(f"vr_{role}_sum", 0.0) + ratio
                totals[f"vr_{role}_count"] = totals.get(f"vr_{role}_count", 0.0) + 1.0
                totals[f"vr_{role}_ge05"] = totals.get(f"vr_{role}_ge05", 0.0) + float(ratio >= 0.5)
                if role == "N":
                    values = np.log(np.maximum(window_f0[voiced], 1e-6))
                    totals["voiced_frames_nuc"] = totals.get("voiced_frames_nuc", 0.0) + float(voiced.sum())
                    totals["f0_sum_nuc"] = totals.get("f0_sum_nuc", 0.0) + float(values.sum())
                    totals["f0_sumsq_nuc"] = totals.get("f0_sumsq_nuc", 0.0) + float((values ** 2).sum())
                    totals[f"tone_{tone}_vr_sum"] = totals.get(f"tone_{tone}_vr_sum", 0.0) + ratio
                    totals[f"tone_{tone}_vr_count"] = totals.get(f"tone_{tone}_vr_count", 0.0) + 1.0
                energy = float(frame_energy[low:high].mean())
                totals["e_sum"] = totals.get("e_sum", 0.0) + energy
                totals["e_sumsq"] = totals.get("e_sumsq", 0.0) + energy * energy
                totals["e_count"] = totals.get("e_count", 0.0) + 1.0
                if role == "N":
                    totals["e_nuc_sum"] = totals.get("e_nuc_sum", 0.0) + energy
                    totals["e_nuc_sumsq"] = totals.get("e_nuc_sumsq", 0.0) + energy * energy
                    totals["e_nuc_count"] = totals.get("e_nuc_count", 0.0) + 1.0
            else:
                shape, n_voiced = segment_shape(window_f0, window_vuv, f0_mean, f0_std)
                if n_voiced < 2:
                    continue
                for k, value in enumerate(shape.tolist()):
                    for prefix in ("s_all", f"s_{role}"):
                        totals[f"{prefix}_{k}_sum"] = totals.get(f"{prefix}_{k}_sum", 0.0) + value
                        totals[f"{prefix}_{k}_sumsq"] = totals.get(f"{prefix}_{k}_sumsq", 0.0) + value * value
                        totals[f"{prefix}_{k}_count"] = totals.get(f"{prefix}_{k}_count", 0.0) + 1.0
                    if role == "N":
                        key = f"tone_{tone}_{k}"
                        totals[f"{key}_sum"] = totals.get(f"{key}_sum", 0.0) + value
                        totals[f"{key}_sumsq"] = totals.get(f"{key}_sumsq", 0.0) + value * value
                        totals[f"{key}_count"] = totals.get(f"{key}_count", 0.0) + 1.0
                totals[f"nv_{role}_sum"] = totals.get(f"nv_{role}_sum", 0.0) + n_voiced
                totals[f"nv_{role}_count"] = totals.get(f"nv_{role}_count", 0.0) + 1.0
    return totals


def pause_histogram(rows: list[dict[str, Any]], hop: int, sample_rate: int) -> dict[str, Any]:
    """Phân bố gap (ms) theo boundary của âm tiết trước — cơ sở chọn ngưỡng gate ≥ 3 frame."""
    classes: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        gaps = list(row.get("gaps") or [])
        syllables = row["syllables"]
        for index, gap in enumerate(gaps):
            if index >= len(syllables):
                break
            boundaries = syllables[index].get("boundaries") or {}
            punctuation = syllables[index].get("punctuation") or ""
            if boundaries.get("sentence_end"):
                key = "SENTENCE_ELLIPSIS" if "..." in punctuation else "SENTENCE"
            elif boundaries.get("phrase_end"):
                key = "PHRASE"
            else:
                key = "WORD"
            classes[key].append(int(gap))
    milliseconds = lambda frames: frames * hop / sample_rate * 1000.0
    summary: dict[str, Any] = {}
    for key, values in classes.items():
        array = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "n": int(array.size),
            "median_ms": round(float(np.median(milliseconds(array))), 1),
            "p90_ms": round(float(np.percentile(milliseconds(array), 90)), 1),
            "zero_frac": round(float((array == 0).mean()), 4),
            "ge3_frac": round(float((array >= 3).mean()), 4),
            "ge3_median_ms": round(float(np.median(milliseconds(array[array >= 3]))), 1) if (array >= 3).any() else None,
        }
    return summary


def duration_by_role(rows: list[dict[str, Any]], hop: int, sample_rate: int) -> dict[str, Any]:
    """Duration segment (ms) theo role × thanh (từ `seg_durations`, không cần cache)."""
    by_role: dict[str, list[float]] = defaultdict(list)
    by_tone: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        for syllable, segment in zip(row["syllables"], row.get("seg_durations") or []):
            tone = syllable["abstract"]["tone"]
            for role_index, role in enumerate(ROLES):
                frames = int(segment[role_index])
                if frames > 0:
                    milliseconds = frames * hop / sample_rate * 1000.0
                    by_role[role].append(milliseconds)
                    by_tone[(tone, role)].append(milliseconds)

    def quantile(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "p10": round(float(np.percentile(values, 10)), 1),
            "median": round(float(np.median(values)), 1),
            "p90": round(float(np.percentile(values, 90)), 1),
        }

    return {
        "by_role": {role: quantile(by_role[role]) for role in ROLES},
        "nucleus_by_tone": {tone: quantile(by_tone[(tone, "N")]) for tone in TONES},
    }


def merge_stats(totals: dict[str, float], prefix: str) -> tuple[float, float, int]:
    """(mean, std, count) từ các key `{prefix}_sum/sumsq/count`."""
    count = int(totals.get(f"{prefix}_count", 0.0))
    if count <= 0:
        return 0.0, 0.0, 0
    mean = totals.get(f"{prefix}_sum", 0.0) / count
    variance = max(0.0, totals.get(f"{prefix}_sumsq", 0.0) / count - mean * mean)
    return mean, math.sqrt(variance), count


def compute_stats(manifest: str, config: dict[str, Any], *, sample: int, seed: int, workers: int,
                  cache_dir: str | None, voiced_ratio_min: float) -> dict[str, Any]:
    """Chạy 2 pass trên mẫu row, trả report đầy đủ + bản cho model."""
    import random

    all_rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    picked = random.Random(seed).sample(all_rows, min(sample, len(all_rows)))

    def run(mode: str, f0_mean: float, f0_std: float) -> dict[str, float]:
        payloads = [(row, config, cache_dir, voiced_ratio_min, mode, f0_mean, f0_std) for row in picked]
        totals: dict[str, float] = {}
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            for index, part in enumerate(pool.map(_worker, payloads), start=1):
                for key, value in part.items():
                    totals[key] = totals.get(key, 0.0) + value
                if index % 500 == 0:
                    print(f"  [{mode}] {index}/{len(payloads)} row", flush=True)
        return totals

    scalar = run("scalar", 0.0, 1.0)
    voiced_nuc = scalar.get("voiced_frames_nuc", 0.0)
    if voiced_nuc > 0:
        f0_mean_nuc = scalar.get("f0_sum_nuc", 0.0) / voiced_nuc
        f0_std_nuc = math.sqrt(max(0.0, scalar.get("f0_sumsq_nuc", 0.0) / voiced_nuc - f0_mean_nuc ** 2))
    else:
        f0_mean_nuc, f0_std_nuc = 0.0, 0.0
    voiced_all = scalar.get("voiced_frames_all", 0.0)
    if voiced_all > 0:
        f0_mean_all = scalar.get("f0_sum_all", 0.0) / voiced_all
        f0_std_all = math.sqrt(max(0.0, scalar.get("f0_sumsq_all", 0.0) / voiced_all - f0_mean_all ** 2))
    else:
        f0_mean_all, f0_std_all = 0.0, 0.0
    dct = run("dct", f0_mean_nuc, f0_std_nuc)

    energy_mean, energy_std, energy_count = merge_stats(scalar, "e")
    energy_nuc_mean, energy_nuc_std, energy_nuc_count = merge_stats(scalar, "e_nuc")

    shape = {"mean": [], "std": [], "count": []}
    for k in range(4):
        mean, std, count = merge_stats(dct, f"s_all_{k}")
        shape["mean"].append(round(mean, 6))
        shape["std"].append(round(std, 6))
        shape["count"].append(count)
    shape_by_role: dict[str, Any] = {}
    for role in ROLES:
        role_stats = {"mean": [], "std": [], "count": []}
        for k in range(4):
            mean, std, count = merge_stats(dct, f"s_{role}_{k}")
            role_stats["mean"].append(round(mean, 6))
            role_stats["std"].append(round(std, 6))
            role_stats["count"].append(count)
        shape_by_role[role] = role_stats
    tone_prior: dict[str, Any] = {}
    for tone in TONES:
        means, counts = [], []
        for k in range(4):
            mean, _std, count = merge_stats(dct, f"tone_{tone}_{k}")
            means.append(round(mean, 6))
            counts.append(count)
        tone_prior[tone] = {"mean": means if max(counts) > 0 else [], "count": counts}

    voiced_ratio: dict[str, Any] = {}
    for role in ROLES:
        count = int(scalar.get(f"vr_{role}_count", 0.0))
        if count > 0:
            voiced_ratio[role] = {
                "mean": round(scalar.get(f"vr_{role}_sum", 0.0) / count, 4),
                "ge05_frac": round(scalar.get(f"vr_{role}_ge05", 0.0) / count, 4),
                "count": count,
            }
    tone_voiced = {tone: round(scalar.get(f"tone_{tone}_vr_sum", 0.0)
                               / max(1.0, scalar.get(f"tone_{tone}_vr_count", 1.0)), 4)
                   for tone in TONES if scalar.get(f"tone_{tone}_vr_count", 0.0) > 0}
    n_voiced = {role: round(dct.get(f"nv_{role}_sum", 0.0) / max(1.0, dct.get(f"nv_{role}_count", 1.0)), 2)
                for role in ROLES if dct.get(f"nv_{role}_count", 0.0) > 0}

    hop = int(config["mel"]["hop_length"])
    sample_rate = int(config.get("sample_rate", 24000))
    report: dict[str, Any] = {
        "version": "r5b-v2",
        "manifest": manifest,
        "sample": len(picked), "seed": seed, "rows": len(all_rows),
        "clips": int(scalar.get("clips", 0.0)),
        "low_voiced_rows": int(scalar.get("low_voiced", 0.0)),
        "f0": {
            "mean_nucleus": round(f0_mean_nuc, 6), "std_nucleus": round(f0_std_nuc, 6),
            "voiced_frames_nucleus": int(voiced_nuc),
            "mean_all": round(f0_mean_all, 6), "std_all": round(f0_std_all, 6),
            "voiced_frames_all": int(voiced_all),
        },
        "shape": shape,
        "shape_by_role": shape_by_role,
        "shape_tone_prior": tone_prior,
        "voiced_ratio_by_role": voiced_ratio,
        "voiced_ratio_nucleus_by_tone": tone_voiced,
        "n_voiced_frames_mean_by_role": n_voiced,
        "energy": {"mean": round(energy_mean, 6), "std": round(energy_std, 6), "count": energy_count,
                   "mean_nucleus": round(energy_nuc_mean, 6), "std_nucleus": round(energy_nuc_std, 6),
                   "count_nucleus": energy_nuc_count},
        "pause_by_boundary": pause_histogram(all_rows, hop, sample_rate),
        "duration_by_role": duration_by_role(all_rows, hop, sample_rate),
    }
    nucleus_shape = shape_by_role["N"]
    f0_stats = {
        "version": report["version"],
        "manifest": manifest,
        "sample": len(picked), "seed": seed,
        "f0_log_mean": round(f0_mean_nuc, 6),
        "f0_log_std": round(f0_std_nuc, 6),
        "voiced_frames": int(voiced_nuc),
        "shape_mean": nucleus_shape["mean"], "shape_std": nucleus_shape["std"],
        "shape_count": nucleus_shape["count"],
        "shape_mean_all": shape["mean"], "shape_std_all": shape["std"],
        "shape_by_role": {role: {"mean": entry["mean"], "std": entry["std"], "count": entry["count"]}
                          for role, entry in shape_by_role.items()},
        "energy_mean": round(energy_mean, 6), "energy_std": round(energy_std, 6),
        "energy_count": energy_count,
        "energy_mean_nucleus": round(energy_nuc_mean, 6), "energy_std_nucleus": round(energy_nuc_std, 6),
        "shape_tone_prior": {tone: {"mean": prior["mean"], "count": prior["count"]}
                             for tone, prior in tone_prior.items()},
        "f0_mask": "slot O: onset ∈ F0_VOICED_ONSETS ∧ ratio ≥ 0,5; N: luôn (trọng số = ratio); "
                   "M/C: ratio ≥ 0,5 — Addendum 6 (docs/ACOUSTIC_V2.md §19)",
        "notes": "s_k trên frame hữu thanh (vuv>0.5) của segment; chuẩn hoá chính = NÚCLEUS "
                 "(F4 + Addendum 6); T3.2 dùng shape_std + prior mềm.",
    }
    return {"report": report, "f0_stats": f0_stats}


def main() -> None:
    """Đọc tham số, tính stats, ghi report + f0_stats_v2 + in bảng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--sample", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--voiced-ratio-min", type=float, default=0.3)
    parser.add_argument("--report", required=True)
    parser.add_argument("--f0-out", required=True)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stats = compute_stats(args.manifest, config, sample=args.sample, seed=args.seed,
                          workers=args.workers, cache_dir=args.cache_dir,
                          voiced_ratio_min=args.voiced_ratio_min)
    report, f0_stats = stats["report"], stats["f0_stats"]
    for path, payload in ((args.report, report), (args.f0_out, f0_stats)):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Đã ghi {target}")

    print(f"\n== R5b · {report['sample']}/{report['rows']} row · low-voiced {report['low_voiced_rows']} ==")
    print(f"log-F0 nucleus: mean {report['f0']['mean_nucleus']:.3f} · std {report['f0']['std_nucleus']:.3f} "
          f"({report['f0']['voiced_frames_nucleus']} frame)")
    print(f"log-F0 all:     mean {report['f0']['mean_all']:.3f} · std {report['f0']['std_all']:.3f}")
    print("\n-- shape s_k (mọi segment) --")
    for k in range(4):
        print(f"  s{k}: mean {report['shape']['mean'][k]:+.4f} · std {report['shape']['std'][k]:.4f} "
              f"· n {report['shape']['count'][k]}")
    print("-- shape s_k theo role (mean) --")
    for role in ROLES:
        entry = report["shape_by_role"][role]
        values = " · ".join(f"s{k} {entry['mean'][k]:+.3f}" for k in range(4))
        print(f"  {role}: {values}  (n0 {entry['count'][0]})")
    print("-- voiced ratio theo role --")
    for role, entry in report["voiced_ratio_by_role"].items():
        print(f"  {role}: mean {entry['mean']:.3f} · ≥0,5 {entry['ge05_frac']:.1%} · n {entry['count']}")
    print("-- prior theo thanh (nucleus, s0..s3, n) --")
    for tone, prior in report["shape_tone_prior"].items():
        values = " · ".join(f"{value:+.3f}" for value in prior["mean"]) or "(trống)"
        print(f"  {tone:6s} {values}  (n {prior['count'][0]})")
    print("-- pause theo boundary --")
    for key, entry in report["pause_by_boundary"].items():
        print(f"  {key:18s} n {entry['n']:6d} · median {entry['median_ms']:7.1f} ms · "
              f"0 {entry['zero_frac']:5.1%} · ≥3 frame {entry['ge3_frac']:5.1%}")
    print("-- duration segment theo role (ms) --")
    for role, entry in report["duration_by_role"]["by_role"].items():
        if entry["n"]:
            print(f"  {role}: n {entry['n']:6d} · p10 {entry['p10']:5.1f} · median {entry['median']:5.1f} "
                  f"· p90 {entry['p90']:5.1f}")
    print("-- nucleus ms theo thanh --")
    for tone, entry in report["duration_by_role"]["nucleus_by_tone"].items():
        if entry["n"]:
            print(f"  {tone:6s} n {entry['n']:6d} · median {entry['median']:5.1f} ms")


if __name__ == "__main__":
    main()

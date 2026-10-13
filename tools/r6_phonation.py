"""tools/r6_phonation.py — R6: chọn kênh phonation cho v2 (docs/ACOUSTIC_V2.md §7).

So **3 ứng viên** trên N clip (âm tiết ngã/nặng = dương, ngang = âm), AUC từng kênh
(gộp toàn cục + z-score trong clip):

1. 3 kênh hiện tại `data/phonation.py`: `hnr` · `tilt` · `glottalization` (<80 Hz);
2. **D4C aperiodicity** (pyworld `cheaptrick`): trung bình 0–4 kHz trên frame hữu thanh;
3. **độ bất ổn chu kỳ** (proxy jitter): median |Δ log2 f0| giữa 2 frame kề nhau.

Cổng quyết định: `max(AUC, 1−AUC) ≥ 0,7` cho ít nhất một kênh → bật phonation ở cache v3;
không đạt → **tắt** phonation ở v2 (giữ kênh hiện tại làm chẩn đoán).

Chạy:
    python tools/r6_phonation.py --manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
        --config configs/small_codelung.yaml --files 100 --workers 8 \
        --out artifacts/dataset_v2/r6_phonation.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from r5b_segment_stats import segment_windows

CHANNELS = ("hnr", "tilt", "glot", "aper", "jit")
POSITIVE_TONES = {"NGA", "NANG"}
NEGATIVE_TONES = {"NGANG"}


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Hạng trung bình (xử lý đồng hạng) cho Mann-Whitney. O(n log n)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < values.size:
        end = index
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        ranks[order[index:end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    return ranks


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUC (Mann-Whitney U / n_pos·n_neg); 0,5 = vô dụng, 1,0 = hoàn hảo."""
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    values = np.concatenate([positive, negative])
    ranks = average_ranks(values)
    u = ranks[:positive.size].sum() - positive.size * (positive.size + 1) / 2.0
    return float(u / (positive.size * negative.size))


def _worker(payload: tuple[str, str, list[dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
    """Một clip (parent): trả record từng âm tiết ngang/ngã/nặng có nucleus hữu thanh."""
    from vifasttts.data.features import estimate_f0_pyworld, load_wav
    from vifasttts.data.phonation import estimate_aperiodicity, extract_phonation_frames

    parent, audio_path, rows, config = payload
    sample_rate = int(config.get("sample_rate", 24000))
    mel_config = config.get("mel", {})
    hop = int(mel_config.get("hop_length", 256))
    f0_config = dict(config.get("f0", {}) or {})
    f_min, f_max = float(f0_config.get("f_min", 70.0)), float(f0_config.get("f_max", 400.0))

    wav = load_wav(audio_path, sample_rate)
    f0, vuv = estimate_f0_pyworld(wav, sample_rate, hop, f_min=f_min, f_max=f_max, interpolate=True)
    r6_config = {**config, "phonation": {**dict(config.get("phonation") or {}),
                                         "enabled": True,
                                         "channels": ["hnr", "tilt", "glottalization"]}}
    phonation = extract_phonation_frames(wav, r6_config, f0=f0, vuv=vuv)
    hnr = phonation["hnr"].numpy().astype(np.float64)
    tilt = phonation["tilt"].numpy().astype(np.float64)
    glot = phonation["glottalization"].numpy().astype(np.float64)
    aper = estimate_aperiodicity(wav, sample_rate, hop, f0, vuv).numpy().astype(np.float64)
    f0_np = f0.numpy().astype(np.float64)
    vuv_np = vuv.numpy().astype(np.float64)

    # độ bất ổn chu kỳ: |Δ log2 f0| giữa 2 frame kề nhau đều hữu thanh
    voiced_frame = (vuv_np > 0.5) & (f0_np > 1e-3)
    jitter = np.full(f0_np.size, np.nan)
    pair = voiced_frame[:-1] & voiced_frame[1:]
    jitter[:-1][pair] = np.abs(np.diff(np.log2(np.maximum(f0_np, 1e-6))))[pair]

    records: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        for syllable, windows in zip(row["syllables"], segment_windows(row)):
            tone = syllable["abstract"]["tone"]
            if tone not in POSITIVE_TONES and tone not in NEGATIVE_TONES:
                continue
            key = (row["id"], syllable.get("raw", ""), tuple(windows[0]) if windows else None)
            if key in seen:
                continue
            seen.add(key)
            by_role = {role: (low, high) for role, low, high in windows}
            if "N" not in by_role:
                continue
            low, high = by_role["N"]
            low, high = max(low, 0), min(high, f0_np.size)
            if high <= low:
                continue
            index = np.arange(low, high)
            mask = voiced_frame[index]
            if int(mask.sum()) < 4:
                continue
            selected = index[mask]
            jit_values = jitter[selected[np.isfinite(jitter[selected])]]
            records.append({
                "parent": parent, "tone": tone, "n_voiced": int(mask.sum()),
                "final": bool((syllable.get("boundaries") or {}).get("phrase_end")
                              or (syllable.get("boundaries") or {}).get("sentence_end")),
                "hnr": float(hnr[selected].mean()),
                "tilt": float(tilt[selected].mean()),
                "glot": float(glot[selected].mean()),
                "aper": float(aper[selected].mean()),
                "jit": float(np.median(jit_values)) if jit_values.size else float("nan"),
            })
    return records


def _subset_auc(records: list[dict[str, Any]], channel: str, positive_tones: set[str],
                negative_tones: set[str], final_only: bool = False) -> float:
    positive = np.asarray([record[channel] for record in records
                           if record["tone"] in positive_tones and (not final_only or record["final"])],
                          dtype=np.float64)
    negative = np.asarray([record[channel] for record in records
                           if record["tone"] in negative_tones and (not final_only or record["final"])],
                          dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    return auc(positive, negative)


def _clip_z(records: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
    """Bản sao record với kênh đã z-score **trong từng clip** (khử thiên lệch clip)."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = record[channel]
        if np.isfinite(value):
            grouped[record["parent"]].append(value)
    stats = {parent: (float(np.mean(values)), float(np.std(values)) if np.std(values) > 1e-9 else 1.0)
             for parent, values in grouped.items()}
    output = []
    for record in records:
        value = record[channel]
        if not np.isfinite(value) or record["parent"] not in stats:
            continue
        mean, std = stats[record["parent"]]
        output.append({**record, channel: (value - mean) / std})
    return output


def main() -> None:
    """Chạy R6, ghi JSON + in bảng AUC + kết luận cổng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--files", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audio_of: dict[str, str] = {}
    for row in rows:
        parent = str(row.get("parent") or row["id"])
        by_parent[parent].append(row)
        audio_of[parent] = str(row["audio_path"])
    parents = sorted(by_parent)
    picked = random.Random(args.seed).sample(parents, min(args.files, len(parents)))
    print(f"R6 · {len(picked)} clip (trong {len(parents)}) · {len(rows)} row")

    payloads = [(parent, audio_of[parent], by_parent[parent], config) for parent in picked]
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, part in enumerate(pool.map(_worker, payloads), start=1):
            records.extend(part)
            if index % 20 == 0:
                print(f"  [{index}/{len(payloads)}] clip", flush=True)

    positives = sum(1 for record in records if record["tone"] in POSITIVE_TONES)
    negatives = sum(1 for record in records if record["tone"] in NEGATIVE_TONES)
    report: dict[str, Any] = {
        "manifest": args.manifest, "clips": len(picked), "seed": args.seed,
        "syllables": len(records), "positives": positives, "negatives": negatives,
        "final_frac": round(float(np.mean([record["final"] for record in records])), 3) if records else 0.0,
        "channels": {},
    }
    for channel in CHANNELS:
        pooled = _subset_auc(records, channel, POSITIVE_TONES, NEGATIVE_TONES)
        pooled_final = _subset_auc(records, channel, POSITIVE_TONES, NEGATIVE_TONES, final_only=True)
        ngang = _subset_auc(records, channel, {"NGA"}, NEGATIVE_TONES)
        nang = _subset_auc(records, channel, {"NANG"}, NEGATIVE_TONES)
        clip_records = _clip_z(records, channel)
        clip = _subset_auc(clip_records, channel, POSITIVE_TONES, NEGATIVE_TONES)
        power = max(pooled, 1 - pooled) if np.isfinite(pooled) else float("nan")
        report["channels"][channel] = {
            "auc_pooled": None if not np.isfinite(pooled) else round(pooled, 4),
            "auc_final_only": None if not np.isfinite(pooled_final) else round(pooled_final, 4),
            "auc_nga_vs_ngang": None if not np.isfinite(ngang) else round(ngang, 4),
            "auc_nang_vs_ngang": None if not np.isfinite(nang) else round(nang, 4),
            "auc_clip_z": None if not np.isfinite(clip) else round(clip, 4),
            "power": None if not np.isfinite(power) else round(power, 4),
        }
    best = max(report["channels"], key=lambda name: report["channels"][name]["power"] or 0.0)
    gate = (report["channels"][best]["power"] or 0.0) >= 0.7
    report["best_channel"] = best
    report["gate_pass"] = bool(gate)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n== AUC (dương = ngã+nặng, âm = ngang) · {positives} vs {negatives} · "
          f"final {report['final_frac']:.0%} ==")
    print(f'{"kênh":6s} {"gộp":>7s} {"final":>7s} {"ngã":>7s} {"nặng":>7s} {"clip-z":>7s} {"power":>7s}')
    for channel in CHANNELS:
        entry = report["channels"][channel]
        fmt = lambda value: "  –  " if value is None else f"{value:.3f}"
        print(f'{channel:6s} {fmt(entry["auc_pooled"]):>7s} {fmt(entry["auc_final_only"]):>7s} '
              f'{fmt(entry["auc_nga_vs_ngang"]):>7s} {fmt(entry["auc_nang_vs_ngang"]):>7s} '
              f'{fmt(entry["auc_clip_z"]):>7s} {fmt(entry["power"]):>7s}')
    print(f"\nKênh tốt nhất: **{best}** · power {report['channels'][best]['power']} · "
          f"cổng ≥ 0,7 → {'ĐẠT' if gate else 'KHÔNG ĐẠT'}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()

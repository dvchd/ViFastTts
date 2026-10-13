"""tools/r5a_stats.py — thống kê cấp âm tiết (R5a, Plan §4 / B8).

- Duration hữu thanh theo 6 thanh × coda tắc/không tắc (p10/median/p90) — kiểm tra sớm
  chất lượng durations MFA (nặng ngắn hơn thật không, coda tắc có ngắn hơn không...).
- Phân bố pause (gaps) theo boundary của âm tiết đứng trước: WORD / PHRASE / SENTENCE,
  tách riêng trường hợp "..." giữa câu (hiện gộp SENTENCE — theo dõi cho v2.1).
- Slope log-F0 theo thanh (octave/s, median), chỉ frame hữu thanh thật (`vuv > 0.5` — F4:
  f0 đã nội suy qua vùng vô thanh nên `f0 > 0` không phải mask hợp lệ).
- Đếm âm tiết `d_syl < 3` frame (mask ratio loss B5).

Chạy:
    python tools/r5a_stats.py --manifest data/codelung/train_clean_v2_dur_seg.jsonl \
        --config configs/small_codelung.yaml --cache-dir cache/codelung_v2 \
        --sample 500 --out artifacts/dataset_v2/r5a_stats.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

TONES = ["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]
STOP_CODAS = {"P", "T", "C", "CH"}


def _quantiles(values: list[float]) -> dict:
    """n / p10 / median / p90 / mean của một danh sách số."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "p10": round(float(np.percentile(array, 10)), 2),
        "median": round(float(np.median(array)), 2),
        "p90": round(float(np.percentile(array, 90)), 2),
        "mean": round(float(array.mean()), 2),
    }


def duration_stats(rows: list[dict], hop: int, sample_rate: int) -> dict:
    """Duration hữu thanh (ms) theo thanh và theo coda tắc/không tắc."""
    by_tone: dict[str, list[float]] = defaultdict(list)
    by_tone_coda: dict[tuple[str, bool], list[float]] = defaultdict(list)
    short = total = 0
    for row in rows:
        for syllable, duration in zip(row["syllables"], row["durations"]):
            milliseconds = duration * hop / sample_rate * 1000.0
            tone = syllable["abstract"]["tone"]
            stop = syllable["abstract"]["coda"] in STOP_CODAS
            by_tone[tone].append(milliseconds)
            by_tone_coda[(tone, stop)].append(milliseconds)
            short += duration < 3
            total += 1
    return {
        "by_tone": {tone: _quantiles(by_tone[tone]) for tone in TONES},
        "by_tone_coda": {
            f"{tone}_{'stop' if stop else 'open'}": _quantiles(by_tone_coda[(tone, stop)])
            for tone in TONES for stop in (False, True)
        },
        "short_lt3_frames": {"count": short, "total": total,
                             "frac": round(short / max(1, total), 5)},
    }


def pause_stats(rows: list[dict], hop: int, sample_rate: int) -> dict:
    """Gaps (ms) theo boundary của âm tiết đứng trước; tách riêng "..." giữa câu."""
    classes: dict[str, list[float]] = {"WORD": [], "PHRASE": [], "SENTENCE": [],
                                       "SENTENCE_ELLIPSIS": []}
    for row in rows:
        gaps = row.get("gaps") or []
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
            classes[key].append(gap * hop / sample_rate * 1000.0)
    summary: dict[str, dict] = {}
    for key, values in classes.items():
        stats = _quantiles(values)
        array = np.asarray(values)
        if array.size:
            stats["zero_frac"] = round(float((array == 0).mean()), 3)
            stats["ge3_frames_frac"] = round(float((array >= 3).mean()), 3)
        summary[key] = stats
    return summary


def f0_slope_stats(rows: list[dict], config: dict, cache_dir: str | None, sample: int,
                   hop: int, sample_rate: int, seed: int = 42) -> dict:
    """Slope log-F0 (octave/s) theo thanh, mask `vuv > 0.5` (F4)."""
    from vifasttts.data.cache import FeatureCache
    from vifasttts.data.features import extract_features

    picked = random.Random(seed).sample(rows, min(sample, len(rows)))
    cache = FeatureCache(cache_dir, config) if cache_dir else None
    loaded: dict[str, dict] = {}
    slopes: dict[str, list[float]] = defaultdict(list)
    voiced_ratio: dict[str, list[float]] = defaultdict(list)
    for row in picked:
        path = str(row["audio_path"])
        extracted = loaded.get(path)
        if extracted is None:
            extracted = cache.get(row) if cache else None
            if extracted is None:
                extracted = extract_features(path, config)
                if cache:
                    cache.put(row, extracted)
            loaded[path] = extracted
        frame_range = row.get("frame_range")
        position = int(frame_range[0]) if frame_range else 0
        durations = row["durations"]
        gaps = list(row.get("gaps") or [])
        spans = [duration + gap for duration, gap in
                 zip(durations, gaps + [0] * (len(durations) - len(gaps)))]
        for syllable, duration, span in zip(row["syllables"], durations, spans):
            f0 = extracted["f0"][position:position + duration].numpy()
            vuv = extracted["vuv"][position:position + duration].numpy() > 0.5
            if int(vuv.sum()) >= 4:
                values = np.log2(np.maximum(f0[vuv], 1e-6))
                axis = np.arange(values.size, dtype=np.float64)
                slope = float(np.polyfit(axis, values, 1)[0]) * (sample_rate / hop)
                tone = syllable["abstract"]["tone"]
                slopes[tone].append(slope)
                voiced_ratio[tone].append(float(vuv.mean()))
            position += span
    summary = {}
    for tone in TONES:
        stats = _quantiles(slopes[tone])
        if stats.get("n"):
            stats["negative_frac"] = round(float(np.mean(np.asarray(slopes[tone]) < 0)), 3)
            stats["voiced_ratio_mean"] = round(float(np.mean(voiced_ratio[tone])), 3)
        summary[tone] = stats
    return summary


def main() -> None:
    """Chạy thống kê, ghi JSON + in bảng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hop = int(config["mel"]["hop_length"])
    sample_rate = int(config.get("sample_rate", 24000))
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]

    report = {
        "manifest": args.manifest, "rows": len(rows),
        "durations": duration_stats(rows, hop, sample_rate),
        "pauses": pause_stats(rows, hop, sample_rate),
        "f0_slope": f0_slope_stats(rows, config, args.cache_dir, args.sample, hop, sample_rate),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"== R5a · {len(rows)} row · {out} ==")

    print(f'\n{"thanh":6s} {"n":>7s} {"p10":>7s} {"median":>8s} {"p90":>8s}  (ms)')
    for tone in TONES:
        stats = report["durations"]["by_tone"][tone]
        if stats.get("n"):
            print(f'{tone:6s} {stats["n"]:7d} {stats["p10"]:7.1f} {stats["median"]:8.1f} '
                  f'{stats["p90"]:8.1f}')

    print("\n-- theo coda tắc --")
    for key, stats in report["durations"]["by_tone_coda"].items():
        if stats.get("n"):
            print(f'{key:12s} n={stats["n"]:7d} median {stats["median"]:6.1f} ms')

    print("\n-- pause theo boundary (ms) --")
    for key, stats in report["pauses"].items():
        if stats.get("n"):
            print(f'{key:18s} n={stats["n"]:6d} zero={stats["zero_frac"]:5.1%} '
                  f'≥3fr={stats["ge3_frames_frac"]:5.1%} median {stats["median"]:6.1f}')

    print("\n-- slope log-F0 theo thanh (octave/s) --")
    for tone in TONES:
        stats = report["f0_slope"][tone]
        if stats.get("n"):
            print(f'{tone:6s} n={stats["n"]:5d} median {stats["median"]:+.3f} '
                  f'p10 {stats["p10"]:+.3f} p90 {stats["p90"]:+.3f} '
                  f'neg={stats["negative_frac"]:5.1%} voiced={stats["voiced_ratio_mean"]:.2f}')

    short = report["durations"]["short_lt3_frames"]
    print(f'\nshort d<3 frame: {short["count"]}/{short["total"]} ({short["frac"]:.2%})')


if __name__ == "__main__":
    main()

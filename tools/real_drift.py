"""tools/real_drift.py — mốc drift độ to giữa các MỆNH ĐỀ của GIỌNG THẬT (A0 v10, Việc 1).

Vì sao: thước drift cũ chỉ so với v3.2, chưa biết người thật đọc nhiều mệnh đề thì drift tự nhiên
bao nhiêu. Nếu thật cũng ~4 dB thì v3.2 đang đều quá mức (đọc phẳng), không phải chuẩn.

Cách đo (đúng công thức `premise_drift._voiced_rms_db`):
  - Gom các row cùng `parent` (hoặc cùng `id` nếu manifest không có `parent`) thành một câu.
  - Cắt audio câu tại các khoảng lặng dài > 0,64 s (frame 10 ms, ngưỡng đỉnh − 35 dB).
  - Bỏ chunk < 0,15 s hoặc mức hữu thanh < −50 dBFS; cần >= 3 chunk.
  - drift = max − min RMS hữu thanh giữa các chunk; drift_norm = drift / sqrt(số chunk).
  - median, p90 + CI 90 % bootstrap (1.000 lần).

Chạy:
  .venv/bin/python tools/real_drift.py \
      --manifest data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl \
      --manifest data/codelung/holdout_clean_v2_dur_slot4.jsonl \
      --out artifacts/bench/drift_real.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def voiced_rms_db(x: np.ndarray, sr: int) -> float:
    """RMS dB trên frame 'hữu thanh' (frame > đỉnh − 30 dB) — giống premise_drift."""
    hop = max(1, int(sr * 0.01))
    n = max(1, len(x) // hop)
    frames = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    db = 20 * np.log10(frames + 1e-12)
    keep = db > db.max() - 30.0
    if not keep.any():
        return float("nan")
    return float(10 * np.log10(np.mean(frames[keep] ** 2) + 1e-12))


def within_chunk_spike(x: np.ndarray, sr: int) -> float:
    """Chênh p95 − trung vị của RMS 10 ms trong chunk (dB) — 'một từ tự to' trong mệnh đề."""
    hop = max(1, int(sr * 0.01))
    n = max(1, len(x) // hop)
    frames = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    db = 20 * np.log10(frames + 1e-12)
    keep = db > db.max() - 30.0
    if keep.sum() < 5:
        return float("nan")
    values = db[keep]
    return float(np.percentile(values, 95) - np.median(values))


def split_chunks(x: np.ndarray, sr: int, *, gap_s: float = 0.64, thr_db: float = 35.0,
                 min_chunk_s: float = 0.15, floor_db: float = -50.0,
                 with_spike: bool = False) -> list[float] | tuple[list[float], list[float]]:
    """Cắt tại khoảng lặng > `gap_s`, trả mức RMS dB từng chunk (bỏ chunk lặng/quá ngắn)."""
    hop = max(1, int(sr * 0.01))
    n = len(x) // hop
    if n == 0:
        return []
    frames = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    db = 20 * np.log10(frames + 1e-12)
    silent = db < db.max() - thr_db
    # các đoạn lặng đủ dài -> cắt hai đầu
    cuts = [0]
    start = None
    for i, s in enumerate(silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) * 0.01 >= gap_s:
                cuts += [start, i]
            start = None
    if start is not None and (n - start) * 0.01 >= gap_s:
        cuts += [start, n]
    cuts.append(n)
    levels: list[float] = []
    spikes: list[float] = []
    for j in range(0, len(cuts) - 1, 2):
        a, b = cuts[j], cuts[j + 1]
        if (b - a) * 0.01 < min_chunk_s:
            continue
        level = voiced_rms_db(x[a * hop:b * hop], sr)
        if np.isnan(level) or level < floor_db:
            continue
        levels.append(level)
        if with_spike:
            spikes.append(within_chunk_spike(x[a * hop:b * hop], sr))
    return (levels, spikes) if with_spike else levels


def _ci(values: list[float], fn, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    stats = [fn(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n_boot)]
    return float(np.percentile(stats, 5)), float(np.percentile(stats, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--gap", type=float, default=0.64)
    parser.add_argument("--min-chunks", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=0, help="0 = không giới hạn")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sentences: dict[str, str] = {}
    for path in args.manifest:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("parent") or row.get("id"))
            sentences.setdefault(key, str(row["audio_path"]))

    drifts: list[float] = []
    spikes_all: list[float] = []
    norms: list[float] = []
    counts: list[int] = []
    per_sentence = []
    for key, audio_path in sentences.items():
        try:
            x, sr = sf.read(audio_path, dtype="float32")
        except Exception as exc:  # noqa: BLE001 - thiếu file thì bỏ qua, báo ở summary
            per_sentence.append({"id": key, "error": str(exc)})
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        levels, spikes = split_chunks(x, sr, gap_s=args.gap, with_spike=True)
        if len(levels) < args.min_chunks:
            continue
        if args.max_chunks and len(levels) > args.max_chunks:
            continue
        drift = max(levels) - min(levels)
        drifts.append(drift)
        spikes_all.extend([s for s in spikes if np.isfinite(s)])
        norms.append(drift / float(np.sqrt(len(levels))))
        counts.append(len(levels))
        per_sentence.append({"id": key, "chunks": len(levels), "drift": round(drift, 2),
                             "levels": [round(v, 1) for v in levels]})

    if not drifts:
        raise SystemExit("Không có câu nào đủ chunk — kiểm lại manifest/ngưỡng.")

    med = float(np.median(drifts))
    p90 = float(np.percentile(drifts, 90))
    med_ci = _ci(drifts, np.median)
    p90_ci = _ci(drifts, lambda a: float(np.percentile(a, 90)))
    result = {
        "source": args.manifest,
        "n": len(drifts),
        "drift_median": round(med, 2),
        "drift_median_ci90": [round(med_ci[0], 2), round(med_ci[1], 2)],
        "drift_p90": round(p90, 2),
        "drift_p90_ci90": [round(p90_ci[0], 2), round(p90_ci[1], 2)],
        "drift_norm_median": round(float(np.median(norms)), 2),
        "spike_median": round(float(np.median(spikes_all)), 2) if spikes_all else None,
        "spike_p90": round(float(np.percentile(spikes_all, 90)), 2) if spikes_all else None,
        "chunks_median": float(np.median(counts)),
        "chunks_mean": round(float(np.mean(counts)), 2),
        "chunks_hist": {str(k): int(counts.count(k)) for k in sorted(set(counts))},
        "per_sentence": per_sentence,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REAL n={result['n']} · drift median {result['drift_median']} "
          f"CI {result['drift_median_ci90']} · p90 {result['drift_p90']} CI {result['drift_p90_ci90']} "
          f"· norm {result['drift_norm_median']} · chunk median {result['chunks_median']} "
          f"· spike {result['spike_median']} (p90 {result['spike_p90']})")


if __name__ == "__main__":
    main()

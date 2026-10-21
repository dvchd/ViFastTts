"""tools/premise_drift.py — drift độ to giữa các MỆNH ĐỀ của MODEL (A0 v10, Việc 1).

Vì sao: lỗi "từ tự nhiên to lên" là lệch mức giữa các chunk (mệnh đề) khi chạy free-run chế độ
đọc đều; thước theo segment (trừ trung vị chính segment) bị mù. Tool này chạy đúng đường
production (`iter_synthesize_steady`, α=0), đo RMS dB **frame hữu thanh** của từng chunk:

  drift = max(RMS) − min(RMS) giữa các chunk của câu · drift_norm = drift / sqrt(số chunk).

Báo median, p90 + CI 90 % bootstrap (1.000 lần) để so với mốc giọng thật (`tools/real_drift.py`).

Chạy:
  .venv/bin/python tools/premise_drift.py --config configs/small_codelung_v32.yaml \
      --texts examples/drift_texts.json --checkpoint runs/adv_p2_20k/ckpt_2000.pt --tag p2_2k \
      --out artifacts/bench/drift_p2_2k.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def voiced_rms_db(x: np.ndarray, sr: int) -> float:
    """RMS dB trên các frame 'hữu thanh' (frame > đỉnh − 30 dB) — bỏ khoảng lặng hai đầu."""
    hop = max(1, int(sr * 0.01))
    n = max(1, len(x) // hop)
    frames = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    db = 20 * np.log10(frames + 1e-12)
    keep = db > db.max() - 30.0
    if not keep.any():
        return float("nan")
    return float(10 * np.log10(np.mean(frames[keep] ** 2) + 1e-12))


def _ci(values: list[float], fn, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    stats = [fn(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n_boot)]
    return float(np.percentile(stats, 5)), float(np.percentile(stats, 95))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--texts", default="examples/drift_texts.json")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--min-chunks", type=int, default=3)
    parser.add_argument("--chunk-loudness", action="store_true",
                        help="bật 6d (chuẩn hoá độ to từng chunk trên log-mel)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    sr = engine.sample_rate
    rows = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    drifts, norms, stds, counts, per_sentence = [], [], [], [], []
    for row in rows:
        levels = []
        for audio, _pause in engine.iter_synthesize_steady(
                str(row["text"]), sharpen_alpha=0.0, chunk_loudness=args.chunk_loudness):
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size < int(0.15 * sr):
                continue
            level = voiced_rms_db(audio, sr)
            if level < -50.0:  # chunk lặng/nhiễu nền — bỏ
                continue
            levels.append(level)
        if len(levels) < args.min_chunks:
            per_sentence.append({"id": row["id"], "chunks": len(levels), "skipped": True})
            continue
        drift = max(levels) - min(levels)
        drifts.append(drift)
        norms.append(drift / float(np.sqrt(len(levels))))
        stds.append(float(np.std(levels)))
        counts.append(len(levels))
        per_sentence.append({"id": row["id"], "chunks": len(levels), "drift": round(drift, 2),
                             "levels": [round(v, 1) for v in levels]})
    if not drifts:
        raise SystemExit("Không có câu nào đủ chunk — kiểm lại bộ câu / --min-chunks.")

    med = float(np.median(drifts))
    p90 = float(np.percentile(drifts, 90))
    med_ci = _ci(drifts, np.median)
    p90_ci = _ci(drifts, lambda a: float(np.percentile(a, 90)))
    result = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "n": len(drifts),
        "drift_median": round(med, 2),
        "drift_median_ci90": [round(med_ci[0], 2), round(med_ci[1], 2)],
        "drift_p90": round(p90, 2),
        "drift_p90_ci90": [round(p90_ci[0], 2), round(p90_ci[1], 2)],
        "drift_norm_median": round(float(np.median(norms)), 2),
        "std_median": round(float(np.median(stds)), 2),
        "chunks_median": float(np.median(counts)),
        "chunks_mean": round(float(np.mean(counts)), 2),
        "chunks_hist": {str(k): int(counts.count(k)) for k in sorted(set(counts))},
        "per_sentence": per_sentence,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.tag or 'ckpt':10s} median {result['drift_median']} {result['drift_median_ci90']} "
          f"· p90 {result['drift_p90']} {result['drift_p90_ci90']} · norm {result['drift_norm_median']} "
          f"· chunk median {result['chunks_median']:.0f} · n={result['n']}")


if __name__ == "__main__":
    main()

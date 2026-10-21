"""tools/premise_drift.py — drift độ to giữa các MỆNH ĐỀ (A0 lượt 7, Việc 5b).

Vì sao: lỗi "từ tự nhiên to lên" là lệch mức giữa các chunk (mệnh đề) khi chạy free-run chế độ
đọc đều; thước theo segment (trừ trung vị chính segment) bị mù. Tool này chạy đúng đường
production (`iter_synthesize_steady`, α=0), đo RMS dB **frame hữu thanh** của từng chunk:

  drift = max(RMS) − min(RMS) giữa các chunk của câu · std giữa các chunk.

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


def _voiced_rms_db(x: np.ndarray, sr: int) -> float:
    """RMS dB trên các frame 'hữu thanh' (frame > đỉnh − 30 dB) — bỏ khoảng lặng hai đầu."""
    hop = max(1, int(sr * 0.01))
    n = max(1, len(x) // hop)
    frames = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    db = 20 * np.log10(frames + 1e-12)
    keep = db > db.max() - 30.0
    if not keep.any():
        return float("nan")
    return float(10 * np.log10(np.mean(frames[keep] ** 2) + 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--texts", default="examples/drift_texts.json")
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--normalize", action="store_true", help="chuẩn hoá độ to từng chunk (5d)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    sr = engine.sample_rate
    rows = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    drifts, stds, per_sentence = [], [], []
    for row in rows:
        levels = []
        for audio, _pause in engine.iter_synthesize_steady(str(row["text"]), sharpen_alpha=0.0):
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size < int(0.15 * sr):
                continue
            if _voiced_rms_db(audio, sr) < -50.0:  # chunk lặng/nhiễu nền — bỏ
                continue
            if args.normalize and levels:
                target = 0.0
                level = _voiced_rms_db(audio, sr)
                audio = audio * (10 ** ((target - level) / 20.0))
            levels.append(_voiced_rms_db(audio, sr))
        if len(levels) < 2:
            continue
        drift = max(levels) - min(levels)
        drifts.append(drift)
        stds.append(float(np.std(levels)))
        per_sentence.append({"id": row["id"], "chunks": len(levels), "drift": round(drift, 2),
                             "levels": [round(v, 1) for v in levels]})
    result = {"tag": args.tag, "checkpoint": args.checkpoint, "n": len(drifts),
              "drift_median": round(float(np.median(drifts)), 2) if drifts else None,
              "drift_p90": round(float(np.percentile(drifts, 90)), 2) if drifts else None,
              "std_median": round(float(np.median(stds)), 2) if stds else None,
              "per_sentence": per_sentence}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.tag or 'ckpt':10s} drift median {result['drift_median']} dB · p90 {result['drift_p90']} dB "
          f"· std {result['std_median']} · n={result['n']}")


if __name__ == "__main__":
    main()

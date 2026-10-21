"""tools/measure_voiced_level.py — mức to hữu thanh của GIỌNG THẬT (A0 v10, Việc 4/6d).

Vì sao: chuẩn hoá độ to theo chunk ở runtime cần một mức MỤC TIÊU lấy từ dữ liệu thật,
không đặt tay. Tool đo trên mel của cache (cùng pipeline feature với lúc train):

  level_frame = ln( sum_b exp(2·logmel_b) )        # log tự nhiên của tổng năng lượng mel
  dB_frame    = level_frame · 10/ln(10)
  target      = trung vị dB_frame trên các frame hữu thanh (vuv ≥ 0,5) của N row

Chạy:
  .venv/bin/python tools/measure_voiced_level.py --config configs/small_codelung_v32.yaml \
      --manifest data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl --rows 400 \
      --out data/codelung/voiced_level_v2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="cache/codelung_v2")
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()


    from vifasttts.data.dataset_v2 import SlotDataset

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json"))
                          .read_text(encoding="utf-8"))
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=2)
    stats = json.loads(Path(config["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)

    levels: list[np.ndarray] = []
    per_row: list[float] = []
    for index in range(min(args.rows, len(dataset))):
        row = dataset[index]
        mel = np.asarray(row["mel"], dtype=np.float64)          # [T, 100] z-space
        vuv = np.asarray(row["vuv"], dtype=np.float64)
        if mel.ndim == 3:
            mel = mel[0]
        if vuv.ndim > 1:
            vuv = vuv.reshape(-1)
        frames = min(mel.shape[0], vuv.shape[0])
        mel, vuv = mel[:frames], vuv[:frames]
        linear = np.exp(2.0 * (mel * std + mean))
        level_db = np.log(np.sum(linear, axis=-1) + 1e-12) * (10.0 / np.log(10.0))
        voiced = level_db[vuv >= 0.5]
        if voiced.size < 3:
            continue
        levels.append(voiced)
        per_row.append(float(np.median(voiced)))

    if not levels:
        raise SystemExit("Không có frame hữu thanh — kiểm manifest/cache.")
    pooled = np.concatenate(levels)
    result = {
        "target_db": round(float(np.median(pooled)), 3),
        "n_rows": len(per_row),
        "n_frames": int(pooled.size),
        "row_median_median": round(float(np.median(per_row)), 3),
        "row_median_p10": round(float(np.percentile(per_row, 10)), 3),
        "row_median_p90": round(float(np.percentile(per_row, 90)), 3),
        "definition": "ln(sum_b exp(2*(z*std+mean))) * 10/ln(10), trung vị trên frame vuv>=0.5",
        "manifest": args.manifest,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"target {result['target_db']} dB · {result['n_rows']} row · {result['n_frames']} frame "
          f"· row median p10-p90 {result['row_median_p10']}…{result['row_median_p90']}")


if __name__ == "__main__":
    main()

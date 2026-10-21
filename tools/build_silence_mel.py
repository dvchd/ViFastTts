"""tools/build_silence_mel.py — mel lặng chuẩn của corpus (A0 v15, Việc 11a).

Vì sao: frame thuộc gap (pause do runtime chèn) nên là lặng thật, không phải embedding học của
model. Lấy trung vị từng bin trên các frame vuv < 0,5 và năng lượng thấp của corpus.

Chạy:
  .venv/bin/python tools/build_silence_mel.py --config configs/small_codelung_v32.yaml \
      --manifest data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl --rows 300 \
      --out data/codelung/silence_mel_v2.json
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
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from vifasttts.data.dataset_v2 import SlotDataset

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json"))
                          .read_text(encoding="utf-8"))
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=2)

    frames = []
    for index in range(min(args.rows, len(dataset))):
        item = dataset[index]
        mel = np.asarray(item["mel"], dtype=np.float64)
        vuv = np.asarray(item["vuv"], dtype=np.float64).reshape(-1)
        energy = np.asarray(item["energy"], dtype=np.float64).reshape(-1)
        n = min(len(mel), len(vuv), len(energy))
        silent = (vuv[:n] < 0.5) & (energy[:n] <= np.percentile(energy[:n], 25))
        if silent.sum() >= 3:
            frames.append(mel[:n][silent])
    if not frames:
        raise SystemExit("Không có frame lặng — kiểm cache.")
    pooled = np.concatenate(frames, axis=0)
    mel = np.median(pooled, axis=0)
    result = {"mel": [round(float(v), 5) for v in mel], "n_frames": int(pooled.shape[0]),
              "source": args.manifest}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"{result['n_frames']} frame lặng · mel 100 bin · ghi {out}")


if __name__ == "__main__":
    main()

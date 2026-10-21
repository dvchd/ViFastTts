"""tools/build_duration_prior.py — prior độ dài từng âm tiết từ corpus (A0 v12, bổ sung cho 8b).

Vì sao: "rồi" bị model dự đoán 42 frame (448 ms) trong khi corpus: median 17, p90 30 frame.
Kẹp duration dự đoán vào [p10, p90] của chính âm tiết (chỉ khi n ≥ 20) là cách tất định, rẻ,
chỉ sửa các giá trị ngoại lai — không đổi các âm tiết đã đúng.

Chạy:
  .venv/bin/python tools/build_duration_prior.py \
      --manifest data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl \
      --out data/codelung/duration_prior_v2.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-count", type=int, default=20)
    args = parser.parse_args()

    buckets: dict[str, list[int]] = defaultdict(list)
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for syllable, duration in zip(row.get("syllables", []), row.get("durations", [])):
            word = str(syllable.get("normalized") or syllable.get("raw") or "").lower()
            if word:
                buckets[word].append(int(duration))

    prior = {}
    for word, values in buckets.items():
        if len(values) < args.min_count:
            continue
        array = np.asarray(values)
        prior[word] = {"median": float(np.median(array)),
                       "p10": float(np.percentile(array, 10)),
                       "p90": float(np.percentile(array, 90)),
                       "n": int(array.size)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prior, ensure_ascii=False), encoding="utf-8")
    print(f"{len(prior)} âm tiết (n ≥ {args.min_count}) · 'rồi' = {prior.get('rồi')} · ghi {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""tools/v3_denoise_clips.py — Data v3 · B1: DFN3 denoise **từng clip** (an toàn RAM, nhanh).

Chạy bằng venv DeepFilterNet (Python 3.9 + torch 2.8 — cài riêng):
    ~/dvchd/.dfn-venv/bin/python tools/v3_denoise_clips.py \
        --manifests data/codelung/train_clean_v2_dur.jsonl,data/codelung/val_clean_v2_dur.jsonl,data/codelung/holdout_clean_v2_dur.jsonl \
        --out-dir data/codelung/v3/denoised --workers 4

- Mỗi clip 24k → upsample 48k → DFN3 → downsample 24k, ghi PCM_16 (nguồn vốn 24k).
- Idempotent: file đã có ⇒ bỏ qua. Multiprocessing: mỗi worker load model 1 lần.
- Vì sao clip-level (không video-level): video 1,6h × 4 worker ngốn swap 16 GB (19/10); clip ≤15s
  giữ RAM thấp, và clip chính là đơn vị của pipeline v3.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_MODEL = None
_STATE = None


def _init_model() -> None:
    global _MODEL, _STATE
    import torch

    torch.set_num_threads(1)  # chống bùng thread khi nhiều worker (19/10: từng gây treo máy)
    from df.enhance import init_df

    _MODEL, _STATE, _ = init_df()


def _process(payload: tuple[str, str]) -> str:
    src, dst = payload
    if Path(dst).exists():
        return "skip"
    import soundfile as sf
    import torch
    import torchaudio
    from df.enhance import enhance

    x, sr = sf.read(src, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    xt = torch.from_numpy(x).unsqueeze(0)
    target = _STATE.sr()
    if sr != target:
        xt = torchaudio.functional.resample(xt, sr, target)
    with torch.no_grad():
        y = enhance(_MODEL, _STATE, xt)
    if sr != target:
        y = torchaudio.functional.resample(y, target, sr)
    sf.write(dst, y.squeeze(0).numpy().astype("float32"), sr, subtype="PCM_16")
    return "done"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", required=True, help="danh sách manifest cách nhau dấu phẩy")
    parser.add_argument("--out-dir", default="data/codelung/v3/denoised")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=300, help="số file mỗi lô pool (giải phóng RAM định kỳ)")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, str]] = []
    seen: set[str] = set()
    for manifest in args.manifests.split(","):
        for line in (ROOT / manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            src = ROOT / str(row["audio_path"])
            name = Path(str(row["audio_path"])).name
            if src.exists() and name not in seen:
                seen.add(name)
                tasks.append((str(src), str(out_dir / name)))
    print(f"[v3-dn] {len(tasks)} clip · {args.workers} workers · pool theo lô {args.chunk}", flush=True)
    done = 0
    for start in range(0, len(tasks), args.chunk):  # pool theo lô: giải phóng RAM định kỳ (chống treo máy)
        batch = tasks[start:start + args.chunk]
        with Pool(args.workers, initializer=_init_model) as pool:
            for _ in pool.imap_unordered(_process, batch, chunksize=4):
                done += 1
        print(f"[v3-dn] {done}/{len(tasks)}", flush=True)
    print("[v3-dn] XONG", flush=True)


if __name__ == "__main__":
    sys.exit(main())

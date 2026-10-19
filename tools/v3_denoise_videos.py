#!/usr/bin/env python
"""tools/v3_denoise_videos.py — Data v3 · B1: DFN3 denoise **đồng nhất** 25 video thô.

Chạy bằng venv DeepFilterNet (Python 3.9, torch 2.8 — cài riêng, xem docs/ACOUSTIC_V2 §49):
    ~/dvchd/.dfn-venv/bin/python tools/v3_denoise_videos.py \
        --raw-dir data/codelung/raw --out-dir data/codelung/v3/denoised --workers 6

- Chỉ 25 video corpus (12 val + 13 train theo `data/codelung/split_lock.json`); bỏ 6 video khách.
- DFN cần 48 kHz → upsample 24k→48k, denoise, downsample về 24k (nguồn vốn 24k ⇒ không mất gì).
- Idempotent: file đã có trong out-dir thì bỏ qua (chạy lại an toàn).
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


def _init_model():
    global _MODEL, _STATE
    from df.enhance import init_df

    _MODEL, _STATE, _ = init_df()


def _process(payload: tuple[str, str]) -> str:
    """Denoise 1 video → wav 24k. Trả tên video khi xong."""
    src, dst = payload
    import soundfile as sf
    import torch
    import torchaudio
    from df.enhance import enhance

    if Path(dst).exists():
        return f"skip {Path(src).stem}"
    x, sr = sf.read(src, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    xt = torch.from_numpy(x).unsqueeze(0)
    target_sr = _STATE.sr()  # 48000
    if sr != target_sr:
        xt = torchaudio.functional.resample(xt, sr, target_sr)
    with torch.no_grad():
        y = enhance(_MODEL, _STATE, xt)
    if target_sr != sr:
        y = torchaudio.functional.resample(y, target_sr, sr)
    sf.write(dst, y.squeeze(0).numpy().astype("float32"), sr, subtype="PCM_16")
    return f"done {Path(src).stem}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/codelung/raw")
    parser.add_argument("--out-dir", default="data/codelung/v3/denoised")
    parser.add_argument("--split-lock", default="data/codelung/split_lock.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--videos", default=None, help="danh sách video cách nhau dấu phẩy (mặc định: 25 video corpus)")
    args = parser.parse_args()

    raw_dir = ROOT / args.raw_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.videos:
        videos = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        lock = json.loads((ROOT / args.split_lock).read_text(encoding="utf-8"))
        videos = list(lock["train_videos"]) + list(lock["val_videos"])
    tasks = [(str(raw_dir / f"{v}.wav"), str(out_dir / f"{v}.wav")) for v in videos
             if (raw_dir / f"{v}.wav").exists()]
    print(f"[v3-denoise] {len(tasks)} video · {args.workers} workers", flush=True)
    with Pool(args.workers, initializer=_init_model) as pool:
        for i, msg in enumerate(pool.imap_unordered(_process, tasks), 1):
            print(f"[v3-denoise] {i}/{len(tasks)} {msg}", flush=True)
    print("[v3-denoise] XONG", flush=True)


if __name__ == "__main__":
    sys.exit(main())

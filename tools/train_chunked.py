"""Train theo từng khối, tự resume — chống phình bộ nhớ Metal (MPS) khi train dài.

MPS giữ cache MPSGraph/command buffer theo hình dạng tensor; train liên tục hàng
nghìn bước làm "other allocations" leo tới hàng chục GB rồi OOM. Runner này chạy
từng khối (mặc định 5000 bước): mỗi khối là một tiến trình mới (giải phóng sạch
bộ nhớ Metal), resume từ `last.pt` nhờ tính năng `--resume`.

Chạy:
    python tools/train_chunked.py --config configs/small_codelung.yaml \
        --train-manifest data/codelung/train.jsonl \
        --val-manifest data/codelung/val.jsonl \
        --output runs/small_codelung --total-steps 60000 --chunk 5000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def read_step(checkpoint: Path) -> int:
    """Đọc số bước hiện tại từ checkpoint (0 nếu chưa có)."""
    if not checkpoint.exists():
        return 0
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(state.get("step", 0))


def main() -> None:
    """Vòng lặp khối: train → lưu → thoát → resume cho tới đủ tổng số bước."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=5000, help="Số bước mỗi khối")
    parser.add_argument("--max-failures", type=int, default=5, help="Số lần lỗi liên tiếp cho một mốc bước")
    args = parser.parse_args()

    out = Path(args.output)
    failures = 0
    while True:
        step = read_step(out / "last.pt")
        if step >= args.total_steps:
            print(f"[chunked] xong: {step}/{args.total_steps} bước", flush=True)
            break
        target = min(step + args.chunk, args.total_steps)
        command = [
            sys.executable, "-m", "vifasttts.cli", "train",
            "--config", args.config,
            "--train-manifest", args.train_manifest,
            "--val-manifest", args.val_manifest,
            "--output", str(out),
            "--max-steps", str(target),
        ]
        if step > 0:
            command += ["--resume", str(out / "last.pt")]
        print(f"[chunked] {step} → {target} bước (khối {args.chunk})", flush=True)
        code = subprocess.run(command, check=False).returncode
        new_step = read_step(out / "last.pt")
        if code == 0:
            failures = 0
            continue
        if new_step > step:
            failures = 0  # có tiến độ rồi crash → lần sau resume tiếp
        else:
            failures += 1
        print(f"[chunked] lỗi rc={code} tại bước {new_step}; thử lại sau 30s "
              f"({failures}/{args.max_failures})", flush=True)
        if failures >= args.max_failures:
            raise SystemExit(f"[chunked] dừng: lỗi {failures} lần liên tiếp tại bước {new_step}")
        time.sleep(30)


if __name__ == "__main__":
    main()

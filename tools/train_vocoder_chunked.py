"""Train vocoder theo từng khối, tự resume — chống phình bộ nhớ Metal (MPS) khi chạy dài.

Cùng ý tưởng với `tools/train_chunked.py` (dành cho acoustic): mỗi khối là một
tiến trình mới, giải phóng sạch bộ nhớ MPS; resume từ `last.pt` nhờ `--resume`.
Phù hợp GAN vocoder chạy 100k+ bước trên MPS (xem docs/REVISION_PLAN_V2.md §10).

Chạy:
    python tools/train_vocoder_chunked.py --config configs/small_codelung.yaml \
        --manifest data/codelung/train.jsonl \
        --output runs/vocoder_codelung_v2 --total-steps 100000 --chunk 5000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _mps_env() -> dict[str, str]:
    """E1: đặt watermark MPS cho tiến trình con (tránh leo 40GB khi chạy dài)."""
    env = dict(os.environ)
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
    env.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")
    return env


def _base_seed(config_path: str) -> int:
    """Seed gốc từ config (mặc định 42) — dùng để đổi batch khi rollback NaN."""
    try:
        import yaml

        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        return int(payload.get("seed", 42))
    except Exception:  # noqa: BLE001 - thiếu/không đọc được config thì dùng mặc định
        return 42


def read_step(checkpoint: Path) -> int:
    """Đọc số bước hiện tại từ checkpoint (0 nếu chưa có)."""
    if not checkpoint.exists():
        return 0
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(state.get("step", 0))


def _pick_checkpoint(out: Path) -> Path | None:
    """Chọn checkpoint **sạch** để resume: last.pt → prev.pt (rev.6 §1a).

    Bỏ qua checkpoint có `clean: false` hoặc `last_loss` không hữu hạn (nghi nhiễm NaN).
    """
    import math

    import torch

    for name in ("last.pt", "prev.pt"):
        path = out / name
        if not path.exists():
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001 - file hỏng thì thử bản kế tiếp
            print(f"[chunked-vc] bỏ qua {path.name}: {type(error).__name__} — thử checkpoint kế tiếp", flush=True)
            continue
        if state.get("clean") is False:
            continue
        loss = state.get("last_loss")
        if loss is not None and not math.isfinite(float(loss)):
            continue
        return path
    return None


def main() -> None:
    """Vòng lặp khối: train → lưu → thoát → resume cho tới đủ tổng số bước."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=5000, help="Số bước mỗi khối")
    parser.add_argument("--max-failures", type=int, default=5,
                        help="Số lần lỗi liên tiếp cho một mốc bước")
    args = parser.parse_args()

    out = Path(args.output)
    base_seed = _base_seed(args.config)
    failures = 0
    while True:
        resume_path = _pick_checkpoint(out)
        step = read_step(resume_path) if resume_path is not None else 0
        if step >= args.total_steps:
            print(f"[chunked-vc] xong: {step}/{args.total_steps} bước", flush=True)
            break
        target = min(step + args.chunk, args.total_steps)
        command = [
            sys.executable, "-m", "vifasttts.cli", "train-vocoder",
            "--config", args.config,
            "--manifest", args.manifest,
            "--output", str(out),
            "--max-steps", str(target),
        ]
        if resume_path is not None:
            # rev.6 §1a: chỉ resume từ checkpoint đã kiểm sạch (last.pt → prev.pt).
            command += ["--resume", str(resume_path)]
        if failures:
            # E4: rollback → đổi seed (tránh gặp lại đúng NaN cũ)...
            command += ["--seed", str(base_seed + failures * 1000)]
        if failures >= 2:
            # ...và hạ LR 0.5× nếu NaN lặp lại từ 2 lần trở lên.
            command += ["--lr-scale", str(0.5 ** (failures - 1))]
        print(f"[chunked-vc] {step} → {target} bước (khối {args.chunk}"
              f"{f', từ {resume_path.name}' if resume_path else ''}"
              f"{f', lần lỗi {failures}' if failures else ''})", flush=True)
        code = subprocess.run(command, check=False, env=_mps_env()).returncode
        new_resume = _pick_checkpoint(out)
        new_step = read_step(new_resume) if new_resume is not None else 0
        if code == 0:
            failures = 0
            continue
        if new_step > step:
            failures = 0  # có tiến độ rồi crash → lần sau resume tiếp
        else:
            failures += 1
        print(f"[chunked-vc] lỗi rc={code} tại bước {new_step}; thử lại sau 30s "
              f"({failures}/{args.max_failures})", flush=True)
        if failures >= args.max_failures:
            raise SystemExit(f"[chunked-vc] dừng: lỗi {failures} lần liên tiếp tại bước {new_step}")
        time.sleep(30)


if __name__ == "__main__":
    main()

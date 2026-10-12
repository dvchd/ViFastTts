"""Tính mean/std toàn cục per-bin cho mel log-magnitude → JSON cho `mel.normalize_path`.

Chạy:
    python tools/compute_mel_stats.py --manifest data/codelung/train.jsonl \
        --config configs/small_codelung.yaml --out data/codelung/mel_stats.json

Lưu ý (R7-1, xem docs/VOCODER_VOCOS.md §2):
- Với `log_floor_mode: clamp` (biểu diễn Vocos), các frame/bin rất yếu dồn về đúng
  `log(log_floor)` (−16,12 nats với 1e-7). Công cụ **đo tỉ lệ bị kẹp theo từng bin**
  và cảnh báo nếu bin cao vượt 1%; lúc đó nên tính stats chỉ trên frame có tiếng:
  `--min-log-energy -6` (log-energy trung bình phổ > −6).
- Vocoder không bị ảnh hưởng: nó nhận mel đã khử chuẩn hoá (affine ngược).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any


def _stats_worker(payload: tuple[str, dict[str, Any], float | None]) -> tuple[Any, Any, int, Any]:
    """Worker: trả (sum, sum², số frame, số phần tử bị kẹp ở floor) cho một clip."""
    audio_path, mel_config, min_log_energy = payload

    from vifasttts.data.features import extract_mel_energy, load_wav

    wav = load_wav(audio_path, int(mel_config["sample_rate"]))
    mel, energy = extract_mel_energy(wav, mel_config)
    if min_log_energy is not None:
        mel = mel[energy > float(min_log_energy)]
    values = mel.double()
    floor = float(mel_config.get("log_floor", 1e-5))
    clamped = (values <= math.log(floor) + 1e-6).sum(dim=0)
    return values.sum(dim=0).numpy(), (values * values).sum(dim=0).numpy(), int(values.size(0)), clamped.numpy()


def main() -> None:
    """Đọc tham số, tính thống kê, ghi JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Manifest JSONL (nên là train)")
    parser.add_argument("--config", required=True, help="Config YAML để lấy mục mel/sample_rate")
    parser.add_argument("--out", required=True, help="Đường dẫn JSON đầu ra")
    parser.add_argument("--sample", type=int, default=400, help="Số clip lấy mẫu (mặc định 400)")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Số tiến trình")
    parser.add_argument("--seed", type=int, default=42, help="Seed chọn mẫu")
    parser.add_argument("--min-log-energy", type=float, default=None,
                        help="Chỉ tính stats trên frame có log-energy > ngưỡng này (ví dụ -6)")
    args = parser.parse_args()

    import numpy as np
    import yaml

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("Manifest rỗng")
    rng = random.Random(args.seed)
    picked = rng.sample(rows, min(args.sample, len(rows)))

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mel_config: dict[str, Any] = dict(config.get("mel", {}))
    mel_config.pop("normalize_path", None)  # tránh phụ thuộc chính file đang ghi
    mel_config["log_magnitude"] = True
    mel_config.setdefault("sample_rate", int(config.get("sample_rate", 24000)))
    n_mels = int(mel_config["n_mels"])

    payloads = [(str(row["audio_path"]), mel_config, args.min_log_energy) for row in picked]
    sums = np.zeros(n_mels, dtype=np.float64)
    sumsq = np.zeros(n_mels, dtype=np.float64)
    clamp_counts = np.zeros(n_mels, dtype=np.float64)
    frames = 0
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, (part_sum, part_sumsq, part_frames, part_clamped) in enumerate(pool.map(_stats_worker, payloads), start=1):
            sums += part_sum
            sumsq += part_sumsq
            clamp_counts += part_clamped
            frames += part_frames
            if index % 50 == 0 or index == len(payloads):
                print(f"  {index}/{len(payloads)} clip · {frames} frame", flush=True)

    mean = sums / max(1, frames)
    variance = np.clip(sumsq / max(1, frames) - mean * mean, 0.0, None)
    std = np.sqrt(variance).clip(min=1e-4)

    clamp_ratio = clamp_counts / max(1, frames)
    high_ratio = float(clamp_ratio[n_mels // 2:].max()) if n_mels > 1 else 0.0
    print(f"Tỉ lệ bị kẹp ở floor: trung bình {clamp_ratio.mean():.3%} · max {clamp_ratio.max():.3%} "
          f"(bin {int(clamp_ratio.argmax())}) · max nửa trên {high_ratio:.3%}")
    if high_ratio > 0.01:
        print("  ⚠️  Bin cao bị kẹp >1% — nên tính lại stats với `--min-log-energy -6` "
              "(chỉ frame có tiếng) để tránh méo mean/std, xem docs/VOCODER_VOCOS.md §2.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n_mels": n_mels,
                "clips": len(picked),
                "frames": frames,
                "norm": mel_config.get("norm", "slaney"),
                "log_magnitude": True,
                "log_floor": float(mel_config.get("log_floor", 1e-5)),
                "log_floor_mode": mel_config.get("log_floor_mode", "add"),
                "min_log_energy": args.min_log_energy,
                "clamp_ratio_mean": float(clamp_ratio.mean()),
                "clamp_ratio_max_high": high_ratio,
                "mean": [float(v) for v in mean],
                "std": [float(v) for v in std],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"Đã ghi {out} ({len(picked)} clip · {frames} frame)")
    print("Thêm vào config (mục mel):")
    print("  norm: none" if str(mel_config.get("norm", "slaney")) == "none" else "")
    print("  log_magnitude: true")
    print(f"  log_floor: {float(mel_config.get('log_floor', 1e-5)):g} · log_floor_mode: {mel_config.get('log_floor_mode', 'add')}")
    print(f"  normalize_path: {out}")


if __name__ == "__main__":
    main()

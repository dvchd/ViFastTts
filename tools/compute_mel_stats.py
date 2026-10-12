"""Tính mean/std toàn cục per-bin cho mel log-magnitude → JSON cho `mel.normalize_path`.

Chạy:
    python tools/compute_mel_stats.py --manifest data/codelung/train.jsonl \
        --config configs/small_codelung.yaml --out data/codelung/mel_stats.json

Dùng một mẫu ngẫu nhiên (`--sample`, mặc định 400 clip) — đủ ổn định cho thống kê
per-bin mà không cần quét toàn bộ 51 giờ. Xem `docs/REVISION_PLAN_V2.md` §2.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


def _stats_worker(payload: tuple[str, dict[str, Any]]) -> tuple[Any, Any, int]:
    """Worker: trả (sum, sum², số frame) của mel log-magnitude cho một clip."""
    audio_path, mel_config = payload


    from vifasttts.data.features import extract_mel_energy, load_wav

    wav = load_wav(audio_path, int(mel_config["sample_rate"]))
    mel, _ = extract_mel_energy(wav, mel_config)
    values = mel.double()
    return values.sum(dim=0).numpy(), (values * values).sum(dim=0).numpy(), int(values.size(0))


def main() -> None:
    """Đọc tham số, tính thống kê, ghi JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Manifest JSONL (nên là train)")
    parser.add_argument("--config", required=True, help="Config YAML để lấy mục mel/sample_rate")
    parser.add_argument("--out", required=True, help="Đường dẫn JSON đầu ra")
    parser.add_argument("--sample", type=int, default=400, help="Số clip lấy mẫu (mặc định 400)")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Số tiến trình")
    parser.add_argument("--seed", type=int, default=42, help="Seed chọn mẫu")
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

    payloads = [(str(row["audio_path"]), mel_config) for row in picked]
    sums = np.zeros(n_mels, dtype=np.float64)
    sumsq = np.zeros(n_mels, dtype=np.float64)
    frames = 0
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, (part_sum, part_sumsq, part_frames) in enumerate(pool.map(_stats_worker, payloads), start=1):
            sums += part_sum
            sumsq += part_sumsq
            frames += part_frames
            if index % 50 == 0 or index == len(payloads):
                print(f"  {index}/{len(payloads)} clip · {frames} frame", flush=True)

    mean = sums / max(1, frames)
    variance = np.clip(sumsq / max(1, frames) - mean * mean, 0.0, None)
    std = np.sqrt(variance).clip(min=1e-4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n_mels": n_mels,
                "clips": len(picked),
                "frames": frames,
                "log_magnitude": True,
                "log_floor": float(mel_config.get("log_floor", 1e-5)),
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
    print("  log_magnitude: true")
    print(f"  normalize_path: {out}")


if __name__ == "__main__":
    main()

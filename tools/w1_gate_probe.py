"""tools/w1_gate_probe.py — v18.28 (W1): kiểm giả thuyết **noise gate** trong audio gốc.

Giả thuyết: chỗ lặng của bản thu lúc còn nền nhiễu, lúc về 0 hẳn (gate/khử ồn). Model không
đoán được cổng đóng/mở ⇒ L1 lấy mức trung gian ⇒ nền nhiễu đều (+12,6 dB).

Đo trên 300 row val + 2.000 row train (seed cố định):
1. Histogram RMS dBFS (bước 5 dB) và histogram mel-z trung bình mỗi frame — tìm hai đỉnh.
2. Tỉ lệ frame "im kỹ thuật số" (RMS < −90 dBFS) theo prefix video nguồn.
3. 5 khoảng lặng ≥ 300 ms mỗi loại (im kỹ thuật số / nền nhiễu) — in đường RMS 10 ms.
4. Mel P2 (TF) tại frame im kỹ thuật số vs frame nền nhiễu: model nằm giữa hai nhóm?

JSON ghi **trước** khi in (`artifacts/bench/w1_gate.json`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

HOP = 256
FRAME_MS = 1000.0 * HOP / 24000.0
DIGITAL_DB = -90.0
FLOOR_DB = -50.0
MIN_GAP_FRAMES = int(300 / FRAME_MS)  # 300 ms


def _read(row: dict, frames: int) -> np.ndarray:
    start = int(row["frame_range"][0])
    with sf.SoundFile(row["audio_path"]) as handle:
        handle.seek(start * HOP)
        audio = handle.read(frames * HOP, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32)


def _db(audio: np.ndarray, frames: int) -> np.ndarray:
    """RMS dBFS theo block 256 mẫu (10,67 ms) — đúng hop của model."""
    blocks = audio[: frames * HOP].reshape(frames, HOP).astype(np.float64)
    rms = np.sqrt(np.maximum((blocks**2).mean(axis=1), 1e-24))
    return 20.0 * np.log10(rms)


def _gaps(db: np.ndarray, limit: int = 0) -> list[tuple[int, int]]:
    """Các đoạn liên tục < −50 dBFS dài ≥ 300 ms."""
    quiet = db < FLOOR_DB
    spans = []
    start = None
    for index, flag in enumerate(quiet):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= MIN_GAP_FRAMES:
                spans.append((start, index))
            start = None
    if start is not None and len(quiet) - start >= MIN_GAP_FRAMES:
        spans.append((start, len(quiet)))
    if limit:
        spans = spans[:limit]
    return spans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small_codelung_v32.yaml")
    parser.add_argument("--train-manifest", default="data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--val-gta", default="/private/tmp/opencode/val300_gta")
    parser.add_argument("--val-manifest", default="/private/tmp/opencode/val300_manifest.jsonl")
    parser.add_argument("--train-gta", default="cache/gta_p2")
    parser.add_argument("--train-rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="artifacts/bench/w1_gate.json")
    args = parser.parse_args()

    from vifasttts.data.features import extract_mel_energy_batched

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    z_cfg = dict(config["mel"])

    train_rows = [json.loads(x) for x in Path(args.train_manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(train_rows), size=min(args.train_rows, len(train_rows)), replace=False)
    train_rows = [train_rows[i] for i in sorted(pick)]
    val_rows = [json.loads(x) for x in Path(args.val_manifest).read_text(encoding="utf-8").splitlines() if x.strip()]

    hist_rms = np.zeros(20, dtype=np.int64)  # −100..0 dB, bước 5
    hist_mel = np.zeros(40, dtype=np.int64)  # mean mel z −8..2, bước 0,25
    per_prefix = defaultdict(lambda: [0, 0])  # prefix → [digital, total]
    group_sum = {"digital": [0.0, 0], "noise": [0.0, 0], "voiced": [0.0, 0]}  # real mel mean
    model_sum = {"digital": [0.0, 0], "noise": [0.0, 0], "voiced": [0.0, 0]}  # model mel mean
    totals = {"frames": 0, "digital": 0, "noise": 0, "voiced": 0}
    gap_examples: dict[str, list] = {"digital": [], "noise": []}
    gap_scan = {"digital": 0, "noise": 0}

    with torch.no_grad():
        for split, rows, cache in (("train", train_rows, Path(args.train_gta)), ("val", val_rows, Path(args.val_gta))):
            for row in rows:
                payload_path = cache / f"{row['id']}.pt"
                if not payload_path.exists():
                    continue
                payload = torch.load(payload_path, map_location="cpu", weights_only=False)
                frames = int(payload["frames"])
                audio = _read(row, frames)
                if audio.size < frames * HOP:
                    continue
                db = _db(audio, frames)
                wav = torch.from_numpy(audio).unsqueeze(0)
                mel_z, _ = extract_mel_energy_batched(wav, z_cfg)
                mel_z = mel_z[0, :frames]
                real_mean = mel_z.mean(dim=1).numpy()
                # 1. histogram
                idx = np.clip(((db + 100.0) // 5.0).astype(int), 0, 19)
                hist_rms += np.bincount(idx, minlength=20)
                midx = np.clip(((real_mean + 8.0) / 0.25).astype(int), 0, 39)
                hist_mel += np.bincount(midx, minlength=40)
                # 2/4. phân nhóm frame
                digital = db < DIGITAL_DB
                noise = (db >= DIGITAL_DB) & (db < FLOOR_DB)
                voiced = db >= FLOOR_DB
                totals["frames"] += frames
                totals["digital"] += int(digital.sum())
                totals["noise"] += int(noise.sum())
                totals["voiced"] += int(voiced.sum())
                prefix = str(row["id"]).split("_")[0]
                per_prefix[prefix][0] += int(digital.sum())
                per_prefix[prefix][1] += frames
                for name, mask in (("digital", digital), ("noise", noise), ("voiced", voiced)):
                    if mask.any():
                        group_sum[name][0] += float(real_mean[mask].sum())
                        group_sum[name][1] += int(mask.sum())
                        if split == "train":
                            model_mean = payload["mel"].mean(dim=1).numpy()
                            model_sum[name][0] += float(model_mean[mask].sum())
                            model_sum[name][1] += int(mask.sum())
                # 3. khoảng lặng
                for start, end in _gaps(db):
                    inner = db[start:end]
                    kind = "digital" if float(np.median(inner)) < DIGITAL_DB else "noise"
                    gap_scan[kind] += 1
                    if len(gap_examples[kind]) >= 5:
                        continue
                    lo, hi = max(0, start - 20), min(frames, end + 20)
                    gap_examples[kind].append(
                        {
                            "id": str(row["id"]),
                            "split": split,
                            "gap_ms": round((end - start) * FRAME_MS, 1),
                            "before_db": round(float(db[max(0, start - 10):start].mean()), 1),
                            "inside_first5": [round(float(v), 1) for v in db[start:start + 5]],
                            "inside_min_db": round(float(inner.min()), 1),
                            "inside_median_db": round(float(np.median(inner)), 1),
                            "inside_last5": [round(float(v), 1) for v in db[end - 5:end]],
                            "after_db": round(float(db[end:min(frames, end + 10)].mean()), 1),
                            "contour_db": [round(float(v), 1) for v in db[lo:hi]],
                        }
                    )

    result = {
        "rows": {"train": len(train_rows), "val": len(val_rows)},
        "frames": totals,
        "frame_ms": round(FRAME_MS, 2),
        "hist_rms_dbfs": {
            "edges": [round(-100 + 5 * i, 1) for i in range(21)],
            "counts": hist_rms.tolist(),
        },
        "hist_mel_z_frame_mean": {
            "edges": [round(-8 + 0.25 * i, 2) for i in range(41)],
            "counts": hist_mel.tolist(),
        },
        "digital_share_pct": round(100.0 * totals["digital"] / max(1, totals["frames"]), 3),
        "noise_share_pct": round(100.0 * totals["noise"] / max(1, totals["frames"]), 3),
        "per_prefix": {
            key: {
                "frames": value[1],
                "digital_frames": value[0],
                "digital_pct": round(100.0 * value[0] / max(1, value[1]), 3),
            }
            for key, value in sorted(per_prefix.items(), key=lambda kv: -kv[1][1])[:20]
        },
        "mel_mean_by_group_real": {
            name: round(value[0] / max(1, value[1]), 3) for name, value in group_sum.items()
        },
        "mel_mean_by_group_model": {
            name: round(value[0] / max(1, value[1]), 3) for name, value in model_sum.items()
        },
        "group_frames": {name: value[1] for name, value in group_sum.items()},
        "gap_counts": gap_scan,
        "gaps": gap_examples,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

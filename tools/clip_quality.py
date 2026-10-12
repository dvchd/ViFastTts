"""Lọc rẻ chất lượng clip trước MFA (B4): speech-ratio, clipping, RMS, độ phẳng phổ.

Chỉ loại **lỗi rõ** (~2–3%): nhạc nền/clip gần như không có tiếng nói, clipping nặng,
quá nhỏ. Phần tinh tế để score MFA lọc (REVISION_PLAN_V2 rev.5 §9.2).

Chạy:
    # 1) Đo metric toàn bộ clip → JSONL + danh sách loại
    python tools/clip_quality.py --manifest data/codelung/train.jsonl \
        --manifest data/codelung/val.jsonl --workers 8 \
        --out-jsonl data/codelung/quality.jsonl --out-drop data/codelung/quality_drop.txt

    # 2) Áp danh sách loại vào manifest/train/val (ghi đè; clip bị loại lưu vào excluded)
    python tools/clip_quality.py --apply --drop data/codelung/quality_drop.txt \
        --manifest data/codelung/manifest.jsonl --manifest data/codelung/train.jsonl \
        --manifest data/codelung/val.jsonl --excluded-out data/codelung/quality_excluded.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def _frame_db(samples: Any, sample_rate: int, frame_seconds: float = 0.02) -> Any:
    """RMS (dBFS) theo khung 20 ms."""
    import numpy as np

    step = max(1, int(sample_rate * frame_seconds))
    count = int(samples.size // step)
    if count == 0:
        return np.array([20.0 * math.log10(float(np.sqrt(np.mean(samples ** 2))) + 1e-12)])
    frames = samples[: count * step].reshape(count, step)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    return 20.0 * np.log10(rms)


def clip_metrics(samples: Any, sample_rate: int) -> dict[str, float]:
    """Metric chất lượng của một clip (nhẹ, không cần thư viện ngoài)."""
    import numpy as np

    db = _frame_db(samples, sample_rate)
    p90 = float(np.percentile(db, 90))
    speech_threshold = max(p90 - 30.0, -55.0)
    speech_ratio = float((db >= speech_threshold).mean())
    clipping_ratio = float((np.abs(samples) > 0.985).mean())
    rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(samples ** 2))) + 1e-12)
    # Độ phẳng phổ (GM/AM của công suất): nhạc/ồn nền cao hơn tiếng nói rõ rệt.
    window = samples[: min(samples.size, sample_rate * 2)]
    if window.size >= 256:
        spectrum = np.abs(np.fft.rfft(window * np.hanning(window.size)))
        power = spectrum ** 2 + 1e-12
        flatness_db = float(10.0 * (np.log10(power.mean()) - np.log10(power).mean()))
    else:
        flatness_db = 0.0
    return {
        "speech_ratio": round(speech_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 6),
        "rms_db": round(rms_db, 2),
        "flatness_db": round(flatness_db, 2),
    }


def quality_flags(metrics: dict[str, float], *, min_speech: float = 0.5,
                  max_clipping: float = 0.01, min_rms_db: float = -45.0) -> list[str]:
    """Cờ loại theo ngưỡng 'lỗi rõ' (B4)."""
    flags: list[str] = []
    if metrics["speech_ratio"] < min_speech:
        flags.append("speech_thap")
    if metrics["clipping_ratio"] > max_clipping:
        flags.append("clipping")
    if metrics["rms_db"] < min_rms_db:
        flags.append("rms_thap")
    return flags


def _worker(payload: tuple[str, int, dict[str, float]]) -> dict[str, Any]:
    """Worker: tính metric + cờ cho một clip."""
    audio_path, sample_rate, thresholds = payload

    from vifasttts.data.features import load_wav

    wav = load_wav(audio_path, sample_rate).numpy()
    metrics = clip_metrics(wav, sample_rate)
    return {"metrics": metrics, "flags": quality_flags(metrics, **thresholds)}


def measure(manifests: list[str], *, sample_rate: int = 24000, workers: int = 8,
            thresholds: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Đo metric cho mọi clip trong các manifest; trả danh sách bản ghi."""
    thresholds = thresholds or {}
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    payloads = [(str(row["audio_path"]), sample_rate, thresholds) for row in rows]

    from concurrent.futures import ProcessPoolExecutor

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for row, measured in zip(rows, pool.map(_worker, payloads), strict=True):
            results.append({"id": row["id"], "audio_path": row["audio_path"], **measured})
    return results


def apply_drop(manifests: list[str], drop_path: str, excluded_out: str | None) -> dict[str, int]:
    """Ghi đè manifest, bỏ các id trong danh sách loại; lưu clip bị loại ra `excluded_out`."""
    dropped = {line.strip() for line in Path(drop_path).read_text(encoding="utf-8").splitlines() if line.strip()}
    excluded: list[str] = []
    counts: dict[str, int] = {}
    for manifest in manifests:
        path = Path(manifest)
        kept: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["id"]) in dropped:
                excluded.append(line)
            else:
                kept.append(line)
        path.write_text("".join(entry + "\n" for entry in kept), encoding="utf-8")
        counts[path.name] = len(kept)
    if excluded_out:
        Path(excluded_out).write_text("\n".join(excluded) + ("\n" if excluded else ""), encoding="utf-8")
    return counts


def main() -> None:
    """Đo metric (mặc định) hoặc áp danh sách loại (`--apply`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="Manifest JSONL (lặp lại được)")
    parser.add_argument("--out-jsonl", default=None, help="Ghi metric toàn bộ clip")
    parser.add_argument("--out-drop", default=None, help="Ghi danh sách id bị loại")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-speech", type=float, default=0.5)
    parser.add_argument("--max-clipping", type=float, default=0.01)
    parser.add_argument("--min-rms-db", type=float, default=-45.0)
    parser.add_argument("--apply", action="store_true", help="Áp danh sách loại vào manifest")
    parser.add_argument("--drop", default=None, help="File danh sách id bị loại (cho --apply)")
    parser.add_argument("--excluded-out", default=None, help="Lưu các dòng bị loại (cho --apply)")
    args = parser.parse_args()

    if args.apply:
        if not args.drop:
            raise SystemExit("--apply cần --drop")
        counts = apply_drop(args.manifest, args.drop, args.excluded_out)
        print("Đã lọc:", ", ".join(f"{name} còn {count}" for name, count in counts.items()))
        return

    thresholds = {
        "min_speech": args.min_speech, "max_clipping": args.max_clipping, "min_rms_db": args.min_rms_db,
    }
    results = measure(args.manifest, sample_rate=args.sample_rate, workers=args.workers,
                      thresholds=thresholds)
    dropped = [row for row in results if row["flags"]]
    by_flag: dict[str, int] = {}
    for row in dropped:
        for flag in row["flags"]:
            by_flag[flag] = by_flag.get(flag, 0) + 1
    if args.out_jsonl:
        Path(args.out_jsonl).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8"
        )
    if args.out_drop:
        Path(args.out_drop).write_text(
            "\n".join(str(row["id"]) for row in dropped) + ("\n" if dropped else ""), encoding="utf-8"
        )
    print(f"Đo {len(results)} clip · loại {len(dropped)} ({100 * len(dropped) / max(1, len(results)):.1f}%)"
          f" · theo cờ: {by_flag or 'không có'}")


if __name__ == "__main__":
    main()

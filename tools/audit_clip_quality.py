"""QC chất lượng từng clip audio (trên mọi manifest được chỉ định) — phục vụ lọc dữ liệu.

Chỉ số per-clip (numpy + soundfile, PCM16 mono 24k):
- `noise_floor_db`: RMS frame ở percentile 5 (nền nhiễu, gồm cả đoạn lặng);
- `snr_db` = speech(p95) − noise_floor(p5) (ước lượng thô);
- `clip_frac`: tỉ lệ mẫu chạm biên (|x| ≥ 0,999 — nghi clipping);
- `flatness`: độ dẹt phổ trung vị (trên frame có tín hiệu) — cao = nhiễu/ồn;
- `hf_ratio`: phần năng lượng ≥ 4 kHz (trung vị trên frame có tín hiệu) — cao = sạn/sắc kim;
- `dur_s`: độ dài clip.

Khoá clip = `parent` nếu có, ngược lại `audio_path` (holdout không có `parent`).

Chạy:
    python tools/audit_clip_quality.py data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
        data/codelung/val_clean_v2_dur_seg_slot4.jsonl data/codelung/holdout_clean_v2_dur_slot4.jsonl \
        --report artifacts/clip_qc/clip_quality.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

WIN = 1024
HOP = 256
_METRICS = ("dur_s", "noise_floor_db", "speech_db", "snr_db", "clip_frac", "zero_frac",
            "flatness", "hf_ratio", "hf_ratio_quiet")


def analyze_clip(path: str) -> dict:
    """Đo các chỉ số QC cho một clip; lỗi đọc tệp → trả `error`."""
    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as error:  # tệp hỏng/không đọc được
        return {"error": f"{type(error).__name__}: {error}"}
    if data.ndim > 1:
        data = data.mean(axis=1)
    total = len(data)
    result: dict = {"dur_s": total / float(sample_rate)}
    if total < WIN:
        return {"error": "clip quá ngắn"}

    frames = 1 + (total - WIN) // HOP
    index = np.arange(WIN)[None, :] + HOP * np.arange(frames)[:, None]
    window = data[index]
    rms_db = 20.0 * np.log10(np.sqrt((window**2).mean(axis=1)) + 1e-12)
    # Bỏ khung im lặng SỐ (0 tuyệt đối) khi đo nền — chúng làm nền "0" giả (SNR phồng).
    nonzero = rms_db > -90.0
    base = rms_db[nonzero] if int(nonzero.sum()) >= 20 else rms_db
    noise_floor = float(np.percentile(base, 5))
    speech = float(np.percentile(base, 95))
    result.update(
        noise_floor_db=noise_floor,
        speech_db=speech,
        snr_db=speech - noise_floor,
        clip_frac=float((np.abs(data) >= 0.999).mean()),
        zero_frac=float((np.abs(data) < 1e-6).mean()),
        flatness=float("nan"),
        hf_ratio=float("nan"),
    )
    active = rms_db > (noise_floor + 10.0)
    if int(active.sum()) >= 5:
        hann = np.hanning(WIN).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(window[active] * hann, axis=1)) ** 2 + 1e-12
        geometric = np.exp(np.log(spectrum).mean(axis=1))  # trung bình nhân phổ mỗi frame
        arithmetic = spectrum.mean(axis=1)
        result["flatness"] = float(np.median(geometric / arithmetic))
        freqs = np.fft.rfftfreq(WIN, 1.0 / sample_rate)
        high = freqs >= 4000.0
        result["hf_ratio"] = float(np.median(spectrum[:, high].sum(axis=1) / spectrum.sum(axis=1)))
    # Hiss/sizzle: cao tần xuất hiện ngay TRÊN NỀN (khung lặng) — khác fricative (khung to).
    quiet = (rms_db > noise_floor + 3.0) & (rms_db <= noise_floor + 10.0)
    if int(quiet.sum()) >= 5:
        hann = np.hanning(WIN).astype(np.float32)
        quiet_spectrum = np.abs(np.fft.rfft(window[quiet] * hann, axis=1)) ** 2 + 1e-12
        freqs = np.fft.rfftfreq(WIN, 1.0 / sample_rate)
        high = freqs >= 4000.0
        result["hf_ratio_quiet"] = float(
            np.median(quiet_spectrum[:, high].sum(axis=1) / quiet_spectrum.sum(axis=1))
        )
    return result


def _job(item: tuple[str, str]) -> tuple[str, dict]:
    key, path = item
    return key, analyze_clip(path)


def collect_clips(manifests: list[str]) -> dict[str, str]:
    """Gom clip duy nhất từ các manifest: khoá = `parent` (nếu có) hoặc `audio_path`."""
    clips: dict[str, str] = {}
    for manifest in manifests:
        with Path(manifest).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("parent") or row["audio_path"])
                clips[key] = str(row["audio_path"])
    return clips


def summarize(rows: dict[str, dict]) -> dict:
    """Phân bố percentile cho từng chỉ số (bỏ hàng lỗi)."""
    summary: dict[str, dict] = {}
    for metric in _METRICS:
        values = np.array([
            float(item[metric]) for item in rows.values()
            if "error" not in item and not np.isnan(item.get(metric, float("nan")))
        ])
        if values.size == 0:
            continue
        summary[metric] = {
            "n": int(values.size),
            **{f"p{p}": float(np.percentile(values, p)) for p in (1, 5, 25, 50, 75, 95, 99)},
            "min": float(values.min()), "max": float(values.max()),
        }
    return summary


def main() -> None:
    """Đọc tham số, chạy QC song song, in phân bố và ghi báo cáo JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--report", required=True, help="Đường dẫn JSON báo cáo")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Chỉ chạy N clip đầu (thử nhanh)")
    args = parser.parse_args()

    clips = collect_clips(args.manifests)
    items = sorted(clips.items())
    if args.limit:
        items = items[: args.limit]
    print(f"Clip duy nhất: {len(clips)} | chạy QC: {len(items)} clip | workers={args.workers}")

    rows: dict[str, dict] = {}
    errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (key, metrics) in enumerate(pool.map(_job, items, chunksize=8), 1):
            if "error" in metrics:
                errors += 1
            rows[key] = {"path": clips[key], **metrics}
            if index % 1000 == 0:
                print(f"  ... {index}/{len(items)}")

    report = {
        "manifests": args.manifests,
        "clips": len(items),
        "errors": errors,
        "percentiles": summarize(rows),
        "rows": rows,
    }
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["percentiles"], ensure_ascii=False, indent=1))
    print(f"Đã ghi {out_path} | lỗi đọc: {errors}")


if __name__ == "__main__":
    main()

"""Dựng dữ liệu v2: lọc giọng khách/nhiễm + chuẩn hoá loudness + manifest mới.

Quyết định (docs/REBUILD_PLAN_M6.md §3, chốt 12/10/2026):
- **Lọc tại chỗ trên clip hiện có** (không recut lại từ raw — recut B3 15:21 hôm nay đã
  là bản cắt cuối; recut lại không đổi chất lượng, tốn thêm ~1 ngày).
- **Bỏ 6 video loạt khách** (`KIrbA-wEURg, LY8_RaLDJT4, W1m6birJBj0, _U4H5FSIb-c,
  nEPr-jIokLw, pLXBrJ-M-Xo` — 937 clip) + **48 clip nhiễm tiếng khác**
  (`data/codelung/review/diar/drop_other_ge_0p5.txt`, other_s ≥ 0,5s).
- **Chuẩn hoá loudness từng clip** về **−26 dBFS RMS trên frame có tiếng**
  (peak-safe ≤ −1 dBFS, boost tối đa +14 dB), ghi gain vào `loudness.jsonl`.
  WAV mới ở `data/codelung/audio_v2/` — **giữ nguyên** audio gốc để đối chiếu.

Chạy:
    python tools/normalize_audio_v2.py --exclude-clips data/codelung/review/diar/drop_other_ge_0p5.txt
Kết quả: `manifest_v2.jsonl`, `loudness.jsonl`, `excluded_speakers.jsonl`, `excluded_other.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

#: Video loạt "khách" bỏ hẳn (REBUILD_PLAN_M6 §3).
EXCLUDE_VIDEOS = ("KIrbA-wEURg", "LY8_RaLDJT4", "W1m6birJBj0", "_U4H5FSIb-c", "nEPr-jIokLw", "pLXBrJ-M-Xo")
TARGET_DB = -26.0          # RMS mục tiêu trên frame có tiếng
MAX_BOOST_DB = 14.0        # trần boost (chống lôi nền ồn lên)
PEAK_CEIL = 0.891          # sau gain, đỉnh ≤ −1 dBFS
SPEECH_P90_OFFSET = 30.0   # ngưỡng tiếng nói: p90 − 30 dB
SPEECH_FLOOR_DB = -55.0


def _frame_db(samples: Any, sample_rate: int, frame_seconds: float = 0.02) -> Any:
    """RMS (dBFS) theo khung 20 ms (giống tools/clip_quality.py)."""
    import numpy as np

    step = max(1, int(sample_rate * frame_seconds))
    count = int(samples.size // step)
    if count == 0:
        return np.array([20.0 * math.log10(float(np.sqrt(np.mean(samples ** 2))) + 1e-12)])
    frames = samples[: count * step].reshape(count, step)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    return 20.0 * np.log10(rms)


def _worker(payload: tuple[str, str, str, int, float]) -> dict[str, Any]:
    """Chuẩn hoá một clip: tính gain (frame có tiếng → −26 dBFS, peak-safe), ghi WAV mới."""
    import numpy as np
    import soundfile as sf

    from vifasttts.data.features import load_wav

    sample_id, src, dst, sample_rate, _target_db = payload
    wav = load_wav(src, sample_rate).numpy().astype(np.float64)
    db = _frame_db(wav, sample_rate)
    p90 = float(np.percentile(db, 90))
    threshold = max(p90 - SPEECH_P90_OFFSET, SPEECH_FLOOR_DB)
    speech = db[db >= threshold]
    if speech.size == 0:
        speech = db
    speech_rms_db = 10.0 * math.log10(float(np.mean(10.0 ** (speech / 10.0))))
    gain_db = min(TARGET_DB - speech_rms_db, MAX_BOOST_DB)
    peak = float(np.max(np.abs(wav))) * (10.0 ** (gain_db / 20.0))
    if peak > PEAK_CEIL:
        gain_db += 20.0 * math.log10(PEAK_CEIL / peak)
    gain = 10.0 ** (gain_db / 20.0)
    scaled = np.clip(wav * gain, -1.0, 1.0).astype(np.float32)
    sf.write(dst, scaled, sample_rate, subtype="PCM_16")
    return {
        "id": sample_id,
        "gain_db": round(gain_db, 2),
        "speech_rms_before_db": round(speech_rms_db, 2),
        "peak_after_db": round(20.0 * math.log10(max(float(np.max(np.abs(scaled))), 1e-9)), 2),
        "src": src,
    }


def main() -> None:
    """Lọc + chuẩn hoá + ghi manifest v2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/codelung/manifest.jsonl")
    parser.add_argument("--clips", default="data/codelung/clips.jsonl")
    parser.add_argument("--out-dir", default="data/codelung")
    parser.add_argument("--audio-dir", default="data/codelung/audio_v2")
    parser.add_argument("--exclude-clips", default=None, help="File danh sách id loại (1 dòng/id)")
    parser.add_argument("--exclude-videos", default=",".join(EXCLUDE_VIDEOS))
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-db", type=float, default=TARGET_DB)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    exclude_videos = {v.strip() for v in str(args.exclude_videos).split(",") if v.strip()}
    exclude_clips: set[str] = set()
    if args.exclude_clips:
        exclude_clips = {line.strip() for line in Path(args.exclude_clips).read_text().splitlines() if line.strip()}

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    clips_meta = {str(c["id"]): c for c in (json.loads(line) for line in Path(args.clips).read_text(encoding="utf-8").splitlines() if line.strip())}

    kept: list[dict[str, Any]] = []
    excluded_speakers: list[dict[str, Any]] = []
    excluded_other: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["id"])
        video = str(clips_meta.get(sample_id, {}).get("video", ""))
        if video in exclude_videos:
            excluded_speakers.append({"id": sample_id, "video": video, "reason": "guest_series"})
            continue
        if sample_id in exclude_clips:
            excluded_other.append({"id": sample_id, "reason": "other_speaker_diarization"})
            continue
        kept.append(row)

    print(f"Giữ {len(kept)} | loại series khách {len(excluded_speakers)} | loại diarization {len(excluded_other)}")

    payloads: list[tuple[str, str, str, int, float]] = []
    for row in kept:
        sample_id = str(row["id"])
        payloads.append((sample_id, str(Path(str(row["audio_path"])).resolve()),
                         str(audio_dir / f"{sample_id}.wav"), args.sample_rate, args.target_db))

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, record in enumerate(pool.map(_worker, payloads, chunksize=16), start=1):
            records.append(record)
            if index % 1000 == 0 or index == len(payloads):
                print(f"  chuẩn hoá {index}/{len(payloads)} clip", flush=True)

    gains = sorted(r["gain_db"] for r in records)
    median = gains[len(gains) // 2]
    print(f"Gain (dB): min {gains[0]:+.1f} · median {median:+.1f} · max {gains[-1]:+.1f}")

    by_id = {r["id"]: r for r in records}
    manifest_path = out_dir / "manifest_v2.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in kept:
            sample_id = str(row["id"])
            new_row = {**row, "audio_path": str(audio_dir / f"{sample_id}.wav")}
            fh.write(json.dumps(new_row, ensure_ascii=False) + "\n")
    (out_dir / "loudness.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in by_id[key].items() if k != 'src'}, ensure_ascii=False) + "\n" for key in sorted(by_id)),
        encoding="utf-8",
    )
    (out_dir / "excluded_speakers.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in excluded_speakers), encoding="utf-8")
    (out_dir / "excluded_other.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in excluded_other), encoding="utf-8")
    print(f"Đã ghi {manifest_path}, loudness.jsonl, excluded_*.jsonl → {out_dir}")


if __name__ == "__main__":
    main()

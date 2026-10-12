"""So chất giọng (creak của ngã/nặng) giữa audio gốc và bản copy-synthesis — cổng R8-1.

Chạy hàng loạt (sau khi có thư mục synth của `tools/eval_vocoder.py`):

    python tools/voice_quality.py --manifest data/codelung/eval_clips_v1.jsonl \
        --synth-dir artifacts/vocoder_eval/step_0/wavs \
        --output artifacts/vocoder_eval/step_0/voice_quality.json

Tùy chọn `--segments segments.jsonl` (mỗi dòng `{"id", "start", "end", "tone"?}` giây)
để đo riêng từng âm tiết/đoạn — sau MFA có thể sinh file này từ TextGrid.
Nhóm theo `tone` (NGA/NANG/...) để xem vocoder có làm mượt thanh hầu hoá không.

Metric (trên frame hữu thanh, cùng config mel của acoustic):
- HNR trung bình (dB): creak ⇒ HNR thấp; vocoder làm mượt ⇒ HNR synth CAO hơn gốc.
- `hnr_creak_delta`: HNR synth − HNR gốc TRÊN đúng các frame gốc thuộc nhóm creak
  (quartile thấp nhất của HNR gốc) — chỉ số chính của R8-1.
- tilt (dB/oct), glottalization (tỉ lệ năng lượng <80 Hz), perturb (|ΔlogF0| giữa
  frame hữu thanh liền kề — proxy jitter).

Ngưỡng R8-1 (đề xuất): `hnr_creak_delta` không dương rõ rệt (≤ ~2 dB) trên ngã/nặng;
tone acc từng thanh (classifier E3) giảm ≤ 2 điểm so với audio gốc.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

#: Quartile chọn frame "creak" theo HNR gốc.
CREAK_QUANTILE = 0.25


def _phonation_tracks(wav: np.ndarray, sample_rate: int, hop_length: int, n_fft: int) -> dict[str, np.ndarray]:
    """HNR/tilt/glottalization (repo) + F0/VUV (pyworld) cấp frame."""
    import torch

    from vifasttts.data.features import estimate_f0_pyworld
    from vifasttts.data.phonation import estimate_hnr, glottalization_ratio, spectral_tilt

    tensor = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    tracks = {
        "hnr": estimate_hnr(tensor, sample_rate, hop_length, frame_length=max(n_fft, 512)).numpy(),
        "tilt": spectral_tilt(tensor, sample_rate, hop_length, n_fft=n_fft).numpy(),
        "glottalization": glottalization_ratio(tensor, sample_rate, hop_length, n_fft=n_fft).numpy(),
    }
    f0, vuv = estimate_f0_pyworld(tensor, sample_rate, hop_length, f_min=60.0, f_max=500.0, interpolate=False)
    tracks["f0"], tracks["vuv"] = f0.numpy(), vuv.numpy()
    return tracks


def _frame_db(wav: np.ndarray, hop_length: int, frame_length: int = 1024) -> np.ndarray:
    """Năng lượng RMS theo frame (dB) để gate im lặng."""
    frames = np.lib.stride_tricks.sliding_window_view(
        np.pad(wav, (frame_length // 2, frame_length // 2)), frame_length
    )[::hop_length]
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def compare_pair(original: np.ndarray, synth: np.ndarray, sample_rate: int, hop_length: int,
                 n_fft: int) -> dict[str, float]:
    """So phonation giữa hai waveform (cùng độ dài ~); trả dict metric."""
    frames = min(len(original), len(synth))
    original, synth = original[:frames], synth[:frames]
    ref = _phonation_tracks(original, sample_rate, hop_length, n_fft)
    out = _phonation_tracks(synth, sample_rate, hop_length, n_fft)
    n = min(len(ref["hnr"]), len(out["hnr"]), len(_frame_db(original, hop_length)))
    voiced = (ref["vuv"][:n] > 0) & (ref["f0"][:n] > 0)
    energy = _frame_db(original, hop_length)[:n]
    voiced &= energy > (energy.max() if energy.size else 0.0) - 35.0
    metrics: dict[str, float] = {"frames": float(n), "voiced_frames": float(voiced.sum())}
    if voiced.sum() < 5:
        return metrics

    def mean(track: str, mask: np.ndarray, values: dict[str, np.ndarray]) -> float:
        return float(np.mean(values[track][:n][mask])) if mask.any() else float("nan")

    for track in ("hnr", "tilt", "glottalization"):
        metrics[f"{track}_mean_orig"] = mean(track, voiced, ref)
        metrics[f"{track}_mean_synth"] = mean(track, voiced, out)
    # Perturbation F0: |ΔlogF0| giữa hai frame hữu thanh liền kề.
    for tag, values in (("orig", ref), ("synth", out)):
        f0 = values["f0"][:n]
        v = (values["vuv"][:n] > 0) & (f0 > 0)
        pairs = v[:-1] & v[1:] & (np.isfinite(np.log(np.clip(f0[:-1], 1e-6, None))))
        if pairs.any():
            metric = np.abs(np.log(np.clip(f0[1:][pairs], 1e-6, None)) - np.log(np.clip(f0[:-1][pairs], 1e-6, None)))
            metrics[f"perturb_{tag}"] = float(np.mean(metric))
    # Nhóm creak theo HNR gốc (quartile thấp nhất trong frame hữu thanh).
    hnr_voiced = ref["hnr"][:n][voiced]
    threshold = float(np.quantile(hnr_voiced, CREAK_QUANTILE))
    creak = voiced & (ref["hnr"][:n] <= threshold)
    metrics["creak_frac"] = float(creak.sum()) / max(1.0, float(voiced.sum()))
    metrics["creak_hnr_threshold_db"] = threshold
    for track in ("hnr", "glottalization"):
        metrics[f"{track}_creak_orig"] = mean(track, creak, ref)
        metrics[f"{track}_creak_synth"] = mean(track, creak, out)
    if np.isfinite(metrics.get("hnr_creak_orig", float("nan"))) and np.isfinite(metrics.get("hnr_creak_synth", float("nan"))):
        metrics["hnr_creak_delta"] = metrics["hnr_creak_synth"] - metrics["hnr_creak_orig"]
    return metrics


def _load_segments(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Đọc segments.jsonl → {id: [{start, end, tone?}]}."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault(str(row["id"]), []).append(row)
    return grouped


def _aggregate(entries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Trung bình/p90 mỗi metric qua các entry."""
    keys = sorted({key for entry in entries for key in entry if isinstance(entry[key], (int, float))})
    aggregate: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.array([entry[key] for entry in entries if key in entry], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            aggregate[key] = {"mean": float(values.mean()), "p90": float(np.percentile(values, 90))}
    return aggregate


def main() -> None:
    """Đọc manifest + synth-dir (+segments), tính metric, ghi JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Manifest của tập eval (audio gốc)")
    parser.add_argument("--synth-dir", required=True, help="Thư mục WAV synth (như eval_vocoder)")
    parser.add_argument("--output", required=True, help="File JSON kết quả")
    parser.add_argument("--segments", default=None, help="segments.jsonl {id,start,end,tone?}")
    parser.add_argument("--clips", type=int, default=50)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-fft", type=int, default=1024)
    args = parser.parse_args()

    from vifasttts.data.features import load_wav

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    segments = _load_segments(args.segments) if args.segments else None
    synth_dir = Path(args.synth_dir)

    per_item: list[dict[str, Any]] = []
    for row in rows[: args.clips]:
        synth_path = synth_dir / f"{row['id']}.wav"
        if not synth_path.exists():
            print(f"  bỏ qua {row['id']}: thiếu {synth_path}")
            continue
        original = load_wav(str(row["audio_path"]), args.sample_rate).numpy()
        synth = load_wav(str(synth_path), args.sample_rate).numpy()
        clips = segments.get(str(row["id"])) if segments else None
        if not clips:
            entry = {"id": row["id"]}
            entry.update(compare_pair(original, synth, args.sample_rate, args.hop_length, args.n_fft))
            per_item.append(entry)
            continue
        for index, segment in enumerate(clips):
            start = int(float(segment["start"]) * args.sample_rate)
            end = int(float(segment["end"]) * args.sample_rate)
            entry = {
                "id": f"{row['id']}#{index}",
                "tone": segment.get("tone"),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
            }
            entry.update(compare_pair(original[start:end], synth[start:end], args.sample_rate,
                                      args.hop_length, args.n_fft))
            per_item.append(entry)

    by_tone: dict[str, list[dict[str, Any]]] = {}
    for entry in per_item:
        tone = str(entry.get("tone") or "ALL")
        by_tone.setdefault(tone, []).append(entry)
    result = {
        "manifest": str(args.manifest),
        "synth_dir": str(synth_dir),
        "segments": bool(segments),
        "clips": len(per_item),
        "aggregate": _aggregate(per_item),
        "by_tone": {tone: _aggregate(items) for tone, items in sorted(by_tone.items())},
        "per_item": per_item,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    def show(label: str, aggregate: dict[str, dict[str, float]]) -> None:
        delta = aggregate.get("hnr_creak_delta", {}).get("mean")
        frac = aggregate.get("creak_frac", {}).get("mean")
        print(f"  {label:12s} hnr_creak_delta={delta if delta is not None else float('nan'):+7.2f} dB "
              f"· creak_frac={frac if frac is not None else float('nan'):.2%}")

    print(f"[voice-quality] {len(per_item)} đoạn → {args.output}")
    show("ALL", result["aggregate"])
    for tone, aggregate in result["by_tone"].items():
        if tone != "ALL":
            show(tone, aggregate)


if __name__ == "__main__":
    main()

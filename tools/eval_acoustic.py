"""tools/eval_acoustic.py — cổng nghiệm thu acoustic (T1.2 / B6).

Chế độ:
  tf_syl — cho `d_syl`/`gaps`/`lead` thật (v1: truyền `spans`); đo mel L1 (z/raw/dB),
           F0 RMSE theo 6 thanh (chỉ frame hữu thanh thật, mask `vuv`), VUV, energy,
           duration (từ `log_duration`).
  tf_seg — v2 (cho thêm `seg_durations`); v1: raise NotImplementedError.
  free   — không cho gì: CER (Whisper), RTF, RAM.

Mask vùng đo lấy từ manifest (Addendum 3 §2.1): mel trên [lead, T); F0/VUV chỉ
frame hữu thanh của âm tiết (`durations`), bỏ gap + lead.

Ví dụ:
  python tools/eval_acoustic.py --config configs/small_codelung.yaml \
      --checkpoint runs/small_codelung_v2/last.pt --mode tf_syl \
      --manifest data/codelung/val_clean_v2_dur_seg.jsonl \
      --cache-dir cache/codelung_v2 --clips 300 \
      --out artifacts/eval_acoustic/b0_last20k_tf_syl.json

  python tools/eval_acoustic.py --config configs/small_codelung.yaml \
      --checkpoint runs/small_codelung_v2/last.pt --mode free \
      --manifest data/codelung/eval_clips_v1.jsonl --clips 20 \
      --out artifacts/eval_acoustic/b0_last20k_free.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TONES = ["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]
_VI_CHARS = "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"


def _read_rows(manifest: str | Path) -> list[dict]:
    """Đọc manifest JSONL."""
    return [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _quantiles(values: list[float]) -> dict[str, float]:
    """n / mean / median cho một danh sách số."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {"n": int(array.size), "mean": round(float(array.mean()), 4),
            "median": round(float(np.median(array)), 4)}


def _char_cer(reference: str, hypothesis: str) -> float:
    """CER cấp ký tự sau chuẩn hoá (bỏ dấu câu, thường hoá, gộp khoảng trắng)."""
    def clean(text: str) -> str:
        text = text.lower()
        text = re.sub(rf"[^\w\s{_VI_CHARS}]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    ref, hyp = clean(reference), clean(hypothesis)
    if not ref:
        return 0.0
    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start=1):
        current = [i]
        for j, hyp_char in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1] / len(ref)


def _load_model(config: dict, checkpoint: str):
    """Nạp model acoustic từ checkpoint (hỗ trợ cả checkpoint `train` lẫn trần state)."""
    import torch

    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.models.vifasttts import ViFastTtsModel

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
             len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ViFastTtsModel(config, sizes)
    model.load_state_dict(state.get("model", state))
    model.eval()
    return model


def run_tf_syl(args, config: dict) -> dict:
    """Teacher-forced với d_syl/gaps thật: đo mel/F0/VUV/energy/duration."""
    import torch

    from vifasttts.data.dataset import TtsDataset

    dataset = TtsDataset(args.manifest, config, cache_dir=args.cache_dir)
    model = _load_model(config, args.checkpoint)
    mel_stats = json.loads(Path(config["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean = torch.tensor(mel_stats["mean"])
    std = torch.tensor(mel_stats["std"]).clamp_min(1e-4)
    hop = int(config["mel"]["hop_length"])
    sample_rate = int(config.get("sample_rate", 24000))
    frame_ms = hop / sample_rate * 1000.0

    tone_f0: dict[str, list[float]] = {tone: [] for tone in TONES}
    per_clip: list[dict] = []
    vuv_hits = vuv_total = 0
    energy_errors: list[float] = []
    duration_errors: list[float] = []

    for index in range(min(args.clips, len(dataset))):
        row = dataset.rows[index]
        item = dataset[index]
        with torch.no_grad():
            outputs = model(item["features"].unsqueeze(0), durations=item["spans"].unsqueeze(0))

        pred_mel, target_mel = outputs["mel"][0], item["mel"]
        frames = min(pred_mel.size(0), target_mel.size(0))
        l1_z = float((pred_mel[:frames] - target_mel[:frames]).abs().mean())
        raw_pred = pred_mel[:frames] * std + mean
        raw_target = target_mel[:frames] * std + mean
        l1_raw = float((raw_pred - raw_target).abs().mean())

        predicted_duration = torch.round((torch.exp(outputs["log_duration"][0]) - 1).clamp_min(1))
        duration_errors.extend((predicted_duration - item["durations"]).abs().tolist())

        predicted_vuv = (torch.sigmoid(outputs["vuv_logits"][0]) > 0.5).float()
        vuv_hits += int((predicted_vuv == item["vuv"]).sum())
        vuv_total += int(item["vuv"].numel())
        energy_errors.append(float((outputs["energy"][0] - item["energy"]).abs().mean()))

        extracted = dataset.cache.get(row) if dataset.cache is not None else None
        if extracted is not None:
            frame_range = row.get("frame_range")
            position = int(frame_range[0]) if frame_range else 0
            predicted_f0 = outputs["f0"][0].tolist()
            tones = [syllable["abstract"]["tone"] for syllable in row["syllables"]]
            for syllable_index, (duration, span) in enumerate(
                    zip(item["durations"].tolist(), item["spans"].tolist())):
                true_f0 = extracted["f0"][position:position + duration].numpy()
                true_vuv = extracted["vuv"][position:position + duration].numpy()
                voiced = true_vuv > 0.5
                if voiced.any():
                    ratio = np.maximum(predicted_f0[syllable_index], 1e-6) / np.maximum(
                        true_f0[voiced], 1e-6)
                    cents = 1200.0 * np.abs(np.log2(np.maximum(ratio, 1e-9)))
                    tone_f0[tones[syllable_index]].extend(cents.tolist())
                position += span

        per_clip.append({
            "id": row["id"], "frames": frames,
            "mel_l1_z": round(l1_z, 4), "mel_l1_raw": round(l1_raw, 4),
            "mel_mae_db": round(8.686 * l1_raw, 3),
            "duration_mae_frames": round(float((predicted_duration - item["durations"]).abs().mean()), 3),
        })

    summary = {
        "mode": "tf_syl", "checkpoint": args.checkpoint, "manifest": args.manifest,
        "clips": len(per_clip), "clip_ids": [entry["id"] for entry in per_clip],
        "mel_l1_z": _quantiles([entry["mel_l1_z"] for entry in per_clip]),
        "mel_l1_raw": _quantiles([entry["mel_l1_raw"] for entry in per_clip]),
        "mel_mae_db": _quantiles([entry["mel_mae_db"] for entry in per_clip]),
        "f0_rmse_cents_by_tone": {tone: _quantiles(values) for tone, values in tone_f0.items()},
        "vuv_accuracy": round(vuv_hits / max(vuv_total, 1), 4),
        "energy_mae": round(float(np.mean(energy_errors)), 4) if energy_errors else None,
        "duration_mae_frames": _quantiles(duration_errors),
        "duration_mae_ms": round(float(np.mean(duration_errors)) * frame_ms, 2) if duration_errors else None,
        "per_clip": per_clip,
    }
    return summary


def _load_whisper():
    """Nạp faster-whisper small (int8, CPU); None nếu thiếu thư viện."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("⚠️  Không có faster-whisper — bỏ qua CER")
        return None
    return WhisperModel("small", device="cpu", compute_type="int8")


def _transcribe(whisper, path: Path) -> str:
    """Whisper transcribe một file WAV (tự resample 16 kHz bằng numpy)."""
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32")
    target = round(len(data) * 16000 / rate)
    resampled = np.interp(np.linspace(0, len(data) - 1, target), np.arange(len(data)),
                          data).astype(np.float32)
    segments, _ = whisper.transcribe(resampled, language="vi", beam_size=1)
    return " ".join(segment.text for segment in segments).strip()


def run_free(args, config: dict) -> dict:
    """Free-run: text → acoustic → vocoder; đo CER (Whisper), RTF, RAM (PyTorch CPU)."""
    import resource
    import time

    from vifasttts.infer import Synthesizer

    rows = _read_rows(args.manifest)[: args.clips]
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    wav_dir = Path(args.out).parent / "free_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    whisper = _load_whisper()

    per_clip: list[dict] = []
    total_audio = total_time = 0.0
    for row in rows:
        target = wav_dir / f"{row['id']}.wav"
        started = time.perf_counter()
        samples, rate = engine.synthesize(row["text"], target)
        elapsed = time.perf_counter() - started
        audio_seconds = samples / rate
        total_audio += audio_seconds
        total_time += elapsed
        hypothesis = _transcribe(whisper, target) if whisper is not None else ""
        per_clip.append({
            "id": row["id"], "audio_seconds": round(audio_seconds, 3),
            "synth_seconds": round(elapsed, 3),
            "cer": round(_char_cer(row["text"], hypothesis), 4),
            "whisper": hypothesis[:200],
        })
        print(f"  [free] {row['id']} · {audio_seconds:.1f}s · CER {per_clip[-1]['cer']:.3f}")

    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes
    return {
        "mode": "free", "checkpoint": args.checkpoint, "manifest": args.manifest,
        "clips": len(per_clip), "device": args.device, "rtf_backend": "pytorch-cpu",
        "cer": _quantiles([entry["cer"] for entry in per_clip]),
        "rtf": round(total_time / max(total_audio, 1e-9), 4),
        "ram_peak_gib": round(rss_gib, 2),
        "per_clip": per_clip,
    }


def main() -> None:
    """Đọc tham số, chạy mode, ghi JSON + in bảng tóm tắt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["tf_syl", "tf_seg", "free"], default="tf_syl")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clips", type=int, default=100)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.mode == "tf_seg":
        raise SystemExit("tf_seg chỉ dùng cho v2 (arch=v2_slot5) — v1 không có seg_durations")
    summary = run_tf_syl(args, config) if args.mode == "tf_syl" else run_free(args, config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {out}")
    if args.mode == "tf_syl":
        print(f"  {summary['clips']} clip · mel L1 z {summary['mel_l1_z']['mean']:.3f} "
              f"· raw {summary['mel_l1_raw']['mean']:.3f} · {summary['mel_mae_db']['mean']:.1f} dB")
        print(f"  VUV acc {summary['vuv_accuracy']:.3f} · energy MAE {summary['energy_mae']:.3f} "
              f"· duration MAE {summary['duration_mae_ms']:.1f} ms")
        for tone, stats in summary["f0_rmse_cents_by_tone"].items():
            if stats.get("n"):
                print(f"  F0 {tone:6s} median {stats['median']:7.1f} cents (n={stats['n']})")
    else:
        print(f"  {summary['clips']} clip · CER TB {summary['cer']['mean']:.3f} "
              f"· RTF {summary['rtf']:.3f} · RAM đỉnh {summary['ram_peak_gib']:.2f} GiB")


if __name__ == "__main__":
    main()

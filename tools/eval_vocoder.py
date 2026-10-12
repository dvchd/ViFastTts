"""Cổng nghiệm thu vocoder (R7-1): copy-synthesis + mel L1 + log-F0 RMSE + VUV + clipping.

Dùng **cùng một script** cho mọi mốc (0 = pretrained zero-shot, 1k/2k/5k/10k...):

    # Bước 0 (zero-shot): lưu đúng định dạng checkpoint "sạch"
    vifasttts train-vocoder --config configs/small_codelung.yaml \
        --manifest data/codelung/train.jsonl --output runs/vocoder_codelung_v3 --max-steps 0

    # Đo tại từng mốc
    python tools/eval_vocoder.py --config configs/small_codelung.yaml \
        --checkpoint runs/vocoder_codelung_v3/last.pt \
        --manifest data/codelung/eval_clips_v1.jsonl \
        --output artifacts/vocoder_eval/step_0 --clips 50

Kết quả: `<output>/wavs/*.wav` (nghe thử) + `metrics.json` (tổng hợp + per-clip).
Metric F0 cần `pyworld` (extra `f0`); thiếu thì bỏ qua kèm cảnh báo.

Lưu ý so sánh: F0 RMSE tuyệt đối còn nhiễu tracker trên audio có tạp âm nền — dùng
chủ yếu để so giữa các mốc/checkpoint (cùng tập eval, cùng script).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_vocoder(config: dict[str, Any], checkpoint: str | Path, device: str = "cpu"):
    """Nạp generator từ checkpoint `train-vocoder` (khóa `vocoder`) hoặc file HF pretrained."""
    import torch

    from vifasttts.data.features import canonical_mel_config
    from vifasttts.models.istftnet import build_generator, load_generator_state

    mel_config = canonical_mel_config(config)
    generator = build_generator(config.get("vocoder_training", {}), mel_config)
    load_generator_state(generator, checkpoint)
    generator.to(torch.device(device))
    generator.eval()
    return generator


def _median_filter(values, window: int = 5):
    """Lọc trung vị đơn giản để giảm jitter của tracker F0 trước khi so."""
    import numpy as np

    if len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.array([np.median(padded[i : i + window]) for i in range(len(values))])


def f0_metrics(reference, synth, sample_rate: int, hop_length: int) -> dict[str, float]:
    """log-F0 RMSE (cents) + lỗi VUV giữa audio gốc và bản synthesis (pyworld)."""
    import numpy as np
    import pyworld

    def track(waveform) -> np.ndarray:
        samples = np.ascontiguousarray(waveform.astype(np.float64))
        f0, times = pyworld.harvest(samples, sample_rate, f0_floor=60.0, f0_ceil=500.0,
                                    frame_period=1000.0 * hop_length / sample_rate)
        return pyworld.stonemask(samples, f0, times, sample_rate)

    n = min(len(reference), len(synth))
    ref = _median_filter(track(reference[:n]))
    syn = _median_filter(track(synth[:n]))
    voiced = ref > 0
    both = voiced & (syn > 0)
    metrics: dict[str, float] = {
        "voiced_frames": float(voiced.sum()),
        "vuv_err": float((~both[voiced]).mean()) if voiced.any() else 0.0,
        "vuv_err_unvoiced": float((syn[~voiced] > 0).mean()) if (~voiced).any() else 0.0,
    }
    if both.any():
        cents = np.abs(np.log(syn[both] / ref[both])) * (1200.0 / np.log(2.0))
        metrics["logf0_rmse_cents"] = float(np.sqrt((cents**2).mean()))
        keep = cents <= 600.0  # bỏ lỗi quãng tám của tracker (đo riêng bằng octave_rate)
        metrics["logf0_rmse_cents_no_oct"] = float(np.sqrt((cents[keep] ** 2).mean())) if keep.any() else float("nan")
        metrics["octave_rate"] = float((cents > 600.0).mean())
        metrics["err_lt25_frac"] = float((cents < 25.0).mean())
    return metrics


def mel_metrics(generator, real_mel, synth_wav, mel_config: dict[str, Any]) -> dict[str, float]:
    """Mel L1 trong **không gian của generator** (Vocos: log-magnitude thô) + clipping."""
    from vifasttts.data.features import extract_mel_energy

    loss_config = generator.loss_mel_config(mel_config) if hasattr(generator, "loss_mel_config") else mel_config
    target = generator.denormalize(real_mel) if hasattr(generator, "denormalize") else real_mel
    synth_mel, _ = extract_mel_energy(synth_wav.cpu(), loss_config)
    frames = min(synth_mel.size(0), target.size(0))
    mel_l1 = float((synth_mel[:frames] - target[:frames]).abs().mean()) if frames else float("inf")
    hop = int(loss_config["hop_length"])
    return {
        "mel_l1": round(mel_l1, 4),
        "clipping_ratio": round(float((synth_wav.abs() > 0.985).float().mean()), 5),
        "length_ratio": round(float(synth_wav.numel()) / max(1, target.size(0) * hop), 4),
        "frames": frames,
    }


def evaluate(config: dict[str, Any], checkpoint: str | Path, rows: list[dict[str, Any]],
             output: str | Path, clips: int = 50, device: str = "cpu",
             sample_rate: int | None = None) -> dict[str, Any]:
    """Chạy copy-synthesis + metric cho `clips` mẫu; ghi WAV + metrics.json; trả tổng hợp."""
    import numpy as np
    import torch

    from vifasttts.data.features import canonical_mel_config, extract_mel_energy, load_wav
    from vifasttts.infer import write_wav

    sample_rate = int(sample_rate or config.get("sample_rate", 24000))
    mel_config = canonical_mel_config(config)
    generator = load_vocoder(config, checkpoint, device=device)
    out_dir = Path(output)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyworld  # noqa: F401

        has_f0 = True
    except ImportError:
        has_f0 = False
        print("⚠️  Không có pyworld — bỏ qua metric F0 (cài extra 'f0' để đủ cổng R7-1)")

    per_clip: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in rows[:clips]:
            wav = load_wav(str(row["audio_path"]), sample_rate)
            real_mel, _ = extract_mel_energy(wav, mel_config)
            synth = generator(real_mel.unsqueeze(0).to(device))[0].float().cpu()
            write_wav(wav_dir / f"{row['id']}.wav", synth.numpy(), sample_rate)
            entry: dict[str, Any] = {"id": row["id"]}
            entry.update(mel_metrics(generator, real_mel, synth, mel_config))
            if has_f0:
                entry.update(f0_metrics(wav.numpy(), synth.numpy(), sample_rate,
                                        int(mel_config["hop_length"])))
            per_clip.append(entry)

    def _agg(name: str) -> dict[str, float]:
        values = [e[name] for e in per_clip if name in e and np.isfinite(e[name])]
        if not values:
            return {}
        return {"mean": float(np.mean(values)), "median": float(np.median(values)), "p90": float(np.percentile(values, 90))}

    aggregate = {
        "checkpoint": str(checkpoint),
        "clips": len(per_clip),
        "mel_space": "vocos_raw" if hasattr(generator, "loss_mel_config") and generator.__class__.__name__ == "VocosGenerator" else "config",
        "metrics": {
            key: _agg(key)
            for key in ("mel_l1", "clipping_ratio", "length_ratio", "logf0_rmse_cents",
                        "logf0_rmse_cents_no_oct", "octave_rate", "err_lt25_frac", "vuv_err", "vuv_err_unvoiced")
        },
        "per_clip": per_clip,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = aggregate["metrics"]
    print(f"[eval] {len(per_clip)} clip → {out_dir}")
    for key in ("mel_l1", "logf0_rmse_cents_no_oct", "vuv_err", "clipping_ratio"):
        stats = summary.get(key)
        if stats:  # rỗng khi không có frame hữu thanh (ví dụ audio test thuần âm)
            print(f"  {key:26s} mean={stats['mean']:.4f} median={stats['median']:.4f}")
    return aggregate


def main() -> None:
    """Đọc tham số, chạy evaluate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt (hoặc pytorch_model.bin HF)")
    parser.add_argument("--manifest", required=True, help="Manifest JSONL tập eval (eval_clips_v1)")
    parser.add_argument("--output", required=True, help="Thư mục kết quả (wavs + metrics.json)")
    parser.add_argument("--clips", type=int, default=50)
    parser.add_argument("--device", default="cpu", help="cpu (mặc định, không tranh MPS) | mps | cuda")
    parser.add_argument("--sample-rate", type=int, default=None)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("Manifest eval rỗng")
    evaluate(config, args.checkpoint, rows, args.output, clips=args.clips,
             device=args.device, sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()

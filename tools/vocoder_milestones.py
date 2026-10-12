"""Nghe thử sớm: xuất copy-synthesis ~10 câu val tại các mốc bước (5k/20k/50k...).

Chạy nền song song với training (dùng **CPU** để không tranh MPS):

    python tools/vocoder_milestones.py --config configs/small_codelung.yaml \
        --checkpoint-dir runs/vocoder_codelung_v2 --manifest data/codelung/val.jsonl \
        --output artifacts/vocoder_demo --steps 5000,20000,50000

Mỗi mốc ghi `<output>/step_<N>/` gồm WAV copy-synthesis + `metrics.json`
(mel L1 so với mel thật đã chuẩn hoá, tỉ lệ clipping, tỉ lệ độ dài) để so nhanh;
nghe 10 câu là chính (REVISION_PLAN_V2 rev.6 §2: mốc 20k còn rè kim loại/lỗi pha → dừng sửa).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def select_clips(rows: list[dict[str, Any]], clips: int) -> list[dict[str, Any]]:
    """Chọn `clips` mẫu trải đều theo danh sách (tất định)."""
    if clips <= 0 or not rows:
        return []
    stride = max(1, len(rows) // clips)
    picked = rows[::stride][:clips]
    return picked


def load_generator(config: dict[str, Any], checkpoint: str | Path, device: str = "cpu"):
    """Nạp generator (lite/vocos) từ checkpoint; chỉ dùng checkpoint đã kiểm sạch."""
    import torch

    from vifasttts.data.features import canonical_mel_config
    from vifasttts.models.istftnet import build_generator

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("clean") is False:
        raise ValueError(f"{checkpoint} không sạch (clean=false) — bỏ qua")
    generator = build_generator(config.get("vocoder_training", {}), canonical_mel_config(config))
    generator.load_state_dict(state.get("vocoder", state))
    generator.to(torch.device(device))
    generator.eval()
    return generator


def copy_synthesis(generator: Any, mel: Any) -> Any:
    """mel [T, M] → waveform [L] (không grad)."""
    import torch

    with torch.no_grad():
        wave = generator(mel.unsqueeze(0))[0]
    return wave


def _metrics(real_mel: Any, synth_wav: Any, mel_config: dict[str, Any]) -> dict[str, float]:
    """mel L1 (mel tái trích của bản synth vs mel thật) + clipping + tỉ lệ độ dài."""

    from vifasttts.data.features import extract_mel_energy

    synth_mel, _ = extract_mel_energy(synth_wav.cpu(), mel_config)
    frames = min(synth_mel.size(0), real_mel.size(0))
    mel_l1 = float((synth_mel[:frames] - real_mel[:frames]).abs().mean()) if frames else float("inf")
    clipping = float((synth_wav.abs() > 0.985).float().mean())
    length_ratio = float(synth_wav.numel()) / max(1, real_mel.size(0) * int(mel_config["hop_length"]))
    return {
        "mel_l1": round(mel_l1, 4),
        "clipping_ratio": round(clipping, 5),
        "length_ratio": round(length_ratio, 4),
        "frames": frames,
    }


def run_milestone(config: dict[str, Any], rows: list[dict[str, Any]], checkpoint: str | Path,
                  step: int, output: str | Path, clips: int = 10, sample_rate: int = 24000) -> dict[str, Any]:
    """Xuất copy-synthesis cho một mốc; trả dict metric tổng hợp."""

    from vifasttts.data.features import canonical_mel_config, extract_mel_energy, load_wav
    from vifasttts.infer import write_wav

    mel_config = canonical_mel_config(config)
    generator = load_generator(config, checkpoint)
    out_dir = Path(output) / f"step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for row in select_clips(rows, clips):
        wav = load_wav(str(row["audio_path"]), sample_rate)
        real_mel, _ = extract_mel_energy(wav, mel_config)
        synth = copy_synthesis(generator, real_mel)
        wav_path = out_dir / f"{row['id']}.wav"
        write_wav(wav_path, synth.cpu().numpy(), sample_rate)
        metrics = _metrics(real_mel, synth, mel_config)
        metrics["id"] = row["id"]
        summary.append(metrics)
    aggregate = {
        "step": step,
        "checkpoint": str(checkpoint),
        "clips": len(summary),
        "mel_l1_mean": round(sum(m["mel_l1"] for m in summary) / max(1, len(summary)), 4),
        "clipping_max": round(max((m["clipping_ratio"] for m in summary), default=0.0), 5),
        "per_clip": summary,
    }
    (out_dir / "metrics.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=1), encoding="utf-8")
    return aggregate


def _read_step(checkpoint: Path) -> int | None:
    """Số bước của checkpoint sạch (None nếu chưa có/không sạch)."""
    if not checkpoint.exists():
        return None
    import torch

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - đang ghi dở
        return None
    if state.get("clean") is False:
        return None
    return int(state.get("step", 0))


def main() -> None:
    """Vòng theo dõi checkpoint → xuất demo tại các mốc (mặc định 5k/20k/50k)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--manifest", required=True, help="Manifest val (JSONL)")
    parser.add_argument("--output", required=True, help="Thư mục artifacts demo")
    parser.add_argument("--steps", default="5000,20000,50000", help="Các mốc bước, phẩy phân cách")
    parser.add_argument("--clips", type=int, default=10, help="Số câu val mỗi mốc")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--poll", type=float, default=60.0, help="Chu kỳ kiểm checkpoint (giây)")
    parser.add_argument("--once", action="store_true", help="Xử lý các mốc đã đủ rồi thoát (test)")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    milestones = [int(value) for value in str(args.steps).split(",") if value.strip()]
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    done_path = output / "milestones.json"
    done: list[int] = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else []

    while True:
        step = _read_step(Path(args.checkpoint_dir) / "last.pt")
        if step is not None:
            for milestone in milestones:
                if milestone <= step and milestone not in done:
                    print(f"[demo] xuất copy-synthesis mốc {milestone} (step hiện tại {step})", flush=True)
                    aggregate = run_milestone(config, rows, Path(args.checkpoint_dir) / "last.pt",
                                             milestone, output, clips=args.clips,
                                             sample_rate=args.sample_rate)
                    print(f"[demo] mốc {milestone}: mel_l1={aggregate['mel_l1_mean']} "
                          f"clipping_max={aggregate['clipping_max']}", flush=True)
                    done.append(milestone)
                    done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")
        if args.once or all(milestone in done for milestone in milestones):
            break
        time.sleep(max(10.0, args.poll))
    print(f"[demo] xong các mốc: {sorted(done)}", flush=True)


if __name__ == "__main__":
    main()

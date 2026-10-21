"""tools/creak_dose_response.py — kiểm tra NHÂN QUẢ "mel mờ ⇒ rè" bằng can thiệp trên mel (không train).

Vì sao (V11S3, phản biện ngoài Muse): §45.5 đo mel dự đoán vùng creak mượt hơn mel thật
−41,7% (thời gian) / −61,8% (tần số) rồi suy ra Vocos render thành rè — nhưng §39 đã cho thấy
Vocos thêm rung cả với mel THẬT. Trước khi tốn 8k bước train (E4c), kiểm quan hệ **liều–đáp ứng**:

    mel THẬT  → làm mờ nhân tạo (α quét) → Vocos → creak_rate   (kỳ vọng: creak TĂNG về phía 2,7)
    mel MODEL → làm nét (sharpen_mel α quét) → Vocos → creak_rate (kỳ vọng: creak GIẢM về phía 1,7)

Nếu cả hai phẳng ⇒ giả thuyết "mờ ⇒ rè" không đủ để đặt cược vào E4c.

Chạy:
    .venv/bin/python tools/creak_dose_response.py --config runs/acoustic_v2_base/resolved_config.json \
        --checkpoint runs/acoustic_v2_base/last.pt --out artifacts/creak25/dose_response.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HOP_SECONDS = 256 / 24000


def _box_smooth_time(mel, alpha: float):
    """Làm mờ theo THỜI GIAN: mel ← (1−α)·mel + α·trung bình 3 frame (biên lặp lại)."""
    import torch

    padded = torch.cat([mel[:1], mel, mel[-1:]], dim=0)
    smooth = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
    return (1.0 - alpha) * mel + alpha * smooth


def _box_smooth_freq(mel, alpha: float):
    """Làm mờ theo TẦN SỐ: mel ← (1−α)·mel + α·trung bình 3 bin (biên lặp lại)."""
    import torch

    padded = torch.cat([mel[:, :1], mel, mel[:, -1:]], dim=1)
    smooth = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3.0
    return (1.0 - alpha) * mel + alpha * smooth


def main() -> None:
    """CLI: sinh wav cho từng biến thể mel rồi đo creak (CreaPy) — in bảng liều–đáp ứng."""
    import subprocess

    import torch

    from vifasttts.data.cache import FeatureCache
    from vifasttts.frontend.pipeline import parse_text
    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.infer import write_wav
    from vifasttts.models.factory import build_model, load_model_state
    from vifasttts.models.istftnet import build_generator
    from vifasttts.runtime.postprocess import sharpen_mel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="checkpoint acoustic (lấy mel dự đoán)")
    parser.add_argument("--clips", default="artifacts/creak25/clips25.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--blur-alphas", default="0.3,0.5,0.7")
    parser.add_argument("--sharp-alphas", default="1.5,3.0")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cache = FeatureCache("cache/codelung_v2", config)
    f0_stats = json.loads(Path("data/codelung/f0_stats_v2.json").read_text(encoding="utf-8"))
    clips = json.loads(Path(args.clips).read_text(encoding="utf-8"))
    device = torch.device(args.device)

    vocab = ComponentVocabs.default()
    sizes = [len(vocab.onset), len(vocab.medial), len(vocab.nucleus), len(vocab.coda),
             len(vocab.tone), len(vocab.boundary), len(vocab.punctuation)]
    model = build_model(config, sizes, arch="v2_slot5")
    load_model_state(model, torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"])
    model.load_f0_stats(f0_stats)
    model.to(device).eval()
    vocoder = build_generator(config.get("vocoder_training", {}), config["mel"]).to(device).eval()

    variants: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for entry in clips:
            row = entry["row"]
            features = cache.get(row)
            start, end = row["frame_range"]
            real = features["mel"][start:end].float()
            items = parse_text(row["text"])
            if not items:
                continue
            ids = torch.tensor([[vocab.encode(item) for item in items]], dtype=torch.long, device=device)
            n_syl = len(row["durations"])
            gaps = list(row.get("gaps") or [])[:n_syl]
            gaps = gaps + [0] * (n_syl - len(gaps))
            with torch.no_grad():
                outputs = model(ids, durations=torch.tensor([row["durations"]], device=device),
                                seg_durations=torch.tensor([row["seg_durations"]], device=device),
                                gaps=torch.tensor([gaps], device=device),
                                lead=torch.tensor([row.get("lead_frames", 0)], device=device))
            pred = outputs["mel"][0].cpu().float()
            length = min(pred.size(0), real.size(0))
            variants.setdefault("that", {})[entry["parent"]] = real[:length]
            variants.setdefault("model", {})[entry["parent"]] = pred[:length]
            for alpha in (float(a) for a in args.blur_alphas.split(",")):
                blurred = _box_smooth_freq(_box_smooth_time(real[:length], alpha), alpha)
                variants.setdefault(f"that_mo_a{alpha}", {})[entry["parent"]] = blurred
            for alpha in (float(a) for a in args.sharp_alphas.split(",")):
                variants.setdefault(f"model_net_a{alpha}", {})[entry["parent"]] = \
                    sharpen_mel(pred[:length], alpha)

    base_dir = Path(args.out).parent / "dose_response"
    results: dict[str, dict] = {}
    for name, per_clip in variants.items():
        out_dir = base_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for parent, mel in per_clip.items():
                waveform = vocoder(mel.unsqueeze(0).to(device))[0]
                write_wav(out_dir / f"{parent}.wav", waveform.detach().cpu(),
                          int(config.get("sample_rate", 24000)))
        # CreaPy (GPL, cài ngoài) cần venv riêng ⇒ đo qua subprocess `creak_rate.py`.
        creapy_python = str(Path("~/creapy-venv/bin/python").expanduser())
        report_path = out_dir.parent / f"{name}_creak.json"
        subprocess.run([creapy_python, "tools/creak_rate.py", "--clips", args.clips,
                        "--synth-dir", str(out_dir), "--out", str(report_path)],
                       check=True, capture_output=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        results[name] = {"ratio": report["ratio"], "groups": report["groups"]}
        print(f"{name:<18} ratio {report['ratio']:.3f} · ngã+nặng "
              f"{report['groups']['ngã+nặng']['mean']:.4f} · ngang+huyền "
              f"{report['groups']['ngang+huyền']['mean']:.4f}", flush=True)

    Path(args.out).write_text(json.dumps({"results": results, "clips": args.clips},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {args.out} (mốc: THẬT 1,717 · copy-synth 2,390 · free-run 2,725)")


if __name__ == "__main__":
    main()

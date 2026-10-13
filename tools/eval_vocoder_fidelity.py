"""tools/eval_vocoder_fidelity.py — đo "độ trung thành" vocoder trên **mel thật** (T5.0 nền).

Với mỗi clip val: lấy mel thật từ cache → tổng hợp audio bằng từng vocoder (Vocos pretrained,
lite đã train ở `runs/vocoder_codelung`, hoặc checkpoint khác) → trích lại mel/F0 từ audio:

- `mel_l1`: L1 giữa mel tái trích và mel đích (cùng không gian chuẩn hoá) — nhỏ = giữ phổ tốt.
- `voiced_recall`: tỷ lệ frame **đích hữu thanh** (`vuv>0,5`) mà audio tái tạo cũng hữu thanh —
  cao = không rụng thanh (lỗi đã thấy ở T3.3 khi nghe ngã/nặng).
- `f0_cents`: RMSE cents trên các frame cùng hữu thanh (nhỏ = giữ cao độ).
- `rtf`: giây suy luận / giây audio + số tham số (so chi phí).

Chạy:
    .venv/bin/python tools/eval_vocoder_fidelity.py \
        --config configs/small_codelung_v2.yaml \
        --manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
        --cache-dir cache/codelung_v2 --clips 30 --device cpu \
        --vocoder pretrained= --vocoder lite=runs/vocoder_codelung/last.pt \
        --out artifacts/eval_vocoder/fidelity_v1.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vifasttts.data.cache import FeatureCache
from vifasttts.data.dataset import ManifestDataset
from vifasttts.data.features import estimate_f0_pyworld, extract_mel_energy


def _mel_denormalize_stats(config: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
    """mean/std per-bin từ `mel_stats.json` (để tự denormalize khi vocoder dùng mel THÔ)."""
    normalize_path = config.get("mel", {}).get("normalize_path")
    if not normalize_path:
        return None
    import json

    stats = json.loads(Path(normalize_path).read_text(encoding="utf-8"))
    return (torch.tensor(stats["mean"], dtype=torch.float32),
            torch.tensor(stats["std"], dtype=torch.float32).clamp_min(1e-6))


def _vocoder(config: dict, checkpoint: str | None, device: str,
             contract: str = "norm", vt_override: dict | None = None):
    """Dựng generator; `contract="raw"` → vocoder cũ dùng mel **thô** (bỏ normalize_path khi build)."""
    from vifasttts.models.istftnet import build_generator

    vt = {**dict(config.get("vocoder_training", {})), **(vt_override or {})}
    mel_config = dict(config["mel"])
    if contract == "raw":
        mel_config.pop("normalize_path", None)
    generator = build_generator(vt, mel_config)
    path = checkpoint or vt.get("pretrained_path")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        generator.load_state_dict(state.get("vocoder", state))
    elif path:
        state = torch.load(path, map_location="cpu", weights_only=False)
        generator.load_state_dict(state.get("vocoder", state), strict=False)
    generator.to(device).eval()
    parameters = sum(p.numel() for p in generator.parameters())
    return generator, parameters


def _measure(generator, mel: torch.Tensor, target_mel: torch.Tensor, config: dict,
             sample_rate: int, hop: int, device: str) -> dict:
    """Tổng hợp 1 clip và đo mel/F0 so đích."""
    with torch.no_grad():
        start = time.perf_counter()
        waveform = generator(mel.unsqueeze(0).to(device))[0].detach().cpu().numpy()
        elapsed = time.perf_counter() - start
    audio = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    recovered, _ = extract_mel_energy(audio, config["mel"])
    frames = min(recovered.size(0), target_mel.size(0))
    mel_l1 = float((recovered[:frames] - target_mel[:frames]).abs().mean())
    f0, vuv = estimate_f0_pyworld(audio, sample_rate, hop, f_min=70, f_max=400, interpolate=False)
    return {"mel_l1": mel_l1, "f0": f0, "vuv": (vuv > 0.5).float(), "seconds": elapsed}


def main() -> None:
    """Chạy so sánh; in bảng + ghi JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--clips", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocoder", action="append", required=True,
                        help="TÊN=đường_dẫn (để trống = pretrained trong config), lặp được nhiều lần")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sample_rate = int(config.get("sample_rate", 24000))
    hop = int(config["mel"]["hop_length"])
    rows = ManifestDataset(args.manifest).rows[: args.clips]
    cache = FeatureCache(args.cache_dir, config) if args.cache_dir else None
    audio_total = 0.0
    report: dict = {"config": args.config, "clips": len(rows), "vocoders": {}}
    specs = []
    for spec in args.vocoder:  # "TÊN=đường_dẫn[#raw][@gen:lite,hidden:192]"
        name, _, rest = spec.partition("=")
        contract = "raw" if "#raw" in rest else "norm"
        rest = rest.replace("#raw", "")
        if "@" in rest:
            path_part, vt_part = rest.split("@", 1)
        else:
            path_part, vt_part = rest, ""
        vt_override = {}
        for piece in filter(None, vt_part.split(",")):
            key, _, value = piece.partition(":")
            vt_override[key] = int(value) if value.isdigit() else value
        specs.append((name, path_part or None, contract, vt_override))

    extracted = {}
    for row in rows:
        features = cache.get(row) if cache else None
        if features is None:
            from vifasttts.data.features import extract_features

            features = extract_features(row["audio_path"], config)
        start = int(row["frame_range"][0])
        frames = int(row["frame_range"][1]) - start
        extracted[row["id"]] = features["mel"][start:start + frames]

    denorm = _mel_denormalize_stats(config)
    for name, path, contract, vt_override in specs:
        generator, parameters = _vocoder(config, path, args.device, contract, vt_override)
        totals = {"mel_l1": [], "voiced_recall": [], "f0_cents": [], "rtf": []}
        for row in rows:
            target_mel = extracted[row["id"]]
            feed = target_mel
            if contract == "raw" and denorm is not None:  # vocoder cũ: denormalize mel về thô
                mean, std = denorm
                feed = target_mel * std.view(1, -1) + mean.view(1, -1)
            result = _measure(generator, feed, target_mel, config, sample_rate, hop, args.device)
            features = extracted[row["id"]]
            # đích: f0/vuv thật của row (qua cache)
            features_all = cache.get(row) if cache else None
            if features_all is None:
                from vifasttts.data.features import extract_features

                features_all = extract_features(row["audio_path"], config)
            start = int(row["frame_range"][0])
            frames = int(row["frame_range"][1]) - start
            target_f0 = features_all["f0"][start:start + frames]
            target_vuv = (features_all["vuv"][start:start + frames] > 0.5)
            n = min(result["f0"].size(0), frames)
            predicted_vuv = result["vuv"][:n] > 0.5
            voiced = target_vuv[:n] & predicted_vuv
            recall = float(predicted_vuv[target_vuv[:n]].float().mean()) if target_vuv[:n].any() else 1.0
            cents = float("nan")
            if int(voiced.sum()) >= 4:
                ratio = result["f0"][:n][voiced] / target_f0[:n][voiced].clamp_min(1e-6)
                cents = float((torch.log2(ratio.clamp_min(1e-6)).abs() * 1200).mean())
            totals["mel_l1"].append(result["mel_l1"])
            totals["voiced_recall"].append(recall)
            if not np.isnan(cents):
                totals["f0_cents"].append(cents)
            audio_seconds = n * hop / sample_rate
            totals["rtf"].append(result["seconds"] / max(audio_seconds, 1e-6))
            audio_total += audio_seconds
        summary = {
            "parameters": parameters,
            "mel_l1": round(float(np.mean(totals["mel_l1"])), 5),
            "voiced_recall": round(float(np.mean(totals["voiced_recall"])), 4),
            "f0_cents_mean": round(float(np.mean(totals["f0_cents"])), 1) if totals["f0_cents"] else None,
            "rtf_audio_seconds": round(float(np.mean(totals["rtf"])), 4),
            "checkpoint": path or "(pretrained)",
        }
        report["vocoders"][name] = summary
        print(f"{name:10s} mel_l1 {summary['mel_l1']:.4f} · voiced_recall {summary['voiced_recall']:.3f} "
              f"· f0 {summary['f0_cents_mean']} cents · RTF {summary['rtf_audio_seconds']:.3f} "
              f"· {parameters / 1e6:.1f}M tham số")
        del generator

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()

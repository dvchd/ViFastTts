"""tools/v1827_domain_probe.py — v18.27 (V1+V3): kiểm **miền mel** và **mốc copy-synth**.

Vì sao: A1–A3 (v18.25) so mel P2 `*std+mean` (z→log bằng `mel_stats.json`) với mel trích
bằng extractor của Vocos (log ln, sàn 1e-7) ⇒ **hai miền khác nhau**, kết luận (i)/(ii) sai.

Lượt này đo lại **trong cùng miền**:
- V1: tensor production đưa vào Vocos (model mel → `sharpen_mel(α=0)`) so với cache GTA;
  Vocos nhận z hay log (đọc code + số); mốc copy-synth tính bằng **đúng hàm loss lúc train**
  (`loss_mel_config` = log thô, KHÔNG normalize) trên 300 row val.
- V3: giả thuyết sàn trong **miền acoustic z** (mel P2 TF vs mel thật cùng extractor + cùng
  chuẩn hoá): min/p1/p5 theo bin, L1 tách frame lặng (RMS copy < −50 dBFS) / có tiếng,
  và đường cong dịch k ∈ [−10..10].

JSON ghi **trước** khi in (`artifacts/bench/v1827_domain.json`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

HOP = 256
BINS = (0, 25, 50, 75, 99)
SILENT_DBFS = -50.0


def _read_crop(row: dict, frames: int) -> torch.Tensor:
    """Đọc đúng `frames*hop` mẫu bắt đầu ở `frame_range[0]` (seek, không nạp cả clip)."""
    start = int(row["frame_range"][0])
    with sf.SoundFile(row["audio_path"]) as handle:
        handle.seek(start * HOP)
        audio = handle.read(frames * HOP, dtype="float32", always_2d=False)
    return torch.from_numpy(np.ascontiguousarray(audio))


def _percentiles(tensor: torch.Tensor, bins=BINS) -> dict:
    """min/p1/p5 theo từng bin (gộp mọi frame của mọi row)."""
    out = {}
    for b in bins:
        col = tensor[..., b].flatten()
        out[str(b)] = {
            "min": round(float(col.min()), 3),
            "p1": round(float(torch.quantile(col, 0.01)), 3),
            "p5": round(float(torch.quantile(col, 0.05)), 3),
        }
    out["all"] = {
        "min": round(float(tensor.min()), 3),
        "p1": round(float(torch.quantile(tensor.flatten(), 0.01)), 3),
        "p5": round(float(torch.quantile(tensor.flatten(), 0.05)), 3),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small_codelung_v32.yaml")
    parser.add_argument("--gta-manifest", default="/private/tmp/opencode/val300_manifest.jsonl")
    parser.add_argument("--gta-cache", default="/private/tmp/opencode/val300_gta")
    parser.add_argument("--pretrained", default="artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin")
    parser.add_argument("--rows", type=int, default=0, help="0 = toàn bộ manifest con")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default="artifacts/bench/v1827_domain.json")
    args = parser.parse_args()

    from vifasttts.data.features import extract_mel_energy_batched
    from vifasttts.models.istftnet import build_generator
    from vifasttts.models.vocos import load_vocos_pretrained
    from vifasttts.runtime.postprocess import sharpen_mel

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    z_cfg = dict(config["mel"])  # có normalize_path → z-score
    raw_cfg = {k: v for k, v in z_cfg.items() if k != "normalize_path"}  # log thô (miền loss)

    rows = [json.loads(line) for line in Path(args.gta_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.rows:
        rows = rows[: args.rows]
    cache = Path(args.gta_cache)

    device = torch.device(args.device)
    generator = build_generator(config["vocoder_training"], config["mel"])
    load_vocos_pretrained(generator, args.pretrained)
    generator.to(device).eval()
    # Hàm loss mel của vocoder = log thô, KHÔNG z (VocosGenerator.loss_mel_config) — dùng đúng nó.
    loss_cfg = generator.loss_mel_config(config["mel"])

    mel_p2, mel_real_z, mel_real_raw, cs_raw, cs_z, silent = [], [], [], [], [], []
    per_row = []
    with torch.no_grad():
        for row in rows:
            payload = torch.load(cache / f"{row['id']}.pt", map_location="cpu", weights_only=False)
            mel_model = payload["mel"].float()
            frames = int(payload["frames"])
            audio = _read_crop(row, frames).unsqueeze(0).to(device)
            real_z, _ = extract_mel_energy_batched(audio, z_cfg)
            real_raw, _ = extract_mel_energy_batched(audio, raw_cfg)
            copy_audio = generator(sharpen_mel(real_z.to(device), 0.0))
            copy_raw, _ = extract_mel_energy_batched(copy_audio[..., : frames * HOP], loss_cfg)
            copy_z, _ = extract_mel_energy_batched(copy_audio[..., : frames * HOP], z_cfg)
            n = min(frames, copy_audio.size(1) // HOP, real_z.size(1), copy_raw.size(1), copy_z.size(1))
            model = mel_model[:n]
            rz, rr = real_z[0, :n], real_raw[0, :n]
            cz, cr = copy_z[0, :n], copy_raw[0, :n]
            # frame lặng theo RMS của **copy-synth** (< −50 dBFS)
            rms = copy_audio[0, : n * HOP].reshape(n, HOP).pow(2).mean(dim=1).clamp_min(1e-24).sqrt()
            db = 20.0 * torch.log10(rms)
            sil = db < SILENT_DBFS
            mel_p2.append(model.cpu())
            mel_real_z.append(rz.cpu())
            mel_real_raw.append(rr.cpu())
            cs_raw.append(cr.cpu())
            cs_z.append(cz.cpu())
            silent.append(sil.cpu())
            per_row.append(
                {
                    "id": str(row["id"]),
                    "frames": n,
                    "l1_dom_z": round(float((model - rz.cpu()).abs().mean()), 3),
                    "l1_dom_raw": round(float((model - rr.cpu()).abs().mean()), 3),
                }
            )
        if str(device).startswith("mps"):
            torch.mps.empty_cache()

    P = torch.cat(mel_p2)
    Rz = torch.cat(mel_real_z)
    Rr = torch.cat(mel_real_raw)
    CR = torch.cat(cs_raw)
    CZ = torch.cat(cs_z)
    S = torch.cat(silent)

    diff = (P - Rz).abs()
    result: dict = {
        "config": args.config,
        "rows": len(rows),
        "frames": int(P.size(0)),
        "device": args.device,
        "mel_normalized_vocoder": bool(getattr(generator, "mel_normalized", False)),
        "loss_cfg_has_normalize_path": "normalize_path" in loss_cfg,
        "silent_ratio": round(float(S.float().mean()), 4),
        # V1.1: cache GTA = output model (build_gta_mel lưu `outputs["mel"]` nguyên vẹn);
        # production truyền `sharpen_mel(mel, 0)` = chính tensor đó (α=0 trả về nguyên object).
        "v11_cache_equals_model_output": True,
        "v11_sharpen_identity_maxdiff": float(
            (sharpen_mel(torch.zeros(2, 3, 4) + 0.5, 0.0) - (torch.zeros(2, 3, 4) + 0.5)).abs().max()
        ),
        # V1.2: miền dữ liệu — mel model gần miền z hay miền log thô?
        "v12_domain_l1_z": round(float(diff.mean()), 3),
        "v12_domain_l1_raw": round(float((P - Rr).abs().mean()), 3),
        "v12_real_z_vs_raw": round(float((Rz - Rr).abs().mean()), 3),
        # V1.3: mốc copy-synth tính bằng ĐÚNG hàm loss lúc train (log thô) và bản z để so
        "v13_copy_floor_raw": round(float((CR - Rr).abs().mean()), 3),
        "v13_copy_floor_z": round(float((CZ - Rz).abs().mean()), 3),
        "v13_copy_floor_raw_per_row_median": round(float(np.median([
            float((c - r).abs().mean()) for c, r in zip(cs_raw, mel_real_raw)
        ])), 3),
        # V3: giả thuyết sàn trong miền acoustic z
        "v3_min_pct_model_z": _percentiles(P),
        "v3_min_pct_real_z": _percentiles(Rz),
        "v3_l1": {
            "all": round(float(diff.mean()), 3),
            "voiced": round(float(diff[~S].mean()), 3) if bool((~S).any()) else None,
            "silent": round(float(diff[S].mean()), 3) if bool(S.any()) else None,
            "band0_30": round(float(diff[:, :30].mean()), 3),
            "band30_60": round(float(diff[:, 30:60].mean()), 3),
            "band60_100": round(float(diff[:, 60:].mean()), 3),
        },
        "v3_example_rows": per_row[:3],
    }

    # B mở rộng: dịch k ∈ [−10..10] CHỈ trên frame có tiếng, đã trừ mean từng bin.
    Pm = P - P.mean(dim=0, keepdim=True)
    Rm = Rz - Rz.mean(dim=0, keepdim=True)
    curve = {}
    for k in range(-10, 11):
        if k < 0:
            a, b, v = Pm[:k], Rm[-k:], (~S)[:k]
        elif k > 0:
            a, b, v = Pm[k:], Rm[:-k], (~S)[k:]
        else:
            a, b, v = Pm, Rm, ~S
        curve[str(k)] = round(float((a[v] - b[v]).abs().mean()), 4)
    result["v3b_shift_curve"] = curve
    result["v3b_shift_argmin"] = min(curve, key=curve.get)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

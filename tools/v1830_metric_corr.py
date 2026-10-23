"""tools/v1830_metric_corr.py — v18.30 (B1): thước nào khớp với tai?

Dữ liệu: 45 ô = 9 câu × 5 bản của trang v1828 (điểm rè 1–5 của người dùng).
Mỗi ô render lại **TF** (α=0, mức tự nhiên) rồi đo:
  flat_V (2–6 kHz) · flat 2–8 kHz · HNR · S1 (syllable MAE vs GT) · N1 bin < −50 (Δ RMS median/energy)
  · mức năng lượng **giữa các hài** so với copy (1–5 kHz).
Correlation: Spearman với điểm rè, **gộp 45 ô** và **trong từng câu** (rank 5 bản), báo |rho|.

JSON ghi trước khi in (`artifacts/bench/v1830_metric_corr.json`).
"""

from __future__ import annotations

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
sys.path.insert(0, str(_ROOT / "tools"))

import v1828_gta_eval as E

HOP, SR, NFFT = E.HOP, E.SR, E.NFFT
CFG35 = "configs/small_codelung_v35.yaml"
CFG32 = "configs/small_codelung_v32.yaml"


def spec_frames(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = int(len(wave) // HOP)
    padded = np.pad(wave, (0, max(0, (count - 1) * HOP + NFFT - len(wave))))
    frames = np.stack([padded[i * HOP:i * HOP + NFFT] for i in range(count)]) if count else np.zeros((0, NFFT))
    spec = np.abs(np.fft.rfft(frames * np.hanning(NFFT), axis=1))
    return spec, np.fft.rfftfreq(NFFT, 1.0 / SR)


def flat_band(spec: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    band = (freqs >= lo) & (freqs <= hi)
    p = spec[:, band] ** 2 + 1e-12
    return np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)


def noise_energy(spec: np.ndarray, freqs: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """Năng lượng giữa các hài (1–5 kHz) từng frame (dB) — dùng f0 của chính frame."""
    out = np.full(spec.shape[0], np.nan, dtype=np.float64)
    band = (freqs >= 1000) & (freqs <= 5000)
    for index in range(spec.shape[0]):
        value = float(f0[index])
        if not 50.0 <= value <= 500.0:
            continue
        row = spec[index]
        total = float(np.sum(row[band] ** 2))
        half = 0.12 * value
        harmonic = 0.0
        k = 1
        while k * value <= 5000:
            harmonic += float(np.sum(row[(freqs >= k * value - half) & (freqs <= k * value + half) & band] ** 2))
            k += 1
        out[index] = 10.0 * np.log10(max(total - harmonic, 1e-12))
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[mask])).astype(float)
    rb = np.argsort(np.argsort(b[mask])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    from vifasttts.data.cache import FeatureCache
    from vifasttts.frontend.pipeline import parse_text
    from vifasttts.infer import Synthesizer
    from vifasttts.runtime.postprocess import sharpen_mel

    key = json.loads(Path("artifacts/listen_small_filt/v1828_gate/key.json").read_text(encoding="utf-8"))
    scores = json.loads(Path("/private/tmp/opencode/uploads/fff0f42a-57a8-4ee6-84bd-fbc94a703585/v1828_scores.json").read_text(encoding="utf-8"))
    rows = {json.loads(x)["id"]: json.loads(x) for x in
            Path("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    cfg35 = yaml.safe_load(Path(CFG35).read_text(encoding="utf-8"))
    cfg32 = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    cache35 = FeatureCache("cache/codelung_v35", cfg35)
    copy_engine = Synthesizer(cfg35, E.V35, device="cpu")
    p2_engine = Synthesizer(cfg32, E.P2, device="cpu")
    gta_engine = Synthesizer(cfg32, E.P2, vocoder_checkpoint=f"{E.GTA_DIR}/gta_4k.pt", device="cpu")

    cells = []
    for sid, item in key["items"].items():
        row = rows[item["row"]]
        start, end = row["frame_range"]
        gt, _ = sf.read(row["audio_path"], dtype="float32", start=start * HOP, frames=(end - start) * HOP)
        gt = np.asarray(gt, dtype=np.float32)
        seg_index = E.seg_index_of(row)
        text = " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])
        items = parse_text(text, resolve_oov=True)
        feats = torch.tensor([[p2_engine.vocabs.encode(it) for it in items]], dtype=torch.long)
        gaps = list(row["gaps"]) + [0] if len(row["gaps"]) == len(row["durations"]) - 1 else list(row["gaps"])
        with torch.no_grad():
            out = p2_engine.model(feats, speed=torch.tensor([1.0]),
                                  durations=torch.tensor([row["durations"]], dtype=torch.long),
                                  gaps=torch.tensor([gaps], dtype=torch.long),
                                  lead=torch.tensor([0], dtype=torch.long))
            mel_model = out["mel"][0]
        mel_real = cache35.get(row)["mel"][start:end].to(copy_engine.device)
        with torch.no_grad():
            copy_audio = np.asarray(copy_engine.vocoder(sharpen_mel(mel_real, 0.0).unsqueeze(0))[0].float().cpu()).reshape(-1)
            p2_audio = np.asarray(p2_engine.vocoder(sharpen_mel(mel_model, 0.0).unsqueeze(0))[0].float().cpu()).reshape(-1)
            gta_audio = np.asarray(gta_engine.vocoder(sharpen_mel(mel_model, 0.0).unsqueeze(0))[0].float().cpu()).reshape(-1)
        audio_by_role = {
            "copy": copy_audio,
            "p2": p2_audio,
            "p2_exp45": E.expander(p2_audio, -45.0, 2.0),
            "gta": gta_audio,
            "gta_exp45": E.expander(gta_audio, -45.0, 2.0),
        }
        ref = E.frame_analysis(copy_audio)
        ref_spec, freqs = spec_frames(copy_audio)
        ref_f0 = f0_series(copy_audio)
        ref_noise = noise_energy(ref_spec, freqs, ref_f0)
        ref_levels = E.syllable_levels(copy_audio, seg_index)
        for entry in item["entries"]:
            role = entry["role"]
            score = scores[sid][entry["label"]]["re"]
            wave = audio_by_role[role]
            metrics = E.frame_analysis(wave)
            spec, _ = spec_frames(wave)
            flat8 = flat_band(spec, freqs, 2000.0, 8000.0)
            f0 = f0_series(wave)
            noise = noise_energy(spec, freqs, f0)
            voiced = ref["voiced"] & np.isfinite(noise) & np.isfinite(ref_noise)
            ihr = float(np.nanmean(noise[voiced])) - float(np.nanmean(ref_noise[voiced])) if voiced.any() else float("nan")
            levels = E.syllable_levels(wave, seg_index)
            n = min(len(levels), len(ref_levels))
            s1 = float(np.mean(np.abs(levels[:n] - ref_levels[:n]))) if n else float("nan")
            n1 = float("nan")
            n1_energy = float("nan")
            n = min(metrics["db"].size, ref["db"].size)
            mask = ref["db"][:n] <= -50.0
            if mask.any():
                n1 = float(np.median(metrics["db"][:n][mask] - ref["db"][:n][mask]))
                n1_energy = float(10.0 * np.log10(
                    max(np.sum(10 ** (metrics["db"][:n][mask] / 10.0)), 1e-30)
                    / max(np.sum(10 ** (ref["db"][:n][mask] / 10.0)), 1e-30)))
            cells.append({
                "sid": sid, "role": role, "re": score,
                "flat_v": float(np.median(metrics["flat"][metrics["voiced"]])) if metrics["voiced"].any() else float("nan"),
                "flat_8k": float(np.median(flat8[metrics["voiced"]])) if metrics["voiced"].any() else float("nan"),
                "hnr": float(np.nanmedian(metrics["hnr"])) if np.isfinite(metrics["hnr"]).any() else float("nan"),
                "s1": s1, "n1_med": n1, "n1_energy": n1_energy, "ihr": ihr,
            })

    fields = ["flat_v", "flat_8k", "hnr", "s1", "n1_med", "n1_energy", "ihr"]
    out = {"cells": cells, "n": len(cells)}
    pooled = {}
    for field in fields:
        values = np.array([cell[field] for cell in cells], dtype=float)
        re = np.array([cell["re"] for cell in cells], dtype=float)
        per_sentence = []
        for sid in key["items"]:
            mask = np.array([cell["sid"] == sid for cell in cells])
            if mask.sum() >= 4:
                per_sentence.append(spearman(values[mask], re[mask]))
        pooled[field] = {
            "rho_pooled": round(spearman(values, re), 3),
            "rho_within_median": round(float(np.nanmedian(per_sentence)), 3),
            "n": int(np.isfinite(values).sum()),
        }
    out["pooled"] = pooled
    Path("artifacts/bench/v1830_metric_corr.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"pooled": pooled}, ensure_ascii=False))


def f0_series(wave: np.ndarray) -> np.ndarray:
    """F0 từng frame bằng autocorrelation (giống `v1828_gta_eval.frame_analysis`)."""
    count = int(len(wave) // HOP)
    padded = np.pad(wave, (0, max(0, (count - 1) * HOP + NFFT - len(wave))))
    frames = np.stack([padded[i * HOP:i * HOP + NFFT] for i in range(count)]) if count else np.zeros((0, NFFT))
    centered = frames - frames.mean(axis=1, keepdims=True)
    autoc = np.fft.irfft(np.abs(np.fft.rfft(centered, 2048, axis=1)) ** 2, axis=1)[:, :1024]
    autoc = autoc / (autoc[:, :1] + 1e-12)
    lo, hi = int(SR / 400), int(SR / 70)
    peak = autoc[:, lo:hi].argmax(axis=1) + lo
    return SR / np.maximum(peak, 1)


if __name__ == "__main__":
    main()

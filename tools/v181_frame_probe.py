"""V18.1 — M1/M2/M3: tách flatness theo loại frame + HNR + nền lặng (40 segment val).

Chuẩn: free-run, α=0, EMA; GT = WAV thật (và copy-synthesis). Chỉ số báo dạng **độ lệch so GT**.

  .venv/bin/python tools/v181_frame_probe.py --out artifacts/bench/v181_frame_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

CFG35 = "configs/small_codelung_v35.yaml"
CFG32 = "configs/small_codelung_v32.yaml"
CANDS = [
    ("v32", "runs/acoustic_v32/best.pt", CFG32),
    ("p2", "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt", CFG32),
    ("v35", "runs/acoustic_v35/best_ema.pt", CFG35),
    ("gan1k", "runs/adv_v35/eval_1000_ema.pt", CFG35),
    ("gan2k", "runs/adv_v35/eval_2k_ema.pt", CFG35),
    ("gan3k", "runs/adv_v35/eval_3000_ema.pt", CFG35),
]
SR = 24000
HOP = 256
NFFT = 1024
DB = 10.0 / np.log(10.0)


def flat_frame(spec_row: np.ndarray, freqs: np.ndarray) -> float:
    band = (freqs >= 2000) & (freqs <= 6000)
    p = spec_row[band] ** 2 + 1e-12
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def hnr_frame(spec_row: np.ndarray, freqs: np.ndarray, f0: float) -> float:
    """HNR 1–5 kHz: năng lượng tại hài (f0..5k) so với phần giữa các hài (dB)."""
    if f0 < 50 or f0 > 500:
        return float("nan")
    band = (freqs >= 1000) & (freqs <= 5000)
    total = float(np.sum(spec_row[band] ** 2))
    half = 0.12 * f0
    harmonic = 0.0
    k = 1
    while k * f0 <= 5000:
        mask = (freqs >= k * f0 - half) & (freqs <= k * f0 + half) & band
        harmonic += float(np.sum(spec_row[mask] ** 2))
        k += 1
    noise = max(total - harmonic, 1e-12)
    return float(10 * np.log10(max(harmonic, 1e-12) / noise))


def periodicity(frame: np.ndarray) -> tuple[float, float]:
    """(độ tuần hoàn 0–1, f0 Hz) bằng autocorrelation — dùng CHUNG cho thật và model."""
    frame = frame - frame.mean()
    spec = np.fft.rfft(frame, 2048)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:1024]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(SR / 400), int(SR / 70)
    peak = int(np.argmax(ac[lo:hi]) + lo)
    return float(ac[peak]), float(SR / peak)


def frame_metrics(wave: np.ndarray, energy_db: np.ndarray) -> dict:
    frames = int(len(wave) // HOP)
    count = min(frames, len(energy_db))
    window = np.hanning(NFFT)
    padded = np.pad(wave, (0, max(0, (count - 1) * HOP + NFFT - len(wave))))
    spec = np.abs(np.fft.rfft(
        np.stack([padded[i * HOP:i * HOP + NFFT] * window for i in range(count)]), axis=1))
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    silence = energy_db[:count] < (np.median(energy_db[:count]) - 30.0)
    period = np.array([periodicity(padded[i * HOP:i * HOP + NFFT])[0] for i in range(count)])
    f0s = np.array([periodicity(padded[i * HOP:i * HOP + NFFT])[1] for i in range(count)])
    voiced = (period >= 0.45) & ~silence
    unvoiced = (~voiced) & ~silence
    flat = np.array([flat_frame(spec[i], freqs) for i in range(count)])
    hnr = np.array([hnr_frame(spec[i], freqs, float(f0s[i])) if voiced[i] else np.nan
                    for i in range(count)])
    rms = np.sqrt(np.mean(padded[:count * HOP].reshape(count, HOP) ** 2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    return {
        "flat_voiced": float(np.nanmedian(flat[voiced])) if voiced.any() else float("nan"),
        "flat_unvoiced": float(np.nanmedian(flat[unvoiced])) if unvoiced.any() else float("nan"),
        "hnr_voiced": float(np.nanmedian(hnr)) if np.isfinite(hnr).any() else float("nan"),
        "silence_rms": float(np.median(rms_db[silence])) if silence.any() else float("nan"),
        "frac_voiced": float(voiced.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", default="data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--segments", type=int, default=40)
    parser.add_argument("--cache", default="cache/codelung_v35")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from vifasttts.data.cache import FeatureCache
    from vifasttts.frontend.pipeline import parse_text
    from vifasttts.infer import Synthesizer
    from vifasttts.runtime.postprocess import sharpen_mel

    cfg35 = yaml.safe_load(Path(CFG35).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in Path(args.val).read_text(encoding="utf-8").splitlines()
            if line.strip()][: args.segments]
    cache = FeatureCache(args.cache, cfg35)
    stats = json.loads(Path(cfg35["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])

    def text_of(row: dict) -> str:
        return " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])

    results: dict[str, list[dict]] = {}
    # GT: WAV thật (dùng f0/vuv/năng lượng từ cache để phân loại frame)
    results["real"] = []
    for row in rows:
        start, end = row["frame_range"]
        wav, _ = sf.read(row["audio_path"], dtype="float32", start=start * HOP,
                         frames=(end - start) * HOP)
        feats = cache.get(row)
        mel = np.asarray(feats["mel"][start:end], dtype=np.float64) * std + mean
        results["real"].append(frame_metrics(wav, mel.mean(axis=1) * DB))

    # K1: copy-synthesis — mel THẬT (chuẩn hoá) → Vocos, đo "trần vocoder"
    engine_voc = Synthesizer(cfg35, "runs/acoustic_v35/best_ema.pt", device=args.device)
    results["copy"] = []
    for row in rows:
        start, end = row["frame_range"]
        feats = cache.get(row)
        mel_norm = feats["mel"][start:end].to(engine_voc.device)
        with torch.no_grad():
            audio = engine_voc.vocoder(sharpen_mel(mel_norm, 0.0).unsqueeze(0))[0]
        wav = np.asarray(audio.float().cpu()).reshape(-1)
        mel_np = np.asarray(mel_norm.float().cpu(), dtype=np.float64) * std + mean
        results["copy"].append(frame_metrics(wav, mel_np.mean(axis=1) * DB))

    for tag, ckpt, cfg_path in CANDS:
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        engine = Synthesizer(cfg, ckpt, device=args.device)
        results[tag] = []
        for row in rows:
            items = parse_text(text_of(row), resolve_oov=True)
            feats = torch.tensor([[engine.vocabs.encode(it) for it in items]],
                                 dtype=torch.long, device=engine.device)
            with torch.no_grad():
                out = engine.model(feats, speed=torch.tensor([1.0], device=engine.device),
                                   gate_threshold=1.01)
                mel = out["mel"][0]
                audio = engine.vocoder(sharpen_mel(mel, 0.0).unsqueeze(0))[0]
            wav = np.asarray(audio.float().cpu()).reshape(-1)
            mel_np = np.asarray(mel.float().cpu(), dtype=np.float64) * std + mean
            results[tag].append(frame_metrics(wav, mel_np.mean(axis=1) * DB))

    def med(tag: str, key: str) -> float:
        return float(np.nanmedian([r[key] for r in results[tag]]))

    def boot_ci(values: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
        vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        rng = np.random.default_rng(seed)
        meds = [float(np.median(rng.choice(vals, vals.size, replace=True))) for _ in range(n)]
        return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

    tags = ["real", "copy", *[c[0] for c in CANDS]]
    table = {tag: {k: round(med(tag, k), 4) for k in
                   ("flat_voiced", "flat_unvoiced", "hnr_voiced", "silence_rms")}
             for tag in tags}
    for tag in tags:
        table[tag]["frac_voiced"] = round(float(np.nanmedian(
            [r["frac_voiced"] for r in results[tag]])), 4)
        table[tag]["flat_voiced_ci"] = [round(v, 4) for v in boot_ci(
            [r["flat_voiced"] for r in results[tag]])]
        table[tag]["hnr_ci"] = [round(v, 3) for v in boot_ci(
            [r["hnr_voiced"] for r in results[tag]])]
    head = (f"{'':8s} {'flat_V':>7s} {'Δreal':>7s} {'Δcopy':>7s} {'CI95 flat_V':>15s} "
            f"{'flat_UV':>8s} {'Δreal':>7s} {'HNR':>7s} {'Δreal':>6s} {'CI95 HNR':>13s} {'%V':>6s}")
    print(head)
    for tag in tags:
        row = table[tag]
        dv = 100 * (row["flat_voiced"] / table["real"]["flat_voiced"] - 1)
        dc = 100 * (row["flat_voiced"] / table["copy"]["flat_voiced"] - 1)
        du = 100 * (row["flat_unvoiced"] / table["real"]["flat_unvoiced"] - 1)
        dh = row["hnr_voiced"] - table["real"]["hnr_voiced"]
        print(f"{tag:8s} {row['flat_voiced']:7.4f} {dv:6.1f}% {dc:6.1f}% "
              f"{row['flat_voiced_ci']!s:>15s} {row['flat_unvoiced']:8.4f} {du:6.1f}% "
              f"{row['hnr_voiced']:7.2f} {dh:6.2f} {row['hnr_ci']!s:>13s} "
              f"{100*row['frac_voiced']:5.1f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

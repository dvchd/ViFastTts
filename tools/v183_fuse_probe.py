"""V18.3 — T1/T2 (TF, nhãn GT) + E1/E2 ghép mel, đo M1/M2/spike/buzz trên 40 segment val.

Mốc xếp hạng: **copy-synth** (mel thật → Vocos). In thêm cột lệch so WAV thật.
Chạy: .venv/bin/python tools/v183_fuse_probe.py --out artifacts/bench/v183_fuse.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

SR, HOP, NFFT = 24000, 256, 1024
DB = 10.0 / np.log(10.0)
CFG35 = "configs/small_codelung_v35.yaml"
CFG32 = "configs/small_codelung_v32.yaml"
MODELS = [
    ("v32", "runs/acoustic_v32/best.pt", CFG32),
    ("p2_2k", "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt", CFG32),
    ("v35", "runs/acoustic_v35/best_ema.pt", CFG35),
    ("gan2k", "runs/adv_v35/eval_2k_ema.pt", CFG35),
    ("gan3k", "runs/adv_v35/eval_3000_ema.pt", CFG35),
]


def flat_frame(spec: np.ndarray, freqs: np.ndarray) -> float:
    band = (freqs >= 2000) & (freqs <= 6000)
    p = spec[band] ** 2 + 1e-12
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def hnr_frame(spec: np.ndarray, freqs: np.ndarray, f0: float) -> float:
    if f0 < 50 or f0 > 500:
        return float("nan")
    band = (freqs >= 1000) & (freqs <= 5000)
    total = float(np.sum(spec[band] ** 2))
    harmonic, k = 0.0, 1
    half = 0.12 * f0
    while k * f0 <= 5000:
        mask = (freqs >= k * f0 - half) & (freqs <= k * f0 + half) & band
        harmonic += float(np.sum(spec[mask] ** 2))
        k += 1
    return float(10 * np.log10(max(harmonic, 1e-12) / max(total - harmonic, 1e-12)))


def periodicity(frame: np.ndarray) -> tuple[float, float]:
    frame = frame - frame.mean()
    spec = np.fft.rfft(frame, 2048)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:1024]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(SR / 400), int(SR / 70)
    peak = int(np.argmax(ac[lo:hi]) + lo)
    return float(ac[peak]), float(SR / peak)


def metrics(wave: np.ndarray, vuv_gt: np.ndarray, seg_index: np.ndarray,
            real_level: np.ndarray, energy_gt: np.ndarray) -> dict:
    count = min(len(wave) // HOP, len(vuv_gt), len(seg_index), len(energy_gt))
    real_level = real_level[:count]
    window = np.hanning(NFFT)
    padded = np.pad(wave, (0, max(0, (count - 1) * HOP + NFFT - len(wave))))
    spec = np.abs(np.fft.rfft(np.stack(
        [padded[i * HOP:i * HOP + NFFT] * window for i in range(count)]), axis=1))
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    per = np.array([periodicity(padded[i * HOP:i * HOP + NFFT]) for i in range(count)])
    energy = energy_gt[:count]
    silence = energy < (np.median(energy) - 30.0)
    voiced_gt = (vuv_gt[:count] >= 0.5) & ~silence
    unvoiced_gt = (~(vuv_gt[:count] >= 0.5)) & ~silence
    flat = np.array([flat_frame(spec[i], freqs) for i in range(count)])
    hnr = np.array([hnr_frame(spec[i], freqs, per[i, 1]) if voiced_gt[i] else np.nan
                    for i in range(count)])
    buzz = float(np.mean(per[unvoiced_gt, 0] >= 0.45)) if unvoiced_gt.any() else float("nan")
    rms = np.sqrt(np.mean(padded[:count * HOP].reshape(count, HOP) ** 2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    levels, reals = [], []
    for syl in range(int(seg_index.max()) + 1):
        mask = seg_index[:count] == syl
        if mask.sum() >= 2:
            levels.append(float(np.mean(rms_db[mask])))
            reals.append(float(np.mean(real_level[mask])))
    levels, reals = np.asarray(levels), np.asarray(reals)
    spikes = 0
    for i, value in enumerate(levels):
        lo, hi = max(0, i - 5), min(len(levels), i + 6)
        neigh = [levels[j] for j in range(lo, hi) if j != i]
        if neigh and value - float(np.median(neigh)) > 4.0:
            spikes += 1
    return {
        "flat_v": float(np.nanmedian(flat[voiced_gt])) if voiced_gt.any() else float("nan"),
        "flat_uv": float(np.nanmedian(flat[unvoiced_gt])) if unvoiced_gt.any() else float("nan"),
        "hnr": float(np.nanmedian(hnr)) if np.isfinite(hnr).any() else float("nan"),
        "buzz": buzz,
        "spike": spikes / max(len(levels), 1),
        "mae": float(np.mean(np.abs(levels - reals))) if len(levels) else float("nan"),
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
    rows = [json.loads(x) for x in Path(args.val).read_text(encoding="utf-8").splitlines()
            if x.strip()][: args.segments]
    cache = FeatureCache(args.cache, cfg35)

    def text_of(row):
        return " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])

    def prepare(row):
        start, end = row["frame_range"]
        wav, _ = sf.read(row["audio_path"], dtype="float32", start=start * HOP,
                         frames=(end - start) * HOP)
        feats = cache.get(row)
        vuv = np.asarray(feats["vuv"][start:end])
        energy = np.asarray(feats["energy"][start:end], dtype=np.float64)
        mel_real = feats["mel"][start:end]
        seg_index = np.zeros(end - start, dtype=np.int64)
        cursor = 0
        for idx, (dur, gap) in enumerate(zip(row["durations"], list(row["gaps"]) + [0])):
            seg_index[cursor:cursor + dur] = idx
            cursor += dur + gap
        return wav, vuv, mel_real, seg_index, energy

    def real_levels(wav, seg_index):
        count = min(len(wav) // HOP, len(seg_index))
        rms = np.sqrt(np.mean(wav[:count * HOP].reshape(count, HOP) ** 2, axis=1) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)
        out = np.zeros(count)
        for syl in range(int(seg_index[:count].max()) + 1):
            mask = seg_index[:count] == syl
            out[mask] = np.mean(db[mask]) if mask.any() else db[mask].mean()
        return out

    results: dict[str, list[dict]] = {name: [] for name, *_ in MODELS}
    results["copy"] = []
    for row in rows:
        wav, vuv, mel_real, seg_index, energy = prepare(row)
        levels_real = real_levels(wav, seg_index)
        # copy-synth
        with torch.no_grad():
            eng = Synthesizer(cfg35, "runs/acoustic_v35/best_ema.pt", device=args.device) \
                if "copy_eng" not in results else results["copy_eng"]
        results["copy_eng"] = eng
        with torch.no_grad():
            audio = eng.vocoder(sharpen_mel(mel_real.to(eng.device), 0.0).unsqueeze(0))[0]
        results["copy"].append(metrics(np.asarray(audio.float().cpu()).reshape(-1), vuv,
                                       seg_index, levels_real, energy))

    def run_model(name, ckpt, cfg_path, row):
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        key = f"eng_{name}"
        if key not in results:
            results[key] = Synthesizer(cfg, ckpt, device=args.device)
        eng = results[key]
        wav, vuv, _mel_real, seg_index, _energy = prepare(row)
        items = parse_text(text_of(row), resolve_oov=True)
        feats = torch.tensor([[eng.vocabs.encode(it) for it in items]], dtype=torch.long,
                             device=eng.device)
        gaps = list(row["gaps"]) + [0] if len(row.get("gaps") or []) == len(row["durations"]) - 1 \
            else list(row["gaps"])
        with torch.no_grad():
            out = eng.model(feats, speed=torch.tensor([1.0], device=eng.device),
                            durations=torch.tensor([row["durations"]], dtype=torch.long,
                                                   device=eng.device),
                            gaps=torch.tensor([gaps], dtype=torch.long, device=eng.device),
                            lead=torch.tensor([0], dtype=torch.long, device=eng.device))
            audio = eng.vocoder(sharpen_mel(out["mel"][0], 0.0).unsqueeze(0))[0]
        wave = np.asarray(audio.float().cpu()).reshape(-1)
        return wave, out["mel"][0], vuv, seg_index, real_levels(wav, seg_index), eng

    def fuse(mel_low, mel_high, cuts: list[int], e1: bool) -> torch.Tensor:
        mel = mel_high.clone()
        if e1:
            delta = mel_low.mean(dim=1, keepdim=True) - mel_high.mean(dim=1, keepdim=True)
            mel = mel + delta
        for cut in cuts:
            fade = 6
            low = mel_low.clone()
            bins = mel.size(1)
            weight = torch.zeros(bins, device=mel.device)
            lo, hi = max(0, cut - fade), min(bins, cut + fade + 1)
            weight[:lo] = 1.0
            if hi > lo:
                weight[lo:hi] = torch.linspace(1.0, 0.0, hi - lo, device=mel.device)
            mel = mel * (1 - weight.unsqueeze(0)) + low * weight.unsqueeze(0)
        return mel

    # bin cắt theo tần số trung tâm mel
    from vifasttts.data.features import mel_filterbank
    fb = mel_filterbank(SR, NFFT, int(cfg35["mel"]["n_mels"]),
                        float(cfg35["mel"]["f_min"]), float(cfg35["mel"]["f_max"]), norm=None)
    centers = np.argmax(np.asarray(fb), axis=1) * SR / NFFT
    cut15 = int(np.searchsorted(centers, 1500))
    cut25 = int(np.searchsorted(centers, 2500))

    for name, ckpt, cfg_path in MODELS:
        for row in rows:
            wave, mel, vuv, seg_index, levels_real, _eng = run_model(name, ckpt, cfg_path, row)
            results[name].append(metrics(wave, vuv, seg_index, levels_real, energy))

    # E1/E2 trên GAN2k, GAN3k (lấy mel v3.5 cùng khung TF)
    for gan in ("gan2k", "gan3k"):
        ckpt = {n: c for n, c, _ in MODELS}.get(gan, "")
        for variant in ("E1", "E2_15", "E2_25", "E1E2_15", "E1E2_25"):
            results[f"{gan}_{variant}"] = []
        for row in rows:
            wav, vuv, _mel_real, seg_index, energy = prepare(row)
            _wave35, mel35, _, _, _, _ = run_model("v35", "runs/acoustic_v35/best_ema.pt",
                                                  CFG35, row)
            _wave_g, mel_g, _, _, levels_real, eng = run_model(gan, ckpt, CFG35, row)
            variants = {
                "E1": fuse(mel35, mel_g, [], True),
                "E2_15": fuse(mel35, mel_g, [cut15], False),
                "E2_25": fuse(mel35, mel_g, [cut25], False),
                "E1E2_15": fuse(mel35, mel_g, [cut15], True),
                "E1E2_25": fuse(mel35, mel_g, [cut25], True),
            }
            for key, mel in variants.items():
                with torch.no_grad():
                    audio = eng.vocoder(sharpen_mel(mel, 0.0).unsqueeze(0))[0]
                results[f"{gan}_{key}"].append(metrics(
                    np.asarray(audio.float().cpu()).reshape(-1), vuv, seg_index, levels_real, energy))

    def med(tag, key):
        return float(np.nanmedian([r[key] for r in results[tag]]))

    def boot(tag, key, n=1000):
        vals = np.asarray([r[key] for r in results[tag] if np.isfinite(r[key])])
        rng = np.random.default_rng(0)
        meds = [float(np.median(rng.choice(vals, vals.size, replace=True))) for _ in range(n)]
        return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

    order = ["copy", *[m[0] for m in MODELS]] + [t for t in results
                                                 if "_E" in t]
    base_v, base_h = med("copy", "flat_v"), med("copy", "hnr")
    print(f"{'':14s} {'flat_V':>7s} {'Δcopy':>7s} {'CI95':>16s} {'flat_UV':>8s} "
          f"{'HNR':>6s} {'Δcopy':>6s} {'spike':>6s} {'MAE':>5s} {'buzz':>5s}")
    for tag in order:
        if tag not in results:
            continue
        fv, hnr = med(tag, "flat_v"), med(tag, "hnr")
        ci = boot(tag, "flat_v")
        print(f"{tag:14s} {fv:7.4f} {100*(fv/base_v-1):6.1f}% "
              f"[{ci[0]:.4f},{ci[1]:.4f}] {med(tag,'flat_uv'):8.4f} {hnr:6.2f} "
              f"{hnr-base_h:6.2f} {med(tag,'spike'):6.3f} {med(tag,'mae'):5.2f} "
              f"{med(tag,'buzz'):5.3f}")
        results[tag] = {k: round(med(tag, k), 4) for k in
                        ("flat_v", "flat_uv", "hnr", "spike", "mae", "buzz")} | \
                       {"flat_v_ci": [round(v, 4) for v in ci]}
        results[tag]["n"] = len(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {k: v for k, v in results.items() if isinstance(v, dict)}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()

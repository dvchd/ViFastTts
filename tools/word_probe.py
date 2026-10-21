"""tools/word_probe.py — soi một câu/từ cụ thể theo đúng đường PRODUCTION (steady, gate 0,999).

Vì sao: bản đầu map âm tiết bằng `durations` trên mel free-run, nhưng model chèn pause (256 frame
âm tiết vs 365 mel) nên các âm tiết cuối rơi vào khoảng lặng — kết luận sai. Bản này cắt tại dấu
câu (như `iter_synthesize_steady`), mỗi chunk gọi model với `gate_threshold=0,999` (gần như không
pause) nên `durations` khớp mel.

Đo từng âm tiết: F0 (pyworld), RMS dB, spectral flatness 2–6 kHz (cao = rè), tỷ lệ năng lượng 2–6 kHz.
`--png` xuất ảnh mel + f0 + vạch âm tiết để soi "ách/ếch".

Chạy:
  .venv/bin/python tools/word_probe.py --config configs/small_codelung_v32.yaml \
      --checkpoint runs/acoustic_v32/best.pt --tag v32 --focus ách --png \
      --text "Cái ách treo trên tường đã cũ, còn con ếch thì ngồi im bên bờ ao." \
      --out artifacts/bench/word_d04_v32.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _f0_median(x: np.ndarray, sr: int) -> float:
    import pyworld as pw

    x = np.asarray(x, dtype=np.float64)
    if x.size < 1024:
        return float("nan")
    f0, t = pw.dio(x, sr, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    voiced = f0[f0 > 0]
    return float(np.median(voiced)) if voiced.size else float("nan")


def _flatness_2_6k(x: np.ndarray, sr: int, n_fft: int = 1024) -> tuple[float, float]:
    if x.size < n_fft:
        return float("nan"), float("nan")
    hop = n_fft // 4
    window = np.hanning(n_fft)
    frames = [x[i:i + n_fft] * window for i in range(0, x.size - n_fft + 1, hop)]
    if not frames:
        return float("nan"), float("nan")
    spec = np.abs(np.fft.rfft(np.stack(frames), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= 2000) & (freqs <= 6000)
    p = spec[:, band] + 1e-12
    flat = np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)
    ratio = spec[:, band].sum(axis=1) / (spec.sum(axis=1) + 1e-12)
    return float(np.mean(flat)), float(np.mean(ratio))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--text", required=True)
    parser.add_argument("--focus", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--wav-dir", default="")
    parser.add_argument("--png", default="")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    import torch

    from vifasttts.frontend.chunking import split_clauses
    from vifasttts.frontend.normalizer import NormalizerOptions
    from vifasttts.frontend.pipeline import parse_text
    from vifasttts.infer import Synthesizer, write_wav

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stats = json.loads(Path(config["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    sr = engine.sample_rate
    hop = 256
    options = NormalizerOptions(dialect="north", exact_overrides={})

    rows: list[dict] = []
    full_audio: list[np.ndarray] = []
    mel_all: list[np.ndarray] = []
    for chunk_text in split_clauses(args.text, max_chars=400):
        items = parse_text(chunk_text, options=options, resolve_oov=True)
        if not items:
            continue
        features = torch.tensor([[engine.vocabs.encode(i) for i in items]], dtype=torch.long,
                                device=engine.device)
        with torch.no_grad():
            out = engine.model(features, gate_threshold=0.999)
            mel = out["mel"][0]
            audio = np.asarray(engine.vocoder(mel.unsqueeze(0))[0].float().cpu().numpy(),
                               dtype=np.float32).reshape(-1)
        durations = [int(d) for d in np.asarray(out["durations"].cpu()).reshape(-1)]
        gates = np.asarray(out["pause_gate"].float().cpu()).reshape(-1)
        log_gaps = np.asarray(out["pause_log_gap"].float().cpu()).reshape(-1)
        gaps = np.where(gates > 0.999, np.maximum(1, np.round(np.exp(log_gaps) - 1)), 0).astype(int)
        cursor = 0
        for i, item in enumerate(items):
            dur = durations[i] if i < len(durations) else 0
            a, b = cursor * hop, min(audio.size, (cursor + dur) * hop)
            cursor += dur + int(gaps[i]) if i < len(gaps) else dur
            if b <= a:
                continue
            seg = audio[a:b]
            rows.append({"i": len(rows), "raw": getattr(item, "raw", ""),
                         "f0": round(_f0_median(seg, sr), 1),
                         "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)), 1),
                         "flat_2_6k": round(_flatness_2_6k(seg, sr)[0], 3),
                         "e_2_6k": round(_flatness_2_6k(seg, sr)[1], 3),
                         "dur_frames": dur, "mel_start": cursor - dur, "end_frame": cursor})
        full_audio.append(audio)
        mel_all.append(np.asarray(mel.float().cpu()) * std + mean)

    audio_all = np.concatenate(full_audio) if full_audio else np.zeros(1, dtype=np.float32)
    f0_vals = [r["f0"] for r in rows if np.isfinite(r["f0"])]
    median_f0 = float(np.median(f0_vals)) if f0_vals else float("nan")
    for r in rows:
        r["f0_rel"] = round(r["f0"] - median_f0, 1) if np.isfinite(r["f0"]) else None

    if args.wav_dir:
        d = Path(args.wav_dir)
        d.mkdir(parents=True, exist_ok=True)
        write_wav(d / f"{args.tag}_full.wav", audio_all, sr)
        for r in rows:
            if args.focus and r["raw"].lower() != args.focus.lower():
                continue
            start = int(r["mel_start"] * hop)
            a = max(0, start - int(0.08 * sr))
            b = min(audio_all.size, int(r["end_frame"] * hop) + int(0.08 * sr))
            write_wav(d / f"{args.tag}_{r['raw']}.wav", audio_all[a:b], sr)

    if args.png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyworld as pw

        mel_cat = np.concatenate(mel_all, axis=0)
        fig, ax = plt.subplots(figsize=(14, 3.2))
        ax.imshow(mel_cat.T, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, mel_cat.shape[0], 0, 100])
        f0, t = pw.dio(audio_all.astype(np.float64), sr, frame_period=5.0)
        f0 = pw.stonemask(audio_all.astype(np.float64), f0, t, sr)
        xs = t * sr / hop
        ax.plot(xs, (np.clip(f0, 0, 400) / 400.0) * 100, color="#4fd1ff", lw=1.0, label="f0 (0–400 Hz)")
        for r in rows:
            ax.axvline(r["end_frame"], color="w", lw=0.4, alpha=0.5)
            ax.text(r["mel_start"] + 1, 92, r["raw"], color="w", fontsize=7, rotation=90)
        ax.set_title(f"{args.tag} · {args.text}", fontsize=8)
        fig.tight_layout()
        Path(args.png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=130)
        plt.close(fig)

    result = {"tag": args.tag, "checkpoint": args.checkpoint, "text": args.text,
              "median_f0": round(median_f0, 1), "syllables": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.tag:10s} F0 median {median_f0:5.1f} Hz · {len(rows)} âm tiết")
    print(f"  {'#':>2} {'từ':10s} {'F0':>6s} {'ΔF0':>6s} {'RMS':>6s} {'flat2-6k':>9s} {'E2-6k':>6s}")
    for r in rows:
        mark = " ←" if r["raw"].lower() == args.focus.lower() else ""
        f0_txt = f"{r['f0']:6.1f}" if np.isfinite(r["f0"]) else "     -"
        d_txt = f"{r['f0_rel']:6.1f}" if r["f0_rel"] is not None else "     -"
        print(f"  {r['i']:2d} {r['raw']:10s} {f0_txt} {d_txt} {r['rms_db']:6.1f} "
              f"{r['flat_2_6k']:9.3f} {r['e_2_6k']:6.3f}{mark}")


if __name__ == "__main__":
    main()

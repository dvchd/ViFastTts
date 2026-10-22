"""tools/v185_blind.py — trang nghe MÙ v18.5: P2@2k · GAN2k · copy-synth (mốc trên).

9 câu lấy từ tập val (để copy-synth được), >=3 câu nhiều phụ âm s/x/kh/ph/ch.
Cùng −23 LUFS, id mới (v185_xx), nhãn A/B/C xáo riêng từng câu, đáp án chỉ ghi JSON.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

SR, HOP = 24000, 256


def flatness(x: np.ndarray, n_fft: int = 1024) -> float:
    if x.size < n_fft:
        return float("nan")
    hop, w = n_fft // 4, np.hanning(n_fft)
    fr = np.stack([x[i:i + n_fft] * w for i in range(0, x.size - n_fft + 1, hop)])
    sp = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n_fft, 1.0 / SR)
    band = (f >= 2000) & (f <= 6000)
    p = sp[:, band] + 1e-12
    return float(np.mean(np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", default="data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--cache", default="cache/codelung_v35")
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from vifasttts.data.cache import FeatureCache
    from vifasttts.infer import Synthesizer, write_wav
    from vifasttts.runtime.loudness import normalize_loudness
    from vifasttts.runtime.postprocess import postprocess, sharpen_mel

    cfg35 = yaml.safe_load(Path("configs/small_codelung_v35.yaml").read_text(encoding="utf-8"))
    cfg32 = yaml.safe_load(Path("configs/small_codelung_v32.yaml").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in Path(args.val).read_text(encoding="utf-8").splitlines()
            if x.strip()]
    cache = FeatureCache(args.cache, cfg35)

    def text_of(row):
        return " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"])

    def fricative_score(row):
        marks = ("s", "x", "kh", "ph", "ch", "th", "h")
        return sum(1 for s in row["syllables"]
                   if str(s.get("raw") or "").lower().startswith(marks))

    scored = sorted(rows, key=fricative_score, reverse=True)
    chosen = scored[:3] + [r for r in rows if r not in scored[:3]][:6]
    p2 = Synthesizer(cfg32, "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt",
                     device=args.device)
    gan = Synthesizer(cfg35, "runs/adv_v35/eval_2k_ema.pt", device=args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(185)
    key, metrics = {}, {}
    page = ["<!doctype html><meta charset='utf-8'><title>v18.5 nghe mù</title>",
            ("<style>body{font:15px system-ui;background:#111;color:#eee;max-width:1100px;"
             "margin:20px auto}td{padding:6px;vertical-align:top}audio{width:320px}"
             ".lbl{font-weight:600}</style>"),
            "<h2>v18.5 — nghe MÙ: P2 @2k · GAN2k · copy-synth (mốc trên)</h2>",
            ("<p>Chấm: (a) rè nền, (b) từ lạc quẻ, (c) phụ âm s/x/kh nghe đục hay rõ. "
             "−23 LUFS · α=0 · cả câu.</p>")]
    for index, row in enumerate(chosen):
        rid = f"v185_{index + 1:02d}"
        start, end = row["frame_range"]
        sf.read(row["audio_path"], dtype="float32", start=start * HOP,
                frames=(end - start) * HOP)
        mel = cache.get(row)["mel"][start:end].to(p2.device)
        with torch.no_grad():
            audio = p2.vocoder(sharpen_mel(mel, 0.0).unsqueeze(0))[0]
        copy = postprocess(np.asarray(audio.float().cpu()).reshape(-1), SR).astype(np.float32)
        pieces = [a for a, _ in p2.iter_synthesize_sentence(text_of(row))]
        p2w = postprocess(np.concatenate(pieces), SR).astype(np.float32)
        pieces = [a for a, _ in gan.iter_synthesize_sentence(text_of(row))]
        ganw = postprocess(np.concatenate(pieces), SR).astype(np.float32)
        variants = {"p2": p2w, "gan2k": ganw, "copy": copy}
        labels = "ABC"
        order = list(variants)
        rng.shuffle(order)
        mapping = {}
        cells = []
        for label, tag in zip(labels, order):
            wav, _ = normalize_loudness(variants[tag], SR, target_lufs=-23.0)
            folder = out / tag
            folder.mkdir(parents=True, exist_ok=True)
            write_wav(folder / f"{rid}.wav", wav, SR)
            metrics.setdefault(tag, []).append(round(flatness(wav), 4))
            mapping[label] = tag
            cells.append(f"<td><div class='lbl'>{label}</div>"
                         f"<audio controls src='{tag}/{rid}.wav'></audio></td>")
        key[rid] = mapping
        page.append(f"<h3>{rid} (phụ âm: {fricative_score(row)}) — {text_of(row)[:100]}…</h3>"
                    "<table><tr>" + "".join(cells) + "</tr></table>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")
    summary = {tag: round(float(np.median(v)), 4) for tag, v in metrics.items()}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"answer_key": key, "flatness": summary},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
    print("flatness (12 câu, α=0, −23 LUFS):", summary)
    print("trang:", out / "index.html")


if __name__ == "__main__":
    main()

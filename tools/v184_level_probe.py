"""V18.4 — S1 (sai số độ to) + B1.2 (vuv dự đoán vs GT), TF, 40 segment, CI bootstrap.

spike_err = tỉ lệ âm tiết |level_model − level_GT| > 3 dB (và > 4 dB); RMS MAE theo âm tiết.
Mọi số in kèm độ lệch so copy-synth.
Chạy: .venv/bin/python tools/v184_level_probe.py --out artifacts/bench/v184_level.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

SR, HOP = 24000, 256
CFG35 = "configs/small_codelung_v35.yaml"
CFG32 = "configs/small_codelung_v32.yaml"
MODELS = [
    ("v32", "runs/acoustic_v32/best.pt", CFG32),
    ("p2_2k", "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt", CFG32),
    ("v35", "runs/acoustic_v35/best_ema.pt", CFG35),
    ("gan1k", "runs/adv_v35/eval_1000_ema.pt", CFG35),
    ("gan2k", "runs/adv_v35/eval_2k_ema.pt", CFG35),
    ("p2_4k", "runs/adv_p2_20k/eval_4000.pt", CFG32),
    ("p2_6k", "runs/adv_p2_20k/eval_6000.pt", CFG32),
]


def syl_levels(wave: np.ndarray, seg_index: np.ndarray) -> np.ndarray:
    count = min(len(wave) // HOP, len(seg_index))
    rms = np.sqrt(np.mean(wave[:count * HOP].reshape(count, HOP) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    out = []
    for syl in range(int(seg_index[:count].max()) + 1 if count else 0):
        mask = seg_index[:count] == syl
        if mask.any():
            out.append(float(np.mean(db[mask])))
    return np.asarray(out)


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
        seg_index = np.zeros(end - start, dtype=np.int64)
        cursor = 0
        for idx, (dur, gap) in enumerate(zip(row["durations"], list(row["gaps"]) + [0])):
            seg_index[cursor:cursor + dur] = idx
            cursor += dur + gap
        gt_syl_vuv = np.asarray([float(np.mean(vuv[seg_index == i])) if (seg_index == i).any()
                                 else 0.0 for i in range(len(row["durations"]))])
        return wav, vuv, seg_index, gt_syl_vuv

    per_model: dict[str, dict[str, list[float]]] = {}
    buzz: dict[str, list[float]] = {}

    # copy-synth: mel thật → Vocos
    eng = Synthesizer(cfg35, "runs/acoustic_v35/best_ema.pt", device=args.device)
    per_model["copy"] = {"spike3": [], "spike4": [], "mae": []}
    for row in rows:
        wav, _vuv, seg_index, _gtv = prepare(row)
        start, end = row["frame_range"]
        mel = cache.get(row)["mel"][start:end].to(eng.device)
        with torch.no_grad():
            audio = eng.vocoder(sharpen_mel(mel, 0.0).unsqueeze(0))[0]
        lv_s = syl_levels(np.asarray(audio.float().cpu()).reshape(-1), seg_index)
        lv_r = syl_levels(wav, seg_index)
        n = min(len(lv_s), len(lv_r))
        d = np.abs(lv_s[:n] - lv_r[:n])
        per_model["copy"]["spike3"].append(float(np.mean(d > 3)))
        per_model["copy"]["spike4"].append(float(np.mean(d > 4)))
        per_model["copy"]["mae"].append(float(np.mean(d)))

    for tag, ckpt, cfg_path in MODELS:
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        engine = Synthesizer(cfg, ckpt, device=args.device)
        per_model[tag] = {"spike3": [], "spike4": [], "mae": []}
        buzz[tag] = []
        for row in rows:
            wav, _vuv, seg_index, gt_syl_vuv = prepare(row)
            items = parse_text(text_of(row), resolve_oov=True)
            feats = torch.tensor([[engine.vocabs.encode(it) for it in items]],
                                 dtype=torch.long, device=engine.device)
            gaps = list(row["gaps"]) + [0] if len(row["gaps"]) == len(row["durations"]) - 1 \
                else list(row["gaps"])
            with torch.no_grad():
                out = engine.model(feats, speed=torch.tensor([1.0], device=engine.device),
                                   durations=torch.tensor([row["durations"]], dtype=torch.long,
                                                          device=engine.device),
                                   gaps=torch.tensor([gaps], dtype=torch.long,
                                                     device=engine.device),
                                   lead=torch.tensor([0], dtype=torch.long, device=engine.device))
                audio = engine.vocoder(sharpen_mel(out["mel"][0], 0.0).unsqueeze(0))[0]
            lv_s = syl_levels(np.asarray(audio.float().cpu()).reshape(-1), seg_index)
            lv_r = syl_levels(wav, seg_index)
            n = min(len(lv_s), len(lv_r))
            d = np.abs(lv_s[:n] - lv_r[:n])
            per_model[tag]["spike3"].append(float(np.mean(d > 3)))
            per_model[tag]["spike4"].append(float(np.mean(d > 4)))
            per_model[tag]["mae"].append(float(np.mean(d)))
            pred = torch.sigmoid(out["seg_vuv_logits"][0][:, 2]).cpu().numpy()
            silent = gt_syl_vuv[:len(pred)] < 0.1
            buzz[tag].append(float(np.mean(pred[silent] > 0.5)) if silent.any() else float("nan"))

    def boot(values, n=1000):
        vals = np.asarray([v for v in values if np.isfinite(v)])
        rng = np.random.default_rng(0)
        meds = [float(np.median(rng.choice(vals, vals.size, replace=True))) for _ in range(n)]
        return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

    base = float(np.median(per_model["copy"]["mae"]))
    print(f"{'':8s} {'spike3':>7s} {'CI95':>14s} {'spike4':>7s} {'MAE dB':>7s} "
          f"{'Δcopy':>7s} {'CI95 MAE':>14s} {'buzz':>6s}")
    table = {}
    for tag in ["copy", *[m[0] for m in MODELS]]:
        m = per_model[tag]
        s3, s4, mae = (float(np.median(m["spike3"])), float(np.median(m["spike4"])),
                       float(np.median(m["mae"])))
        ci_s3, ci_mae = boot(m["spike3"]), boot(m["mae"])
        b = float(np.nanmedian(buzz[tag])) if tag in buzz else float("nan")
        print(f"{tag:8s} {s3:7.3f} [{ci_s3[0]:.3f},{ci_s3[1]:.3f}] {s4:7.3f} "
              f"{mae:7.2f} {mae-base:6.2f} [{ci_mae[0]:.2f},{ci_mae[1]:.2f}] {b:6.3f}")
        table[tag] = {"spike3": round(s3, 4), "spike3_ci": [round(v, 4) for v in ci_s3],
                      "spike4": round(s4, 4), "mae": round(mae, 3),
                      "mae_ci": [round(v, 3) for v in ci_mae],
                      "buzz": None if np.isnan(b) else round(b, 4), "n": len(rows)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

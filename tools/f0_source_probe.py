"""tools/f0_source_probe.py — tách nguồn lỗi prosody: đầu dự đoán hay decoder (A0 v12, Việc 8a).

Ba chế độ cho cùng segment (20 segment có "rồi" ở đầu):
  - tf_dur : duration/slot THẬT, f0/vuv/energy/phonation DỰ ĐOÁN (model tự tính).
  - tf_full: duration/slot THẬT + f0/vuv/energy/phonation THẬT (variance_override).
  - free   : tất cả dự đoán.

Đo trên AUDIO đầu ra (cùng tracker pyworld với giọng thật), tham chiếu f0 trung vị THẬT của segment:
  semis   = f0 token so trung vị segment (semitone)
  contour = f0 cuối 30 % − đầu 30 % (semitone)
  creak%  = tỉ lệ frame không bắt được cao độ
  slope/corr = hồi quy f0 audio theo f0 thật trên TẤT CẢ âm tiết của segment.

Chạy:
  .venv/bin/python tools/f0_source_probe.py --config configs/small_codelung_v32.yaml \
      --checkpoint artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt --tag p2_2k \
      --out artifacts/bench/f0src_p2_2k.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _f0_curve(x: np.ndarray, sr: int) -> np.ndarray:
    import pyworld as pw

    x = np.asarray(x, dtype=np.float64)
    if x.size < 1024:
        return np.zeros(0)
    f0, t = pw.dio(x, sr, frame_period=5.0)
    return pw.stonemask(x, f0, t, sr)


def _semis(hz: np.ndarray, ref: float) -> np.ndarray:
    hz = hz[hz > 0]
    return 12 * np.log2(hz / max(ref, 1e-6)) if hz.size else np.array([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--manifest", default="data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--cache-dir", default="cache/codelung_v2")
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--word", default="rồi")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    import torch

    from vifasttts.data.dataset_v2 import SlotDataset
    from vifasttts.infer import Synthesizer

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json"))
                          .read_text(encoding="utf-8"))
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=2)
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    sr = engine.sample_rate
    hop = 256
    word = args.word.lower()
    rows = dataset.rows

    indices = [i for i, r in enumerate(rows)
               if (r["syllables"][0].get("normalized") or "").lower() == word][: args.rows]
    modes = {"tf_dur": [], "tf_full": [], "free": []}
    phon_rows = []

    def build_override(item, device):
        """[1,T,3+3] f0(log) · vuv · energy · phonation (khung thật)."""
        f0 = np.asarray(item["f0"], dtype=np.float64).reshape(-1)
        vuv = np.asarray(item["vuv"], dtype=np.float64).reshape(-1)
        energy = np.asarray(item["energy"], dtype=np.float64).reshape(-1)
        seg_dur = np.asarray(item["seg_durations"])
        seg_ph = np.asarray(item["seg_phonation"], dtype=np.float64)
        gaps = [int(g) for g in np.asarray(item["gaps"]).reshape(-1)]
        lead = int(np.asarray(item["lead"]).reshape(-1)[0])
        frames = f0.size
        phon = np.zeros((frames, 3), dtype=np.float64)
        cursor = lead
        for i in range(seg_dur.shape[0]):
            for j in range(4):
                d = int(seg_dur[i][j])
                end = min(frames, cursor + d)
                if end > cursor:
                    phon[cursor:end] = seg_ph[i][j]
                cursor += d
            if i < len(gaps):
                cursor += gaps[i]
        variance = np.concatenate([
            np.log(np.maximum(f0, 1e-3))[:, None],
            vuv[:, None],
            energy[:, None],
            phon,
        ], axis=-1)
        # frame nằm trong khoảng lặng (lead/gap) đã zero sẵn ở phon; f0/vuv/energy của gap cũng ~0.
        return torch.tensor(variance[None], dtype=torch.float32, device=device)

    for index in indices:
        item = dataset[index]
        features = item["features"].unsqueeze(0).to(engine.device)
        real_f0 = np.asarray(item["f0"], dtype=np.float64).reshape(-1)
        real_vuv = np.asarray(item["vuv"], dtype=np.float64).reshape(-1)
        ref = float(np.median(real_f0[real_vuv >= 0.5])) if (real_vuv >= 0.5).any() else 120.0
        durations = [int(d) for d in np.asarray(item["durations"]).reshape(-1)]
        gaps = [int(g) for g in np.asarray(item["gaps"]).reshape(-1)]
        lead = int(np.asarray(item["lead"]).reshape(-1)[0])

        with torch.no_grad():
            out_tf = engine.model(
                features, durations=item["durations"].unsqueeze(0).to(engine.device),
                seg_durations=item["seg_durations"].unsqueeze(0).to(engine.device),
                gaps=item["gaps"].unsqueeze(0).to(engine.device),
                lead=item["lead"].reshape(1).to(engine.device))
            out_full = engine.model(
                features, durations=item["durations"].unsqueeze(0).to(engine.device),
                seg_durations=item["seg_durations"].unsqueeze(0).to(engine.device),
                gaps=item["gaps"].unsqueeze(0).to(engine.device),
                lead=item["lead"].reshape(1).to(engine.device),
                variance_override=build_override(item, engine.device))
            out_free = engine.model(features)
            waves = {name: np.asarray(engine.vocoder(o["mel"])[0].float().cpu().numpy(),
                                      dtype=np.float32).reshape(-1)
                     for name, o in (("tf_dur", out_tf), ("tf_full", out_full), ("free", out_free))}

        for name, out in (("tf_dur", out_tf), ("tf_full", out_full), ("free", out_free)):
            wav = waves[name]
            f0_curve = _f0_curve(wav, sr)
            if name == "free":
                dur0 = int(np.asarray(out["durations"].cpu()).reshape(-1)[0])
                lead0 = int(np.asarray(out["slot_durations"].cpu()).reshape(-1)[0])
            else:
                dur0, lead0 = durations[0], lead
            # pyworld dùng frame 5 ms; model dùng hop 256 mẫu (10,67 ms) — quy đổi qua giây.
            fps = sr * 0.005
            a = int(lead0 * hop / fps)
            b = a + int(dur0 * hop / fps)
            token = f0_curve[a:b] if f0_curve.size else np.zeros(0)
            semis = _semis(token, ref)
            voiced = token > 0
            creak = float(1.0 - voiced.mean()) if token.size else float("nan")
            n = token.size
            head, tail = token[: max(1, n // 3)], token[-max(1, n // 3):]
            hf, tf = _semis(head, ref), _semis(tail, ref)
            contour = float(np.median(tf) - np.median(hf)) if hf.size and tf.size else float("nan")
            # slope/corr trên mọi âm tiết (audio vs nhãn thật)
            real_s, model_s = [], []
            cursor = lead
            for k, d in enumerate(durations):
                a2 = int(cursor * hop / (sr * 0.005))
                b2 = int((cursor + d) * hop / (sr * 0.005))
                lab = real_f0[cursor:cursor + d]
                lab = lab[real_vuv[cursor:cursor + d] >= 0.5]
                cur_audio = f0_curve[a2:b2] if f0_curve.size else np.zeros(0)
                cur_audio = cur_audio[cur_audio > 0]
                if lab.size >= 3 and cur_audio.size >= 3:
                    real_s.append(12 * np.log2(np.median(lab) / ref))
                    model_s.append(12 * np.log2(np.median(cur_audio) / ref))
                cursor += d + (gaps[k] if k < len(gaps) else 0)
            if len(real_s) >= 5:
                rs, ms = np.array(real_s), np.array(model_s)
                slope = float(np.polyfit(rs, ms, 1)[0])
                corr = float(np.corrcoef(rs, ms)[0, 1])
            else:
                slope, corr = float("nan"), float("nan")
            modes[name].append({"semis": float(np.median(semis)) if semis.size else None,
                                "contour": contour, "creak": creak, "slope": slope, "corr": corr})

        phon_pred = np.asarray(out_tf["seg_phonation"][0].float().cpu()).reshape(-1, 3)[0]
        phon_real = np.asarray(item["seg_phonation"]).reshape(-1, 3)[0]
        phon_rows.append({"pred": [round(float(v), 3) for v in phon_pred],
                          "real": [round(float(v), 3) for v in phon_real]})

    def agg(rows_in: list[dict]) -> dict:
        def med(key):
            vals = [r[key] for r in rows_in if r[key] is not None and np.isfinite(r[key])]
            return round(float(np.median(vals)), 3) if vals else None
        return {"n": len(rows_in), "semis": med("semis"), "contour": med("contour"),
                "creak": med("creak"), "slope": med("slope"), "corr": med("corr")}

    result = {"tag": args.tag, "checkpoint": args.checkpoint, "word": word,
              "real_semis_ref": 0.0,
              "modes": {k: agg(v) for k, v in modes.items()},
              "phonation_pred_vs_real": {
                  "pred_median": [round(float(np.median([p["pred"][i] for p in phon_rows])), 3) for i in range(3)],
                  "real_median": [round(float(np.median([p["real"][i] for p in phon_rows])), 3) for i in range(3)],
              }}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, a in result["modes"].items():
        print(f"{args.tag:8s} {name:8s} semis {a['semis']} · contour {a['contour']} · creak {a['creak']} "
              f"· slope {a['slope']} · corr {a['corr']} · n={a['n']}")
    print(f"         phonation (hnr, glot, jitter) pred {result['phonation_pred_vs_real']['pred_median']} "
          f"vs real {result['phonation_pred_vs_real']['real_median']}")


if __name__ == "__main__":
    main()

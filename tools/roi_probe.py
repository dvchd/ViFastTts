"""tools/roi_probe.py — soi "rồi" (và từ khoá) theo vị trí: giọng thật vs model (A0 v11, Việc 7/2.2).

Vì sao: "rồi" trong d06 bị quá trầm ở MỌI model. Cần biết giọng thật có trầm ở vị trí đầu mệnh đề
(sau dấu phẩy) không, và model sai ở đâu (nhãn f0 / dự đoán free-run / decoder).

Đo (f0 bằng pyworld trên waveform; nhãn thật lấy từ cache):
  - f0_med tính theo semitone so với f0 trung vị của segment: 12·log2(f0/f0_seg)
  - contour = f0 cuối 30 % − f0 đầu 30 % (semitone)
  - creak% = tỉ lệ frame vuv < 0,5
  - dur_frames, energy_rel_db (năng lượng so trung vị segment)

Chạy:
  .venv/bin/python tools/roi_probe.py --config configs/small_codelung_v32.yaml \
      --checkpoint artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt --tag p2_2k \
      --word rồi --out artifacts/bench/roi_p2_2k.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _f0_curve(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    import pyworld as pw

    x = np.asarray(x, dtype=np.float64)
    f0, t = pw.dio(x, sr, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    return f0, t


def _stats(f0: np.ndarray, vuv: np.ndarray | None, ref_hz: float) -> dict:
    voiced = f0 > 0
    med = float(np.median(f0[voiced])) if voiced.any() else float("nan")
    semis = 12 * np.log2(np.maximum(f0[voiced], 1e-6) / max(ref_hz, 1e-6)) if voiced.any() else np.array([])
    n = f0.size
    head = f0[: max(1, n // 3)]
    tail = f0[-max(1, n // 3):]
    head_v, tail_v = head[head > 0], tail[tail > 0]
    contour = (float(np.median(12 * np.log2(tail_v / max(ref_hz, 1e-6))))
               - float(np.median(12 * np.log2(head_v / max(ref_hz, 1e-6))))) if head_v.size and tail_v.size else float("nan")
    creak = float(1.0 - voiced.mean()) if n else float("nan")
    if vuv is not None and vuv.size == n:
        creak = float((vuv < 0.5).mean())
    return {"f0_hz": round(med, 1) if np.isfinite(med) else None,
            "semis": round(float(np.median(semis)), 2) if semis.size else None,
            "contour": round(contour, 2) if np.isfinite(contour) else None,
            "creak": round(creak, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--manifest", default="data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--cache-dir", default="cache/codelung_v2")
    parser.add_argument("--word", default="rồi")
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--corpus-rows", type=int, default=400)
    parser.add_argument("--text", default="gõ địa chỉ, rồi bấm tìm kiếm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    import torch

    from vifasttts.data.dataset_v2 import SlotDataset
    from vifasttts.infer import Synthesizer

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json"))
                          .read_text(encoding="utf-8"))
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=1)
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    sr = engine.sample_rate
    hop = 256
    word = args.word.lower()

    # ── A. giọng thật: mọi vị trí ──
    rows = dataset.rows
    corpus = {"start": [], "mid": [], "end": []}
    used = 0
    for index, row in enumerate(rows):
        if used >= args.corpus_rows:
            break
        hits = [k for k, s in enumerate(row["syllables"])
                if (s.get("normalized") or s.get("raw") or "").lower() == word]
        if not hits:
            continue
        item = dataset[index]
        f0 = np.asarray(item["f0"], dtype=np.float64).reshape(-1)
        vuv = np.asarray(item["vuv"], dtype=np.float64).reshape(-1)
        energy = np.asarray(item["energy"], dtype=np.float64).reshape(-1)
        durations = [int(d) for d in np.asarray(item["durations"]).reshape(-1)]
        gaps = [int(g) for g in np.asarray(item["gaps"]).reshape(-1)]
        lead = int(np.asarray(item["lead"]).reshape(-1)[0])
        ref = float(np.median(f0[vuv >= 0.5])) if (vuv >= 0.5).any() else float("nan")
        e_ref = float(np.median(energy)) if energy.size else float("nan")
        for k in hits:
            start = lead + sum(durations[:k]) + sum(gaps[:k])
            end = start + durations[k]
            st = _stats(f0[start:end], vuv[start:end], ref)
            st["dur_frames"] = int(durations[k])
            st["energy_db"] = round(float(np.mean(energy[start:end]) - e_ref), 2) if energy.size else None
            prev_end = bool(k == 0) or bool(row["syllables"][k - 1].get("boundaries", {}).get("phrase_end")
                                            or row["syllables"][k - 1].get("boundaries", {}).get("sentence_end"))
            position = "start" if prev_end else ("end" if k == len(row["syllables"]) - 1 else "mid")
            corpus[position].append(st)
        used += 1

    # ── B. model: TF + free-run trên segment thật có từ ở đầu ──
    starts = [i for i, r in enumerate(rows)
              if (r["syllables"][0].get("normalized") or "").lower() == word][: args.rows]
    model_tf, model_free = [], []
    for index in starts:
        item = dataset[index]
        features = item["features"].unsqueeze(0).to(engine.device)
        with torch.no_grad():
            out_tf = engine.model(features,
                                  durations=item["durations"].unsqueeze(0).to(engine.device),
                                  seg_durations=item["seg_durations"].unsqueeze(0).to(engine.device),
                                  gaps=item["gaps"].unsqueeze(0).to(engine.device),
                                  lead=item["lead"].reshape(1).to(engine.device))
            wav_tf = engine.vocoder(out_tf["mel"])[0]
            out_fr = engine.model(features)
            wav_fr = engine.vocoder(out_fr["mel"])[0]
        dur_tf = int(np.asarray(out_tf["durations"].cpu()).reshape(-1)[0])
        dur_fr = int(np.asarray(out_fr["durations"].cpu()).reshape(-1)[0])
        for bucket, wav, dur in ((model_tf, wav_tf, dur_tf), (model_free, wav_fr, dur_fr)):
            x = np.asarray(wav.float().cpu().numpy(), dtype=np.float32).reshape(-1)
            f0c, _t = _f0_curve(x, sr)
            ref = float(np.median(f0c[f0c > 0])) if (f0c > 0).any() else float("nan")
            n_frames = max(1, int(len(x) / hop))
            st = _stats(f0c, None, ref)
            st["dur_frames"] = dur
            st["n_frames"] = n_frames
            bucket.append(st)

    # ── C. câu thật của người dùng (free-run, chunk bắt đầu bằng từ khoá) ──
    d06 = None
    raw = [a for a, _ in engine.iter_synthesize_steady(args.text, sharpen_alpha=0.0)]
    chunks = [c for c in raw if c.size and float(np.max(np.abs(c))) > 1e-3]
    if len(chunks) >= 2:
        seg = chunks[1]  # "rồi bấm tìm kiếm" (sau dấu phẩy)
        f0c, _t = _f0_curve(seg, sr)
        ref = float(np.median(f0c[f0c > 0])) if (f0c > 0).any() else float("nan")
        d06 = _stats(f0c, None, ref)

    def summary(bucket: list[dict]) -> dict:
        if not bucket:
            return {}
        semis = [b["semis"] for b in bucket if b["semis"] is not None]
        creak = [b["creak"] for b in bucket if b["creak"] is not None]
        contour = [b["contour"] for b in bucket if b["contour"] is not None]
        dur = [b.get("dur_frames", 0) for b in bucket]
        return {"n": len(bucket),
                "semis_median": round(float(np.median(semis)), 2) if semis else None,
                "creak_median": round(float(np.median(creak)), 3) if creak else None,
                "contour_median": round(float(np.median(contour)), 2) if contour else None,
                "dur_median": float(np.median(dur)) if dur else None}

    result = {"tag": args.tag, "checkpoint": args.checkpoint, "word": word,
              "corpus": {k: summary(v) for k, v in corpus.items()},
              "model_tf": summary(model_tf), "model_free": summary(model_free), "d06": d06}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.tag:8s} {word}: thật start {result['corpus']['start']}")
    print(f"         thật mid {result['corpus']['mid']} · end {result['corpus']['end']}")
    print(f"         model TF {result['model_tf']}")
    print(f"         model free {result['model_free']}")
    print(f"         d06 free-run {d06}")


if __name__ == "__main__":
    main()

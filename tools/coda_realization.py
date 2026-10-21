#!/usr/bin/env python
"""tools/coda_realization.py — coda mũi (NH/NG/N/M) có bị mất khi synth? (đo tỉ lệ mũi 250–500 Hz).

Vì sao cần: `mel_z` bình quân **không bắt** được lỗi mất coda (năng lượng coda nhỏ, L1 bị chi phối
bởi nguyên âm) — đã gặp thật: "mình" → nghe như "mìn" nhưng mel_z của NH/NG lại *thấp nhất*.
Thước này đo **tỉ lệ năng lượng dải mũi 250–500 Hz / 500–2000 Hz** trong slot **C** (coda) của
`seg_durations`, so bản thật với bản synth (teacher-forced, đúng nhãn).

Chạy:
  .venv/bin/python tools/coda_realization.py \
      --config runs/acoustic_v2_b8/resolved_config.json \
      --checkpoint runs/acoustic_v2_b8/best.pt \
      --manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
      --cache-dir cache/codelung_v2 --clips 20 --device mps \
      --out artifacts/eval_acoustic/b8_coda.json
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

NASAL_CODAS = ("NH", "NG", "N", "M")
HOP = 256


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as fh:
        return np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def band_ratio(x: np.ndarray, sr: int) -> float:
    """Năng lượng 250–500 Hz / 500–2000 Hz (murmur mũi / phần còn lại)."""
    if x.size < 128:
        return float("nan")
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    low = spec[(freqs >= 250) & (freqs < 500)].sum()
    mid = spec[(freqs >= 500) & (freqs < 2000)].sum() + 1e-12
    return float(low / mid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clips", type=int, default=20)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--sharpen-alpha", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    import yaml

    from vifasttts.data.dataset_v2 import SlotDataset
    from vifasttts.infer import Synthesizer
    from vifasttts.runtime.postprocess import sharpen_mel

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    sr = engine.sample_rate
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json"))
                          .read_text(encoding="utf-8"))
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=2)

    rows: list[dict] = []
    for index in range(min(args.clips, len(dataset))):
        row = dataset.rows[index]
        item = dataset[index]
        features = item["features"].unsqueeze(0).to(engine.device)
        with torch.no_grad():
            outputs = engine.model(
                features,
                durations=item["durations"].unsqueeze(0).to(engine.device),
                seg_durations=item["seg_durations"].unsqueeze(0).to(engine.device),
                gaps=item["gaps"].unsqueeze(0).to(engine.device),
                lead=item["lead"].reshape(1).to(engine.device),
            )
            wave_t = engine.vocoder(sharpen_mel(outputs["mel"][0], args.sharpen_alpha).unsqueeze(0))[0]
        synth = np.asarray(wave_t.float().cpu().numpy(), dtype=np.float32)
        real_full = _read_wav(_ROOT / str(row["audio_path"]))
        start, end = int(row["frame_range"][0]), int(row["frame_range"][1])
        real = real_full[start * HOP: end * HOP]
        seg4 = item["seg_durations"].tolist()
        cursor = int(item["lead"])
        for i, syl in enumerate(row["syllables"]):
            coda = str((syl.get("abstract") or {}).get("coda") or "NONE")
            duration = int(row["durations"][i])
            gap = int(row["gaps"][i]) if i < len(row["gaps"]) else 0
            o, m, nucleus, c = seg4[i]
            if coda in NASAL_CODAS and c >= 2:
                c0 = (cursor + o + m + nucleus) * HOP
                c1 = c0 + max(2, int(c)) * HOP
                rows.append({
                    "clip": row["id"], "i": i, "syllable": str(syl.get("raw")),
                    "coda": coda, "slot_frames": int(c),
                    "real": band_ratio(real[c0:c1], sr), "synth": band_ratio(synth[c0:c1], sr),
                })
            cursor += duration + gap
        print(f"[coda] {row['id']}: {sum(1 for r in rows if r['clip'] == row['id'])} âm tiết coda mũi",
              flush=True)

    real_arr = np.array([r["real"] for r in rows], dtype=np.float64)
    synth_arr = np.array([r["synth"] for r in rows], dtype=np.float64)
    summary = {
        "n": len(rows),
        "ratio_real_mean": float(np.nanmean(real_arr)),
        "ratio_synth_mean": float(np.nanmean(synth_arr)),
        "dropped_gt30pct": int(np.sum(synth_arr < 0.7 * real_arr)),
        "by_coda": {},
        "rows": rows,
    }
    for coda in NASAL_CODAS:
        sel = [r for r in rows if r["coda"] == coda]
        if sel:
            summary["by_coda"][coda] = {
                "n": len(sel),
                "real": float(np.mean([r["real"] for r in sel])),
                "synth": float(np.mean([r["synth"] for r in sel])),
            }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} âm tiết coda mũi · tỉ lệ mũi thật {summary['ratio_real_mean']:.2f} → "
          f"synth {summary['ratio_synth_mean']:.2f} · giảm >30%: {summary['dropped_gt30pct']}")
    for coda, entry in summary["by_coda"].items():
        print(f"  coda {coda:3s} n={entry['n']:3d} · thật {entry['real']:7.2f} → synth {entry['synth']:7.2f}")
    print(f"Xong: {out}")


if __name__ == "__main__":
    main()

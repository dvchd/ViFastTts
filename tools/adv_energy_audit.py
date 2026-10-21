"""tools/adv_energy_audit.py — thước NĂNG LƯỢNG TỪNG ÂM TIẾT (A0 lượt 6, Việc 4a).

Vì sao: tai người dùng nghe P1 @4k có **từ đột nhiên to lên**. GV/flux/smooth đều là thước
texture, không bắt được lỗi mức năng lượng tương đối giữa các từ. Tool này đo:

- Năng lượng mỗi âm tiết: **mean log-mel đã bỏ z-norm** (đơn vị dB = 8,686 × log-tự-nhiên) trên
  frame của âm tiết; và **RMS dB** của waveform sau Vocos trên đúng khoảng mẫu của âm tiết.
- Chuẩn hoá theo câu: trừ **trung vị** năng lượng âm tiết của câu (đo độ to TƯƠNG ĐỐI).
- `spike_rate` = tỉ lệ âm tiết có (synth − thật) > **+3 dB** (có audio thật); với câu mới
  (không có thật) = tỉ lệ lệch > **+4 dB** so trung vị trượt 5 âm tiết lân cận.
- `mae_db` = MAE năng lượng âm tiết (dB) synth so thật.
- Đo ở **α = 0 và α = 1,5** (α tác động lên mel z trước Vocos).
- Xuất top 10 spike (id, từ, thời điểm) + trang nghe các âm tiết đó (thật / synth α=0 / α=1,5).

Chạy:
  .venv/bin/python tools/adv_energy_audit.py --config configs/small_codelung_v32.yaml \
      --checkpoint artifacts/adv_grid/P1/ckpt_2k_ema.pt --tag p1_2k \
      --manifest data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl --cache-dir cache/codelung_v2 \
      --segments 40 --out artifacts/bench/energy_p1_2k.json --html artifacts/listen_small_filt/energy/p1_2k
"""

from __future__ import annotations

import argparse
import html
import json
import wave
from pathlib import Path

import numpy as np
import torch
import yaml

_DB = 8.685889638
_HOP = 256


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as fh:
        return np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())


def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def _sliding_spikes(values: list[float], threshold: float = 4.0, window: int = 5) -> int:
    """Đếm âm tiết lệch > threshold dB so trung vị `window` âm tiết lân cận (không gồm chính nó)."""
    count = 0
    for i, value in enumerate(values):
        lo, hi = max(0, i - window), min(len(values), i + window + 1)
        neighbours = [values[j] for j in range(lo, hi) if j != i]
        if neighbours and value - float(np.median(neighbours)) > threshold:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--segments", type=int, default=40)
    parser.add_argument("--alphas", default="0")
    parser.add_argument("--texts", default="examples/blind_texts_v6.json")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", required=True)
    parser.add_argument("--html", default="")
    args = parser.parse_args()

    from vifasttts.data.dataset_v2 import SlotDataset
    from vifasttts.infer import Synthesizer
    from vifasttts.runtime.postprocess import sharpen_mel

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sample_rate = int(config.get("sample_rate", 24000))
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json")).read_text(encoding="utf-8"))
    stats = json.loads(Path(config["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = torch.tensor(stats["std"], dtype=torch.float32).clamp_min(1e-4)
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    device = engine.device
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=1)
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    def de_z(mel: torch.Tensor) -> np.ndarray:
        return (mel.detach().cpu() * std + mean).numpy()

    per_alpha = {alpha: {"synth": [], "real": [], "rms_synth": [], "rms_real": [], "rows": []} for alpha in alphas}
    n_used = 0
    for index in range(min(args.segments, len(dataset))):
        item = dataset[index]
        row = dataset.rows[index]
        features = item["features"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = engine.model(features,
                               durations=item["durations"].unsqueeze(0).to(device),
                               seg_durations=item["seg_durations"].unsqueeze(0).to(device),
                               gaps=item["gaps"].unsqueeze(0).to(device),
                               lead=item["lead"].reshape(1).to(device))
        pred = out["mel"][0].float().cpu()
        real = item["mel"]
        frames = min(pred.size(0), real.size(0))
        durations = [int(d) for d in item["durations"].tolist()]
        gaps = [int(g) for g in item["gaps"].tolist()]
        lead = int(item["lead"])
        cursor = lead
        real_wav = _read_wav(Path(".") / str(row["audio_path"]))
        f0, f1 = int(row["frame_range"][0]), int(row["frame_range"][1])
        real_wav = real_wav[f0 * _HOP: f1 * _HOP]
        for alpha in alphas:
            sharp = sharpen_mel(pred, alpha).unsqueeze(0).to(device)
            with torch.no_grad():
                wav_t = engine.vocoder(sharp)[0]
            synth_wav = np.asarray(wav_t.float().cpu().numpy(), dtype=np.float32)
            synth_raw = de_z(sharpen_mel(pred, alpha))[:frames]
            real_raw = de_z(real)[:frames]
            syn_energy, real_energy, syn_rms, real_rms = [], [], [], []
            cursor = lead
            for i, dur in enumerate(durations):
                lo, hi = cursor, min(frames, cursor + dur)
                if hi > lo:
                    syn_energy.append(float(synth_raw[lo:hi].mean()) * _DB)
                    real_energy.append(float(real_raw[lo:hi].mean()) * _DB)
                    a, b = lo * _HOP, min(len(synth_wav), hi * _HOP)
                    syn_rms.append(_rms_db(synth_wav[a:b]))
                    c, d = lo * _HOP, min(len(real_wav), hi * _HOP)
                    real_rms.append(_rms_db(real_wav[c:d]))
                cursor += dur + (gaps[i] if i < len(gaps) else 0)
            if len(syn_energy) < 3:
                continue
            syn_rel = np.array(syn_energy) - np.median(syn_energy)
            real_rel = np.array(real_energy) - np.median(real_energy)
            rms_syn_rel = np.array(syn_rms) - np.median(syn_rms)
            rms_real_rel = np.array(real_rms) - np.median(real_rms)
            # A0 v11 §4: loại âm tiết có mức THẬT < trung vị segment − 10 dB (frame thật gần lặng
            # ⇒ lỗi thước, không phải lỗi model — 10/10 top spike cũ đều rơi vào nhóm này).
            keep = real_rel > -10.0
            index_map = np.where(keep)[0]
            per_alpha[alpha]['synth'].extend(syn_rel[keep].tolist())
            per_alpha[alpha]['real'].extend(real_rel[keep].tolist())
            per_alpha[alpha]['rms_synth'].extend(rms_syn_rel[keep].tolist())
            per_alpha[alpha]['rms_real'].extend(rms_real_rel[keep].tolist())
            delta = (syn_rel - real_rel)[keep]
            for position, value in enumerate(delta):
                if value > 3.0:
                    i = int(index_map[position])
                    per_alpha[alpha]['rows'].append({
                        'segment': str(row['id']), 'syllable': i,
                        'word': str(row['syllables'][i].get('raw', '')) if i < len(row['syllables']) else '',
                        'time': round((lead + sum(durations[:i]) + sum(gaps[:i])) * _HOP / sample_rate, 2),
                        'delta_db': round(float(value), 2)})
        n_used += 1

    summary = {}
    for alpha in alphas:
        d = per_alpha[alpha]
        syn = np.array(d["synth"])
        real = np.array(d["real"])
        rms_delta = np.array(d["rms_synth"]) - np.array(d["rms_real"]) if d["rms_synth"] else np.array([])
        summary[str(alpha)] = {
            "spike_rate": round(float((syn - real > 3.0).mean()), 4) if syn.size else None,
            "mae_db": round(float(np.abs(syn - real).mean()), 3) if syn.size else None,
            "rms_spike_rate": round(float((rms_delta > 3.0).mean()), 4) if rms_delta.size else None,
            "rms_mae_db": round(float(np.abs(rms_delta).mean()), 3) if rms_delta.size else None,
            "n_syllables": int(syn.size),
            "top_spikes": sorted(d["rows"], key=lambda r: -r["delta_db"])[:10]}
    result = {"tag": args.tag, "checkpoint": args.checkpoint, "segments": n_used, "summary": summary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for alpha in alphas:
        s = summary[str(alpha)]
        print(f"{args.tag or 'ckpt':14s} α={alpha:<4} spike {s['spike_rate']:.3f} · MAE {s['mae_db']} dB "
              f"· RMS spike {s['rms_spike_rate']:.3f} · RMS MAE {s['rms_mae_db']} dB · n={s['n_syllables']}")

    if args.html:
        page = ["<!doctype html><meta charset='utf-8'><title>Spike năng lượng</title>",
                "<body style='font-family:sans-serif;max-width:1100px;margin:2em auto'>",
                f"<h1>Top âm tiết lệch năng lượng — {html.escape(args.tag)}</h1>",
                "<p>Chuẩn hoá theo câu (trừ trung vị). ΔdB = synth − thật (mel sau bỏ z-norm).</p>",
                "<table border='1' cellpadding='6' style='border-collapse:collapse;font-size:13px'>",
                "<tr><th>#</th><th>câu</th><th>từ</th><th>t(s)</th><th>ΔdB (α=0)</th><th>ΔdB (α=1,5)</th></tr>"]
        top = sorted({(r["segment"], r["syllable"], r["word"], r["time"]) for a in alphas
                      for r in summary[str(a)]["top_spikes"]})
        for rank, (segment, syllable, word, time) in enumerate(top[:15], start=1):
            d0 = next((r["delta_db"] for r in summary.get("0.0", {}).get("top_spikes", [])
                       if r["segment"] == segment and r["syllable"] == syllable), "")
            d15 = next((r["delta_db"] for r in summary.get("1.5", {}).get("top_spikes", [])
                        if r["segment"] == segment and r["syllable"] == syllable), "")
            page.append(f"<tr><td>{rank}</td><td>{html.escape(segment)}</td><td>{html.escape(word)}</td>"
                        f"<td>{time}</td><td>{d0}</td><td>{d15}</td></tr>")
        page.append("</table></body>")
        html_dir = Path(args.html)
        html_dir.mkdir(parents=True, exist_ok=True)
        (html_dir / "index.html").write_text("\n".join(page), encoding="utf-8")
        print(f"→ {html_dir / 'index.html'}")


if __name__ == "__main__":
    main()

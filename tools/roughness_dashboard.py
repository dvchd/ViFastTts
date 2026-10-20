#!/usr/bin/env python
"""tools/roughness_dashboard.py — dashboard độ rè **theo từng âm tiết** (synth vs bản thật).

Với mỗi segment val/holdout: chạy model **teacher-forced** (đúng `durations`/`gaps`/`seg_durations`
của nhãn) → vocoder, rồi cắt theo **nucleus** từng âm tiết. Xuất wav thật/synth từng âm tiết và đo:

  - **CPP** (cepstral peak prominence, dB) — thước chuẩn cho giọng khàn/rè: hài càng yếu/đều càng thấp.
  - **HNR** (D4C, 0,1–4 kHz) — tỉ lệ hài/tạp.
  - **Jitter** (%) — dao động chu kỳ f0.
  - **AP 2–6 kHz** (aperiodicity trung bình dải "xì") — tạp vùng tai nhạy.

Cách dùng (nghe + đối chiếu thước nào khớp tai):
  .venv/bin/python tools/roughness_dashboard.py \
      --config runs/acoustic_v2_b8/resolved_config.json \
      --checkpoint runs/acoustic_v2_b8/best.pt \
      --manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
      --cache-dir cache/codelung_v2 --clips 8 --device mps \
      --out artifacts/listen_small_filt/roughness
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TONES = ("NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG")
_HOP = 256
_SR = 24000


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as fh:
        return np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def _write_wav(path: Path, x: np.ndarray, sr: int = _SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())


def cpp_db(frame: np.ndarray, sr: int) -> float:
    """CPP đơn giản: đỉnh cepstrum (quefrency f0 50–400 Hz) so đường hồi quy, đổi dB."""
    if frame.size < 64:
        return float("nan")
    spec = np.fft.rfft(frame * np.hamming(frame.size))
    ceps = np.fft.irfft(np.log(np.abs(spec) + 1e-8))
    qmin, qmax = max(2, int(sr / 400)), min(int(sr / 50), ceps.size - 1)
    q = np.arange(qmin, qmax)
    seg = ceps[q]
    slope, intercept = np.polyfit(q, seg, 1)
    return float((seg - (slope * q + intercept)).max() * 8.685889638)


def syllable_metrics(x: np.ndarray, start: int, end: int, sr: int = _SR) -> dict | None:
    """Đo CPP/HNR/jitter/AP 2–6 kHz trên [start, end) mẫu (nucleus)."""
    import pyworld as pw

    seg = np.asarray(x[start:end], dtype=np.float64)
    if seg.size < int(sr * 0.05):
        return None
    f0, t = pw.harvest(seg, sr, frame_period=10.0)
    vuv = f0 > 0
    if vuv.sum() < 3:
        return None
    sp = pw.cheaptrick(seg, f0, t, sr)
    ap = pw.d4c(seg, f0, t, sr)
    freqs = np.fft.rfftfreq(sp.shape[1] * 2 - 2, 1.0 / sr)
    band = (freqs >= 2000) & (freqs <= 6000)
    hnr = np.mean(10 * np.log10(np.clip(1 / (ap[vuv] + 1e-12), 1e-6, 1e6)))
    ap_band = float(np.mean(ap[vuv][:, band])) if band.any() else float("nan")
    jit = (float(np.mean(np.abs(np.diff(f0[vuv])) / np.maximum(f0[vuv][1:], 1e-6)) * 100)
           if vuv.sum() > 2 else float("nan"))
    # CPP trên các frame hữu thanh, cửa sổ 40 ms quanh tâm frame
    win = int(sr * 0.04)
    cpps = []
    for idx in np.flatnonzero(vuv):
        center = int(t[idx] * sr)
        a, b = max(0, center - win // 2), min(seg.size, center + win // 2)
        if b - a >= 64:
            value = cpp_db(seg[a:b], sr)
            if np.isfinite(value):
                cpps.append(value)
    return {"cpp": float(np.mean(cpps)) if cpps else float("nan"), "hnr": float(hnr),
            "jitter": jit, "ap_2_6k": ap_band, "frames": int(vuv.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--sharpen-alpha", type=float, default=1.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    import yaml

    from vifasttts.data.dataset_v2 import SlotDataset
    from vifasttts.infer import Synthesizer
    from vifasttts.runtime.postprocess import sharpen_mel

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json"))
                           .read_text(encoding="utf-8"))
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir,
                          min_syllables=2)
    out = Path(args.out)
    rows_page: list[dict] = []

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
            mel = outputs["mel"][0]
            wav_t = engine.vocoder(sharpen_mel(mel, args.sharpen_alpha).unsqueeze(0))[0]
        synth = np.asarray(wav_t.float().cpu().numpy(), dtype=np.float32)
        real_path = _ROOT / str(row["audio_path"])
        real_full = _read_wav(real_path)
        f0_frame, f1_frame = int(row["frame_range"][0]), int(row["frame_range"][1])
        real = real_full[f0_frame * _HOP: f1_frame * _HOP]
        n = min(real.size, synth.size)
        real, synth = real[:n], synth[:n]

        syllables = row["syllables"]
        durations = [int(d) for d in row["durations"]]
        gaps = [int(g) for g in row["gaps"]]
        seg4 = item["seg_durations"].tolist()
        lead = int(item["lead"])
        cursor = lead
        clip_dir = out / "clips" / str(row["id"])
        page_rows = []
        for i, syl in enumerate(syllables):
            dur = durations[i]
            o, m_, nuc, _c = seg4[i]
            nucleus_start = (cursor + o + m_) * _HOP
            nucleus_end = nucleus_start + max(1, nuc) * _HOP
            tone = str((syl.get("abstract") or {}).get("tone") or "?")
            raw = str(syl.get("raw", ""))
            real_m = syllable_metrics(real, nucleus_start, nucleus_end)
            synth_m = syllable_metrics(synth, nucleus_start, nucleus_end)
            if real_m is None or synth_m is None:
                cursor += dur + (gaps[i] if i < len(gaps) else 0)
                continue
            name = f"{i:02d}_{raw}"
            # Phát TRỌN âm tiết (cả coda) — trước đây cắt ở nucleus+33 ms làm mất coda,
            # dễ nghe nhầm "mình" thành "mìn".
            syllable_start = max(0, cursor * _HOP - 400)
            syllable_end = min(n, (cursor + dur) * _HOP + 400)
            _write_wav(clip_dir / "real" / f"{name}.wav", real[syllable_start:syllable_end])
            _write_wav(clip_dir / "synth" / f"{name}.wav", synth[syllable_start:syllable_end])
            page_rows.append({"i": i, "raw": raw, "tone": tone, "real": real_m, "synth": synth_m,
                              "name": name})
            cursor += dur + (gaps[i] if i < len(gaps) else 0)
        if page_rows:
            rows_page.append({"id": str(row["id"]), "text": str(row.get("text", ""))[:90],
                              "rows": page_rows})
        print(f"[roughness] {row['id']}: {len(page_rows)} âm tiết", flush=True)

    def fmt(value: float, digits: int = 2) -> str:
        return "—" if not np.isfinite(value) else f"{value:.{digits}f}"

    lines = ["<!doctype html><html lang='vi'><head><meta charset='utf-8'>",
             "<title>Độ rè theo âm tiết — synth vs thật</title>",
             ("<style>body{font-family:sans-serif;max-width:1500px;margin:20px auto}"
              "table{border-collapse:collapse;width:100%;font-size:13px}"
              "td,th{border:1px solid #ccc;padding:4px;vertical-align:middle}"
              "audio{width:150px;height:26px}.clip{margin-top:22px}h2{font-size:17px}"
              ".bad{background:#fde8e8}.warn{background:#fff6e0}small{color:#666}</style></head><body>"),
             "<h1>Rè theo âm tiết — bản thật vs model (TF, đúng nhãn)</h1>",
             ("<p><b>CPP</b> (dB, cao = giọng sạch/hài rõ) · <b>HNR</b> (dB, cao = ít tạp) · "
              "<b>jitter</b> (%, thấp = đều chu kỳ) · <b>AP 2–6 kHz</b> (0–1, cao = nhiều tạp vùng xì). "
              "Nếu <b>thật</b> đã có CPP thấp/HNR thấp ⇒ giọng/thu âm vốn rè; nếu chỉ <b>synth</b> xấu "
              "⇒ model thêm rè. Dòng đỏ = Δ CPP ≤ −2 dB (model làm rè hơn rõ).</p>")]
    for clip in rows_page:
        lines.append(f"<div class='clip'><h2>{html.escape(clip['id'])} <small>{html.escape(clip['text'])}…</small></h2>")
        lines.append("<table><tr><th>#</th><th>âm tiết</th><th>thanh</th><th>thật</th><th>synth</th>"
                     "<th>CPP thật</th><th>CPP synth</th><th>ΔCPP</th><th>HNR thật</th><th>HNR synth</th>"
                     "<th>ΔHNR</th><th>jitter thật</th><th>jitter synth</th><th>AP 2–6k thật</th><th>AP 2–6k synth</th></tr>")
        for r in clip["rows"]:
            d_cpp = r["synth"]["cpp"] - r["real"]["cpp"]
            d_hnr = r["synth"]["hnr"] - r["real"]["hnr"]
            cls = " class='bad'" if d_cpp <= -2.0 else (" class='warn'" if d_cpp <= -1.0 else "")
            lines.append(
                f"<tr{cls}><td>{r['i']}</td><td>{html.escape(r['raw'])}</td><td>{html.escape(r['tone'])}</td>"
                f"<td><audio controls preload='none' src='clips/{clip['id']}/real/{r['name']}.wav'></audio></td>"
                f"<td><audio controls preload='none' src='clips/{clip['id']}/synth/{r['name']}.wav'></audio></td>"
                f"<td>{fmt(r['real']['cpp'])}</td><td>{fmt(r['synth']['cpp'])}</td>"
                f"<td><b>{d_cpp:+.2f}</b></td><td>{fmt(r['real']['hnr'])}</td><td>{fmt(r['synth']['hnr'])}</td>"
                f"<td>{d_hnr:+.2f}</td><td>{fmt(r['real']['jitter'])}</td><td>{fmt(r['synth']['jitter'])}</td>"
                f"<td>{fmt(r['real']['ap_2_6k'], 3)}</td><td>{fmt(r['synth']['ap_2_6k'], 3)}</td></tr>")
        lines.append("</table></div>")
    lines.append("</body></html>")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text("\n".join(lines), encoding="utf-8")

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rows_page, ensure_ascii=False, indent=1), encoding="utf-8")

    all_rows = [r for clip in rows_page for r in clip["rows"]]
    if all_rows:
        d_cpp = np.array([r["synth"]["cpp"] - r["real"]["cpp"] for r in all_rows])
        d_hnr = np.array([r["synth"]["hnr"] - r["real"]["hnr"] for r in all_rows])
        print(f"\nTổng {len(all_rows)} âm tiết · ΔCPP trung vị {np.median(d_cpp):+.2f} dB "
              f"(≤−2 dB: {(d_cpp <= -2).sum()}) · ΔHNR trung vị {np.median(d_hnr):+.2f} dB")
    print(f"Xong: mở {out / 'index.html'}")


if __name__ == "__main__":
    main()

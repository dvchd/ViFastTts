"""tools/v1833_analysis.py — v18.33: so ghép cặp P2 vs G1 (giám khảo + flat/HNR phụ trợ).

Đọc `v1833_judges.json` (đã chấm 300 câu) + WAV trong `v1833_audio`, tính:
- Δ theo từng câu cho BAK/SIG/OVRL/UTMOS/CER/WER và flat_V/HNR (đo lại trên WAV).
- Trung bình/trung vị, KTC 95% **bootstrap theo cụm video** (2.000 lần), tỷ lệ câu tốt hơn.
- Bảng theo video.
In theo luật đã đăng ký ở §25.0; JSON ghi trước khi in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import v1828_gta_eval as E

AUDIO = Path("artifacts/bench/v1833_audio")


def main() -> None:
    data = json.loads(Path("artifacts/bench/v1833_judges.json").read_text(encoding="utf-8"))
    items = {item["id"]: item for item in json.loads((AUDIO / "items.json").read_text(encoding="utf-8"))}
    p2 = data["per_file"]["p2"]
    g1 = data["per_file"]["g1"]

    rows = []
    for key in p2:
        if key not in g1:
            continue
        wave_p2, _ = sf.read(AUDIO / "p2" / f"{key}.wav", dtype="float32")
        wave_g1, _ = sf.read(AUDIO / "g1" / f"{key}.wav", dtype="float32")
        m_p2, m_g1 = E.frame_analysis(wave_p2), E.frame_analysis(wave_g1)
        rows.append({
            "id": key, "video": items[key]["video"], "bucket": items[key]["bucket"],
            "d_bak": g1[key]["dnsmos"]["bak_mos"] - p2[key]["dnsmos"]["bak_mos"],
            "d_sig": g1[key]["dnsmos"]["sig_mos"] - p2[key]["dnsmos"]["sig_mos"],
            "d_ovrl": g1[key]["dnsmos"]["ovrl_mos"] - p2[key]["dnsmos"]["ovrl_mos"],
            "d_utmos": g1[key]["utmos"] - p2[key]["utmos"],
            "d_cer": g1[key]["cer"] - p2[key]["cer"],
            "d_wer": g1[key]["wer"] - p2[key]["wer"],
            "flat_p2": float(np.median(m_p2["flat"][m_p2["voiced"]])) if m_p2["voiced"].any() else float("nan"),
            "flat_g1": float(np.median(m_g1["flat"][m_g1["voiced"]])) if m_g1["voiced"].any() else float("nan"),
            "hnr_p2": float(np.nanmedian(m_p2["hnr"])), "hnr_g1": float(np.nanmedian(m_g1["hnr"])),
        })
    for row in rows:
        row["d_flat"] = row["flat_g1"] - row["flat_p2"]
        row["d_hnr"] = row["hnr_g1"] - row["hnr_p2"]

    fields = ["d_bak", "d_sig", "d_ovrl", "d_utmos", "d_cer", "d_wer", "d_flat", "d_hnr"]
    videos = sorted({row["video"] for row in rows})
    rng = np.random.default_rng(0)
    boot = {field: [] for field in fields}
    for _ in range(2000):
        chosen = rng.choice(len(videos), len(videos), replace=True)
        subset = [row for row in rows if row["video"] in {videos[i] for i in chosen}]
        for field in fields:
            boot[field].append(float(np.nanmedian([row[field] for row in subset])))
    summary = {}
    for field in fields:
        values = np.array([row[field] for row in rows], dtype=float)
        summary[field] = {
            "mean": round(float(np.nanmean(values)), 4),
            "median": round(float(np.nanmedian(values)), 4),
            "ci95": [round(float(np.percentile(boot[field], 2.5)), 4),
                     round(float(np.percentile(boot[field], 97.5)), 4)],
            "better_pct": round(100.0 * float(np.nanmean(values > 0)), 2),
            "n": int(np.isfinite(values).sum()),
        }
    per_video = {}
    for video in videos:
        subset = [row for row in rows if row["video"] == video]
        per_video[video] = {
            "n": len(subset),
            "bak_median": round(float(np.nanmedian([row["d_bak"] for row in subset])), 4),
            "utmos_median": round(float(np.nanmedian([row["d_utmos"] for row in subset])), 4),
            "cer_median": round(float(np.nanmedian([row["d_cer"] for row in subset])), 4),
        }
    # C. thời lượng / pause / trim / speech (cùng định nghĩa v18.32), cho từng điều kiện
    def duration_stats(name: str) -> dict:
        durations, pauses, trims, speech = [], [], [], []
        for key in p2:
            path = AUDIO / name / f"{key}.wav"
            if not path.exists():
                continue
            wave, _ = sf.read(path, dtype="float32")
            frames = int(len(wave) // E.HOP)
            if frames == 0:
                continue
            blocks = np.asarray(wave[: frames * E.HOP], dtype=np.float64).reshape(frames, E.HOP)
            db = 20 * np.log10(np.sqrt(np.maximum((blocks ** 2).mean(axis=1), 1e-24)) + 1e-24)
            peak = float(np.max(db))
            durations.append(len(wave) / 24000.0)
            pauses.append(float(np.mean(db < peak - 35.0)))
            keep = np.flatnonzero(db >= peak - 35.0)
            trims.append((keep[-1] - keep[0] + 1) * E.HOP / 24000.0 if keep.size else 0.0)
            speech.append(int(E.frame_analysis(wave)["voiced"].sum()))
        return {
            "n": len(durations),
            "total_s": round(float(np.sum(durations)), 2),
            "pause_pct": round(100.0 * float(np.mean(pauses)), 2),
            "trim_s": round(float(np.sum(trims)), 2),
            "speech_frames": int(np.sum(speech)),
        }

    result = {"n": len(rows), "videos": len(videos), "summary": summary, "per_video": per_video,
              "duration": {name: duration_stats(name) for name in ("p2", "g1", "real")},
              "rule": "xem §25.0 (đăng ký trước): G1 thắng nếu BAK CI>0, UTMOS ≥ −0,02, CER ≤ +0,002, ≥2/3 video BAK≥0"}
    Path("artifacts/bench/v1833_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

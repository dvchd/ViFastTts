"""tools/v1832_tf_free_diag.py — v18.32: chẩn đoán TF vs free-run (cùng câu, cùng checkpoint).

Điều kiện đo (36 câu của bộ nghe v1832, cùng frontend/vocoder/config, α=0):
  - TF: model(durations=GT, gaps=GT, lead=GT) → vocoder (đây là **duration schedule dùng chung**).
  - free-run: production path `iter_synthesize_sentence` (durations/gaps do model dự đoán).
KHÔNG kéo giãn/cắt đuôi/đệm silence để ép cùng độ dài.

Đo cho mỗi câu: tổng thời lượng, LUFS, phân bố RMS (p10/p50/p90), % và **giây** dưới −50 dBFS,
tỉ lệ speech/pause (cùng quy tắc: frame RMS < max − 35 dB), duration ratio (model/GT),
bản đã cắt silence đầu/cuối và bản chỉ speech.

TF còn đo thêm Δ theo bin cố định của copy-synth (mask lấy từ copy, giống mọi vòng trước).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import v1828_gta_eval as E

HOP = 256
CFG32 = "configs/small_codelung_v32.yaml"
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
G1 = "runs/v1830_g1/ckpt_2000.pt"


def stats(wave: np.ndarray) -> dict:
    """Thống kê mức theo frame 256 mẫu (dBFS), cùng quy tắc speech/pause cho mọi điều kiện."""
    frames = int(len(wave) // HOP)
    if frames == 0:
        return {}
    blocks = np.asarray(wave[: frames * HOP], dtype=np.float64).reshape(frames, HOP)
    db = 20.0 * np.log10(np.sqrt(np.maximum((blocks ** 2).mean(axis=1), 1e-24)) + 1e-24)
    peak = float(np.max(db))
    pause = db < (peak - 35.0)
    below = db < -50.0
    return {
        "duration_s": round(len(wave) / 24000.0, 3),
        "frames": frames,
        "rms_db": {"p10": round(float(np.percentile(db, 10)), 2),
                   "p50": round(float(np.percentile(db, 50)), 2),
                   "p90": round(float(np.percentile(db, 90)), 2),
                   "max": round(peak, 2)},
        "below_minus50": {"frames": int(below.sum()),
                          "pct": round(100.0 * float(below.mean()), 3),
                          "seconds": round(float(below.sum()) * HOP / 24000.0, 3)},
        "pause": {"frames": int(pause.sum()), "pct": round(100.0 * float(pause.mean()), 3),
                  "seconds": round(float(pause.sum()) * HOP / 24000.0, 3)},
        "speech_db_mean": round(float(np.mean(db[~pause])), 2) if (~pause).any() else None,
    }


def trim_silence(wave: np.ndarray, drop_db: float = 35.0) -> np.ndarray:
    """Cắt silence đầu/cuối theo cùng ngưỡng tương đối (chỉ để TÍNH ratio, không ghi âm)."""
    frames = int(len(wave) // HOP)
    if frames == 0:
        return wave
    blocks = np.asarray(wave[: frames * HOP], dtype=np.float64).reshape(frames, HOP)
    db = 20.0 * np.log10(np.sqrt(np.maximum((blocks ** 2).mean(axis=1), 1e-24)) + 1e-24)
    keep = np.flatnonzero(db >= float(np.max(db)) - drop_db)
    if keep.size == 0:
        return wave
    return wave[keep[0] * HOP:(keep[-1] + 1) * HOP]


def main() -> None:
    from vifasttts.eval_text import build_eval_text, legacy_raw_join
    from vifasttts.infer import Synthesizer
    from vifasttts.runtime.loudness import measure_lufs
    from vifasttts.runtime.postprocess import sharpen_mel

    items = json.loads(Path("artifacts/listen_small_filt/v1832_blind/items.json").read_text(encoding="utf-8"))
    rows = {json.loads(x)["id"]: json.loads(x) for x in
            Path("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    engines = {"P2": Synthesizer(cfg, P2, device="mps"), "G1": Synthesizer(cfg, G1, device="mps")}

    per_item = []
    for entry in items:
        row = rows[entry["row"]]
        start, end = row["frame_range"]
        gt_frames = int(end) - int(start)
        gt, _ = sf.read(row["audio_path"], dtype="float32", start=start * HOP, frames=gt_frames * HOP)
        gt = np.asarray(gt, dtype=np.float32)
        text = build_eval_text(row)
        text_legacy = legacy_raw_join(row)
        gaps = list(row["gaps"]) + [0] if len(row["gaps"]) == len(row["durations"]) - 1 else list(row["gaps"])
        record = {"sid": entry["sid"], "row": entry["row"], "bucket": entry["bucket"],
                  "text": text, "text_legacy": text_legacy, "text_mode": "punctuation",
                  "gt_frames": gt_frames, "gt": stats(gt)}
        record["gt"]["lufs"] = round(float(measure_lufs(gt, 24000)), 2)
        record["gt"]["trim_s"] = round(len(trim_silence(gt)) / 24000.0, 3)
        gt_voiced = int(E.frame_analysis(gt)["voiced"].sum())
        waves: dict[str, np.ndarray] = {}
        for name, engine in engines.items():
            from vifasttts.frontend.pipeline import parse_text

            parsed = parse_text(text, resolve_oov=True)
            feats = torch.tensor([[engine.vocabs.encode(it) for it in parsed]], dtype=torch.long, device=engine.device)
            with torch.no_grad():
                out = engine.model(feats, speed=torch.tensor([1.0], device=engine.device),
                                   durations=torch.tensor([row["durations"]], dtype=torch.long, device=engine.device),
                                   gaps=torch.tensor([gaps], dtype=torch.long, device=engine.device),
                                   lead=torch.tensor([0], dtype=torch.long, device=engine.device))
                tf_wave = np.asarray(engine.vocoder(sharpen_mel(out["mel"][0], 0.0).unsqueeze(0))[0].float().cpu()).reshape(-1)
            free_wave = np.concatenate([a for a, _ in engine.iter_synthesize_sentence(text)])
            for condition, wave in (("tf", tf_wave), ("free", free_wave)):
                payload = stats(wave)
                payload["lufs"] = round(float(measure_lufs(wave, 24000)), 2)
                payload["trim_s"] = round(len(trim_silence(wave)) / 24000.0, 3)
                payload["duration_ratio"] = round(payload["duration_s"] / (gt_frames * HOP / 24000.0), 3)
                payload["trim_ratio"] = round(payload["trim_s"] / (len(trim_silence(gt)) / 24000.0), 3)
                voiced = int(E.frame_analysis(wave)["voiced"].sum())
                payload["speech_ratio"] = round(voiced / max(1, gt_voiced), 3)
                record[f"{name}_{condition}"] = payload
            waves[name] = tf_wave
        from vifasttts.data.cache import FeatureCache

        cache35 = FeatureCache("cache/codelung_v35", yaml.safe_load(Path("configs/small_codelung_v35.yaml").read_text(encoding="utf-8")))
        mel = cache35.get(row)["mel"][start:end].to(engines["P2"].device)
        with torch.no_grad():
            copy_wave = np.asarray(engines["P2"].vocoder(sharpen_mel(mel, 0.0).unsqueeze(0))[0].float().cpu()).reshape(-1)
        copy_metrics = E.frame_analysis(copy_wave)
        record["copy"] = {"lufs": round(float(measure_lufs(copy_wave, 24000)), 2), **stats(copy_wave)}
        for name2, wave2 in waves.items():
            metrics = E.frame_analysis(wave2)
            n = min(metrics["db"].size, copy_metrics["db"].size)
            mask = copy_metrics["db"][:n] < -50.0
            if mask.any():
                record[f"{name2}_tf"]["delta_bin50"] = {
                    "mask_frames": int(mask.sum()),
                    "d_median": round(float(np.median(metrics["db"][:n][mask] - copy_metrics["db"][:n][mask])), 2),
                    "d_energy": round(10.0 * float(np.log10(
                        max(float(np.sum(10 ** (metrics["db"][:n][mask] / 10.0))), 1e-30)
                        / max(float(np.sum(10 ** (copy_metrics["db"][:n][mask] / 10.0))), 1e-30))), 2),
                }
        per_item.append(record)

    # tổng hợp
    summary = {}
    for prefix in ("gt", "P2_tf", "P2_free", "G1_tf", "G1_free", "copy"):
        subset = [item[prefix] for item in per_item if prefix in item]
        total_s = sum(s["duration_s"] for s in subset)
        below_s = sum(s["below_minus50"]["seconds"] for s in subset)
        pause_s = sum(s["pause"]["seconds"] for s in subset)
        summary[prefix] = {
            "items": len(subset),
            "total_s": round(total_s, 2),
            "below_minus50_s": round(below_s, 2),
            "below_minus50_pct": round(100.0 * below_s / total_s, 2) if total_s else None,
            "pause_s": round(pause_s, 2),
            "pause_pct": round(100.0 * pause_s / total_s, 2) if total_s else None,
            "lufs_mean": round(float(np.mean([s["lufs"] for s in subset])), 2),
            "rms_p50_mean": round(float(np.mean([s["rms_db"]["p50"] for s in subset])), 2),
            "duration_ratio_mean": round(float(np.mean([s.get("duration_ratio", float("nan")) for s in subset])), 3)
            if any("duration_ratio" in s for s in subset) else None,
            "trim_ratio_mean": round(float(np.mean([s.get("trim_ratio", float("nan")) for s in subset])), 3)
            if any("trim_ratio" in s for s in subset) else None,
            "speech_ratio_mean": round(float(np.mean([s.get("speech_ratio", float("nan")) for s in subset])), 3)
            if any("speech_ratio" in s for s in subset) else None,
        }

    out = {
        "items": len(per_item),
        "summary": summary,
        "per_item": per_item,
        "definitions": {
            "pause_rule": "frame RMS < max(frame RMS) − 35 dB (cùng quy tắc cho mọi điều kiện)",
            "below_minus50": "frame RMS (256 mẫu) < −50 dBFS, không chuẩn hoá LUFS",
            "delta_bin50": "TF: mask lấy từ copy-synth (RMS < −50), cố định giữa P2/G1; Δ = model − copy",
            "duration_ratio": "thời lượng model / thời lượng GT (frame_range, gồm lead+gaps)",
            "trim_ratio": "sau khi cắt silence đầu/cuối (ngưỡng max−35 dB) ở cả hai",
            "speech_ratio": "số frame hữu thanh (autocorr ≥ 0,45) của model / của GT",
        },
    }
    Path("artifacts/bench/v1832_tf_free.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

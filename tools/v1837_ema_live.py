"""v18.37 — kiểm tra confound EMA/live trên 36 câu cố định.

Tạo wrapper **mới** dưới `artifacts/checkpoints/v1837/` (không sửa checkpoint gốc):
  p2_ema  ← `_ckpt/step_02000.pt` (model = EMA)
  p2_live ← `runs/adv_p2_20k/ckpt_2000.pt["model"]`
  g1_live ← `runs/v1830_g1/ckpt_2000.pt["model"]`
  g1_ema  ← `runs/v1830_g1/ckpt_2000.pt["ema"]`

Cùng text **có dấu câu**, cùng sentence mode, cùng Vocos pretrained, α=0. Chấm
BAK/UTMOS bằng `tools/v1833_judges.py`; ghép cặp theo câu, bootstrap theo video.
Đây là kiểm tra confound trên tập phát triển — **không** phải xác nhận độc lập và
không tự đổi production.
"""

from __future__ import annotations

import argparse
import hashlib
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

from vifasttts.data.split import video_id_of_row
from vifasttts.eval_stats import outcome_counts
from vifasttts.eval_text import build_eval_text, legacy_raw_join

CFG32 = "configs/small_codelung_v32.yaml"
P2_EMA = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
P2_LIVE = "runs/adv_p2_20k/ckpt_2000.pt"
G1 = "runs/v1830_g1/ckpt_2000.pt"
OUT = Path("artifacts/bench/v1837_ema_live")
WRAP = Path("artifacts/checkpoints/v1837")

MODELS = {
    "p2_ema": (P2_EMA, "model"),
    "p2_live": (P2_LIVE, "model"),
    "g1_ema": (G1, "ema"),
    "g1_live": (G1, "model"),
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_wrappers(cfg: dict) -> dict:
    base = torch.load(P2_EMA, map_location="cpu", weights_only=False)
    f0_stats = base["f0_stats"]
    mel_hash = base["mel_hash"]
    WRAP.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, (source, weights_key) in MODELS.items():
        target = WRAP / f"{name}.pt"
        if not target.exists():
            payload = torch.load(source, map_location="cpu", weights_only=False)
            model = payload[weights_key]
            state = {
                "model": model,
                "config": cfg,
                "f0_stats": f0_stats,
                "mel_hash": mel_hash,
                "arch": payload.get("arch", "v2_slot5"),
                "ema_decay": payload.get("ema_decay"),
                "step": payload.get("step"),
                "provenance": {
                    "created_by": "tools/v1837_ema_live.py",
                    "source": source,
                    "source_sha256": sha256(source),
                    "weights_key": weights_key,
                    "note": "wrapper mới; không sửa checkpoint gốc",
                },
            }
            torch.save(state, target)
        payload = torch.load(target, map_location="cpu", weights_only=False)
        missing = [key for key in base["model"] if key not in payload["model"]]
        extra = [key for key in payload["model"] if key not in base["model"]]
        shape_mismatch = [key for key in base["model"] if key in payload["model"]
                          and tuple(base["model"][key].shape) != tuple(payload["model"][key].shape)]
        report[name] = {
            "path": str(target), "sha256": sha256(target),
            "source": source, "source_sha256": sha256(source),
            "weights_key": weights_key,
            "missing_vs_p2": missing, "extra_vs_p2": extra,
            "shape_mismatch_vs_p2": shape_mismatch,
            "complete_keys": not missing and not extra and not shape_mismatch,
        }
    return report


def judge_dir(directory: Path, device: str) -> dict:
    from v1833_judges import dnsmos_scores, to16, utmos_dir

    utmos = utmos_dir(directory, device)
    rows = {}
    for path in sorted(directory.glob("*.wav")):
        wave, _ = sf.read(path, dtype="float32")
        rows[path.stem] = {"utmos": utmos.get(path.stem),
                           "dnsmos": dnsmos_scores(to16(wave))}
    return rows


def paired(reference: dict, candidate: dict, videos: list[str]) -> dict:
    rows = []
    for key, ref in reference.items():
        if key not in candidate:
            continue
        cand = candidate[key]
        rows.append({
            "key": key, "video": videos[key],
            "d_bak": cand["dnsmos"]["bak_mos"] - ref["dnsmos"]["bak_mos"],
            "d_utmos": cand["utmos"] - ref["utmos"],
        })
    video_ids = sorted({row["video"] for row in rows})
    rng = np.random.default_rng(0)
    boot = {"d_bak": [], "d_utmos": []}
    for _ in range(2000):
        chosen = rng.choice(len(video_ids), len(video_ids), replace=True)
        subset = [row for row in rows if row["video"] in {video_ids[i] for i in chosen}]
        for field, values in boot.items():
            values.append(float(np.nanmedian([row[field] for row in subset])))
    out = {"n": len(rows), "videos": len(video_ids), "summary": {}}
    for field, higher_better in (("d_bak", True), ("d_utmos", True)):
        values = np.array([row[field] for row in rows], dtype=float)
        counts = outcome_counts(values, higher_better=higher_better)
        out["summary"][field] = {
            "median": round(float(np.nanmedian(values)), 4),
            "mean": round(float(np.nanmean(values)), 4),
            "ci95": [round(float(np.percentile(boot[field], 2.5)), 4),
                     round(float(np.percentile(boot[field], 97.5)), 4)],
            "higher_better": higher_better, **counts,
        }
    out["per_video_bak"] = {
        video: round(float(np.median([row["d_bak"] for row in rows if row["video"] == video])), 4)
        for video in video_ids
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=36)
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer

    cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    wrappers = create_wrappers(cfg)
    rows = {json.loads(line)["id"]: json.loads(line) for line in
            Path("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()}
    blind = json.loads(Path("artifacts/listen_small_filt/v1832_blind/items.json")
                       .read_text(encoding="utf-8"))[: args.limit]
    items = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    engines = {name: Synthesizer(cfg, str(WRAP / f"{name}.pt"), device=args.device)
               for name in MODELS}
    for entry in blind:
        row = rows[entry["row"]]
        text = build_eval_text(row)
        text_legacy = legacy_raw_join(row)
        for name, engine in engines.items():
            pieces = [audio for audio, _ in engine.iter_synthesize_sentence(text)]
            wave = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
            sf.write(OUT / name / f"{row['id']}.wav", np.asarray(wave, dtype=np.float32), 24000)
        items.append({
            "id": row["id"], "video": video_id_of_row(row), "bucket": entry["bucket"],
            "text": text, "text_legacy": text_legacy, "text_mode": "punctuation",
            "frames_gt": int(row["frame_range"][1]) - int(row["frame_range"][0]),
        })
    (OUT / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    result = {
        "set": "36 câu v18.32, chọn trước khi chấm; text có dấu câu",
        "wrappers": wrappers,
        "items": len(items),
        "videos": len({item["video"] for item in items}),
        "note": "kiểm tra confound trên tập phát triển; không phải xác nhận độc lập",
    }
    if not args.skip_judge:
        judges = {name: judge_dir(OUT / name, args.device) for name in MODELS}
        result["judges"] = {
            name: {
                "n": len(per),
                "utmos_mean": round(float(np.mean([x["utmos"] for x in per.values()])), 4),
                "bak_mean": round(float(np.mean([x["dnsmos"]["bak_mos"] for x in per.values()])), 4),
                "sig_mean": round(float(np.mean([x["dnsmos"]["sig_mos"] for x in per.values()])), 4),
                "ovrl_mean": round(float(np.mean([x["dnsmos"]["ovrl_mos"] for x in per.values()])), 4),
            } for name, per in judges.items()
        }
        videos = {item["id"]: item["video"] for item in items}
        result["paired_vs_p2_ema"] = {
            name: paired(judges["p2_ema"], judges[name], videos)
            for name in ("p2_live", "g1_ema", "g1_live")
        }
        (OUT / "judges.json").write_text(json.dumps(judges, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("items", "videos", "wrappers")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

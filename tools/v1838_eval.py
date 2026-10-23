"""v18.38 — đánh giá cuối probe crop: P2 EMA vs REC vs GAN-legacy vs GAN-fixed.

Chỉ dùng **EMA** của cả 4 bản; cùng 36 câu dev (12 video), text có dấu câu, sentence
mode, Vocos pretrained, α=0. BAK/UTMOS + flat_V/HNR; TF mel diagnostic tách riêng.
Không CER, không tai người, không chọn checkpoint từ nhiều mốc.
"""

from __future__ import annotations

import argparse
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
from v1833_judges import dnsmos_scores, to16, utmos_dir

from vifasttts.data.dataset_v2 import SlotDataset, collate_slot_batch
from vifasttts.data.split import video_id_of_row
from vifasttts.eval_stats import outcome_counts
from vifasttts.eval_text import build_eval_text, legacy_raw_join
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.models.factory import arch_of_checkpoint, build_model, load_model_state

CFG32 = "configs/small_codelung_v32.yaml"
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
RUN = Path("runs/v1838_crop_probe")
BRANCHES = ("rec", "gan_legacy", "gan_fixed")


def sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_metrics(wave: np.ndarray) -> dict:
    metrics = E.frame_analysis(np.asarray(wave, dtype=np.float32))
    voiced = metrics["voiced"]
    return {
        "flat_v": round(float(np.median(metrics["flat"][voiced])), 4) if voiced.any() else None,
        "hnr": round(float(np.nanmedian(metrics["hnr"])), 3) if np.isfinite(metrics["hnr"]).any() else None,
    }


def judge_dir(directory: Path, device: str) -> dict:
    utmos = utmos_dir(directory, device)
    rows = {}
    for path in sorted(directory.glob("*.wav")):
        wave, _ = sf.read(path, dtype="float32")
        rows[path.stem] = {"utmos": utmos.get(path.stem),
                           "dnsmos": dnsmos_scores(to16(wave)),
                           **frame_metrics(wave)}
    return rows


def paired_all(reference: dict, candidate: dict, videos: dict[str, str]) -> dict:
    rows = []
    for key, ref in reference.items():
        if key not in candidate:
            continue
        cand = candidate[key]
        rows.append({
            "key": key, "video": videos[key],
            "d_bak": cand["dnsmos"]["bak_mos"] - ref["dnsmos"]["bak_mos"],
            "d_utmos": cand["utmos"] - ref["utmos"],
            "d_flat": (cand["flat_v"] - ref["flat_v"])
            if cand["flat_v"] is not None and ref["flat_v"] is not None else float("nan"),
            "d_hnr": (cand["hnr"] - ref["hnr"])
            if cand["hnr"] is not None and ref["hnr"] is not None else float("nan"),
        })
    video_ids = sorted({row["video"] for row in rows})
    rng = np.random.default_rng(0)
    fields = ("d_bak", "d_utmos", "d_flat", "d_hnr")
    boot = {field: [] for field in fields}
    for _ in range(2000):
        chosen = rng.choice(len(video_ids), len(video_ids), replace=True)
        subset = [row for row in rows if row["video"] in {video_ids[i] for i in chosen}]
        for field, values in boot.items():
            values.append(float(np.nanmedian([row[field] for row in subset])))
    higher_better = {"d_bak": True, "d_utmos": True, "d_flat": False, "d_hnr": True}
    summary = {}
    for field in fields:
        values = np.array([row[field] for row in rows], dtype=float)
        summary[field] = {
            "median": round(float(np.nanmedian(values)), 4),
            "mean": round(float(np.nanmean(values)), 4),
            "ci95": [round(float(np.percentile(boot[field], 2.5)), 4),
                     round(float(np.percentile(boot[field], 97.5)), 4)],
            "higher_better": higher_better[field],
            **outcome_counts(values, higher_better=higher_better[field]),
        }
    return {"n": len(rows), "videos": len(video_ids), "summary": summary,
            "per_video_bak": {video: round(float(np.median(
                [row["d_bak"] for row in rows if row["video"] == video])), 4)
                for video in video_ids},
            "per_video_utmos": {video: round(float(np.median(
                [row["d_utmos"] for row in rows if row["video"] == video])), 4)
                for video in video_ids}}


def make_eval_wrapper(source: Path, out: Path, p2: dict) -> Path:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = {
        "model": payload["ema"],
        "config": payload["config"],
        "f0_stats": p2["f0_stats"],
        "mel_hash": p2["mel_hash"],
        "arch": payload.get("arch", "v2_slot5"),
        "provenance": {
            "created_by": "tools/v1838_eval.py",
            "source": str(source),
            "source_sha256": sha256(source),
            "weights_key": "ema",
            "note": "wrapper eval EMA; không sửa checkpoint gốc",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    return out


def tf_diagnostic(items: list[dict], dataset: SlotDataset,
                  model_paths: dict[str, str], cfg: dict) -> dict:
    vocab = ComponentVocabs.default()
    sizes = [len(vocab.onset), len(vocab.medial), len(vocab.nucleus), len(vocab.coda),
             len(vocab.tone), len(vocab.boundary), len(vocab.punctuation)]
    by_id = {row["id"]: index for index, row in enumerate(dataset.rows)}
    result = {}
    for name, path in model_paths.items():
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(cfg, sizes, arch=arch_of_checkpoint(state))
        load_model_state(model, state["model"])
        if state.get("f0_stats"):
            model.load_f0_stats(dict(state["f0_stats"]))
        model.eval()
        totals = {"all": [], "voiced": []}
        for item in items:
            index = by_id[item["id"]]
            batch = collate_slot_batch([dataset[index]])
            with torch.no_grad():
                out = model(batch["features"], durations=batch["durations"],
                            seg_durations=batch["seg_durations"], gaps=batch["gaps"],
                            lead=batch["lead"])
            pred, target = out["mel"][0], batch["mel"][0]
            frames = min(pred.size(0), target.size(0))
            pred, target = pred[:frames], target[:frames]
            valid = torch.arange(frames) < int(batch["frame_lengths"][0])
            voiced = valid & (batch["vuv"][0][:frames] >= 0.5)
            error = (pred - target).abs().mean(-1)
            totals["all"].append(float(error[valid].mean()))
            if voiced.any():
                totals["voiced"].append(float(error[voiced].mean()))
        result[name] = {
            "mel_l1_z_all": round(float(np.mean(totals["all"])), 4),
            "mel_l1_z_all_median": round(float(np.median(totals["all"])), 4),
            "mel_l1_z_voiced": round(float(np.mean(totals["voiced"])), 4)
            if totals["voiced"] else None,
            "n": len(totals["all"]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(RUN))
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--items", default="artifacts/listen_small_filt/v1832_blind/items.json")
    parser.add_argument("--g-step", type=int, default=200)
    parser.add_argument("--limit", type=int, default=36)
    parser.add_argument("--skip-tf", action="store_true")
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer

    run = Path(args.run)
    out = Path(args.out or run / "eval")
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    p2 = torch.load(P2, map_location="cpu", weights_only=False)
    rows_by_id = {json.loads(line)["id"]: json.loads(line) for line in
                  Path("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
                  .read_text(encoding="utf-8").splitlines() if line.strip()}
    blind = json.loads(Path(args.items).read_text(encoding="utf-8"))[: args.limit]

    model_paths = {"p2_ema": P2}
    for branch in BRANCHES:
        source = run / f"{branch}_g{args.g_step:03d}.pt"
        if not source.exists():
            raise FileNotFoundError(source)
        model_paths[branch] = str(make_eval_wrapper(source, run / f"{branch}_eval_ema.pt", p2))

    items = []
    for name in model_paths:
        (out / name).mkdir(parents=True, exist_ok=True)
    engines = {name: Synthesizer(cfg, path, device=args.device)
               for name, path in model_paths.items()}
    for entry in blind:
        row = rows_by_id[entry["row"]]
        text = build_eval_text(row)
        text_legacy = legacy_raw_join(row)
        for name, engine in engines.items():
            pieces = [audio for audio, _ in engine.iter_synthesize_sentence(text)]
            wave = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
            sf.write(out / name / f"{row['id']}.wav", np.asarray(wave, dtype=np.float32), 24000)
        items.append({"id": row["id"], "video": video_id_of_row(row),
                      "bucket": entry["bucket"], "text": text,
                      "text_legacy": text_legacy, "text_mode": "punctuation"})
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    videos = {item["id"]: item["video"] for item in items}
    judges = {name: judge_dir(out / name, args.device) for name in model_paths}
    (out / "judges.json").write_text(json.dumps(judges, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    means = {}
    for name, per in judges.items():
        flats = [row["flat_v"] for row in per.values() if row["flat_v"] is not None]
        hnrs = [row["hnr"] for row in per.values() if row["hnr"] is not None]
        means[name] = {
            "n": len(per),
            "bak": round(float(np.mean([row["dnsmos"]["bak_mos"] for row in per.values()])), 4),
            "utmos": round(float(np.mean([row["utmos"] for row in per.values()])), 4),
            "flat_v": round(float(np.mean(flats)), 4) if flats else None,
            "hnr": round(float(np.mean(hnrs)), 4) if hnrs else None,
        }
    result = {
        "set": "36 câu v1832 (12 video, 3 bucket), text punctuation, sentence mode",
        "run": str(run),
        "g_step": args.g_step,
        "source_commit": json.loads((run / "resolved.json").read_text(encoding="utf-8"))["source_commit"],
        "batch_order_hash": json.loads((run / "resolved.json").read_text(encoding="utf-8"))["batch_order_hash"],
        "weights": "EMA cho cả 4 bản; không dùng live",
        "items": len(items), "videos": len(set(videos.values())),
        "means": means,
        "paired_vs_p2_ema": {name: paired_all(judges["p2_ema"], judges[name], videos)
                             for name in BRANCHES},
        "paired_between": {
            "gan_fixed_vs_legacy": paired_all(judges["gan_legacy"], judges["gan_fixed"], videos),
            "gan_fixed_vs_rec": paired_all(judges["rec"], judges["gan_fixed"], videos),
            "gan_legacy_vs_rec": paired_all(judges["rec"], judges["gan_legacy"], videos),
        },
        "note": "probe dev; không phải xác nhận độc lập; không dùng CER/tai người",
    }
    if not args.skip_tf:
        dataset = SlotDataset("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl", cfg,
                              p2["f0_stats"], cache_dir="cache/codelung_v2", min_syllables=1)
        tf = tf_diagnostic([{"id": rows_by_id[entry["row"]]["id"]} for entry in blind],
                           dataset, model_paths, cfg)
        result["tf_diagnostic"] = tf
        (out / "tf_diagnostic.json").write_text(json.dumps(tf, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(json.dumps({"means": means,
                      "paired_vs_p2": result["paired_vs_p2_ema"],
                      "paired_between": result["paired_between"],
                      "tf": result.get("tf_diagnostic")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

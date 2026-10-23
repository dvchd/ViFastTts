"""v18.39 — audit UTMOSv2 + chấm lại có seed (không render lại audio).

Phát hiện từ source `utmosv2`:
- `MultiSpecDataset.__getitem__` gọi `select_random_start` (np.random.randint) và
  `np.random.beta` cho mixup **cả ở phase prediction**;
- `SSLDataset.__getitem__` cũng gọi `select_random_start`;
- không có seed trong package. Vì vậy `utmos_dir(..., num_repetitions=1)` cũ là
  **một lần random crop/mixup**, không tái lập và không dùng TTA 5 (config ghi
  `inference.num_tta=5`).

Tool này:
1) Chọn cố định 12 câu từ 36 câu dev (1/video, xoay bucket) trước khi chấm.
2) Chấm 5 seed cố định [0..4], lặp seed 0 lần 2, trên WAV đã có của 4 bản.
3) Báo per-file range, delta per-seed, thứ hạng, same-seed vs different-seed.
4) Chấm lại 36 câu bằng trung bình 5 seed (không render lại audio) và so v18.38.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from vifasttts.data.split import video_id_of_row

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SEEDS = (0, 1, 2, 3, 4)
BRANCHES = ("p2_ema", "rec", "gan_legacy", "gan_fixed")
AUDIO_ROOT = Path("runs/v1838_crop_probe/eval")
V1838_RESULT = Path("runs/v1838_crop_probe/eval/result.json")
OUT = Path("artifacts/bench/v1839_utmos_audit")
UTMOS_CKPT = Path.home() / ".cache/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth"
UTMOS_CFG = Path(".venv/lib/python3.12/site-packages/utmosv2/config/fusion_stage3.py")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_12(blind: list[dict], rows: dict[str, dict]) -> list[dict]:
    """1 câu/video thật, bucket xoay short/mid/long → 4 mỗi bucket."""
    by_video: dict[str, list[dict]] = {}
    for item in blind:
        video = video_id_of_row(rows[item["row"]])
        item = dict(item)
        item["video_actual"] = video
        by_video.setdefault(video, []).append(item)
    picked = []
    for index, video in enumerate(sorted(by_video)):
        items = by_video[video]
        bucket = ("short", "mid", "long")[index % 3]
        match = [item for item in items if item["bucket"] == bucket]
        if not match:
            match = items
        picked.append(match[0])
    if len(picked) != 12:
        raise RuntimeError(f"cần 12 câu, nhận {len(picked)}")
    return picked


def score_file(model, path: Path, seed: int, device: str) -> float:
    """Một pass UTMOS với seed cố định; không đổi model/config."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    with torch.no_grad():
        value = model.predict(
            input_path=str(path), device=device, num_repetitions=1,
            remove_silent_section=True, verbose=False,
        )
    return float(value)


def mean(values):
    return float(np.mean(values)) if values else None


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": round(float(array.mean()), 4),
        "median": round(float(np.median(array)), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
        "range": round(float(array.max() - array.min()), 4),
        "std": round(float(array.std(ddof=0)), 4),
        "n": int(array.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", default="artifacts/listen_small_filt/v1832_blind/items.json")
    parser.add_argument("--audio-root", default=str(AUDIO_ROOT))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--skip-rescore36", action="store_true")
    args = parser.parse_args()

    import utmosv2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blind = json.loads(Path(args.items).read_text(encoding="utf-8"))
    rows = {json.loads(line)["id"]: json.loads(line) for line in
            Path("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()}
    selected = select_12(blind, rows)[: args.limit]

    selection = []
    for item in selected:
        row = rows[item["row"]]
        wavs = {}
        for branch in BRANCHES:
            path = Path(args.audio_root) / branch / f"{row['id']}.wav"
            info = sf.info(str(path))
            wavs[branch] = {
                "path": str(path), "sha256": sha256(path),
                "sample_rate": int(info.samplerate), "samples": int(info.frames),
                "duration_s": round(info.frames / info.samplerate, 4),
            }
        selection.append({
            "id": row["id"], "video": item["video_actual"], "bucket": item["bucket"],
            "text": " ".join(str(s.get("raw") or s.get("normalized")) for s in row["syllables"]),
            "wavs": wavs,
        })
    (out / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=1),
                                        encoding="utf-8")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    model = utmosv2.create_model(pretrained=True, config="fusion_stage3",
                                 fold=0, seed=42, device=args.device)
    audit = {
        "utmosv2_version": getattr(utmosv2, "__version__", None),
        "config": "fusion_stage3",
        "fold": 0,
        "checkpoint": str(UTMOS_CKPT),
        "checkpoint_sha256": sha256(UTMOS_CKPT),
        "config_path": str(UTMOS_CFG),
        "config_sha256": sha256(UTMOS_CFG),
        "device": args.device,
        "model_eval": True,
        "torch_no_grad": True,
        "num_repetitions_per_call": 1,
        "remove_silent_section": True,
        "predict_dataset": "sarulab (default)",
        "input": "WAV 24 kHz; librosa.load native → librosa.resample 16 kHz",
        "random_source": "utmosv2.dataset._utils.select_random_start (np.random.randint), multi_spec mixup np.random.beta; no seed in package",
        "seeds": list(SEEDS),
        "same_seed_repeat": 0,
        "n_items": len(selection),
        "items": {},
    }
    scores: dict[str, dict[str, dict[int, float]]] = {branch: {} for branch in BRANCHES}
    repeats: dict[str, dict[str, float]] = {branch: {} for branch in BRANCHES}
    for entry in selection:
        item_id = entry["id"]
        for branch in BRANCHES:
            scores[branch][item_id] = {}
            path = Path(entry["wavs"][branch]["path"])
            for seed in SEEDS:
                scores[branch][item_id][seed] = round(score_file(model, path, seed, args.device), 4)
            repeats[branch][item_id] = round(score_file(model, path, 0, args.device), 4)
        audit["items"][item_id] = {
            "video": entry["video"], "bucket": entry["bucket"],
            "scores": {branch: scores[branch][item_id] for branch in BRANCHES},
            "repeat_seed0": {branch: repeats[branch][item_id] for branch in BRANCHES},
        }

    # a) per-file variability
    per_file = {}
    for branch in BRANCHES:
        values = [summary(list(scores[branch][item].values())) for item in scores[branch]]
        ranges = [value["range"] for value in values]
        per_file[branch] = {
            "mean_range": round(float(np.mean(ranges)), 4),
            "median_range": round(float(np.median(ranges)), 4),
            "max_range": round(float(np.max(ranges)), 4),
            "per_item": {item: summary(list(scores[branch][item].values()))
                         for item in scores[branch]},
        }

    # b) paired delta per seed
    pairs = {
        "rec_vs_p2": ("p2_ema", "rec"),
        "legacy_vs_p2": ("p2_ema", "gan_legacy"),
        "fixed_vs_p2": ("p2_ema", "gan_fixed"),
        "fixed_vs_legacy": ("gan_legacy", "gan_fixed"),
        "fixed_vs_rec": ("rec", "gan_fixed"),
    }
    paired = {}
    for name, (ref, cand) in pairs.items():
        per_item = {}
        per_seed = {seed: [] for seed in SEEDS}
        for item in scores["p2_ema"]:
            values = []
            for seed in SEEDS:
                delta = scores[cand][item][seed] - scores[ref][item][seed]
                values.append(delta)
                per_seed[seed].append(delta)
            per_item[item] = {
                **summary(values),
                "sign_by_seed": [1 if value > 0 else (-1 if value < 0 else 0) for value in values],
            }
        paired[name] = {
            "per_item": per_item,
            "per_seed_mean": {str(seed): round(float(np.mean(values)), 4)
                              for seed, values in per_seed.items()},
            "mean_of_seed_means": round(float(np.mean([np.mean(values) for values in per_seed.values()])), 4),
            "seed_range": round(float(max(np.mean(values) for values in per_seed.values())
                                      - min(np.mean(values) for values in per_seed.values())), 4),
        }

    # c) ranking per seed
    ranking = {}
    for seed in SEEDS:
        means = {branch: round(float(np.mean([scores[branch][item][seed]
                                              for item in scores[branch]])), 4)
                 for branch in BRANCHES}
        ranking[str(seed)] = {
            "means": means,
            "order": sorted(means, key=lambda key: means[key], reverse=True),
        }

    # d) same seed vs different seed
    same_seed = {}
    for branch in BRANCHES:
        diffs = [abs(repeats[branch][item] - scores[branch][item][0]) for item in repeats[branch]]
        same_seed[branch] = {
            "max_abs_diff": round(float(max(diffs)), 6),
            "all_equal": bool(max(diffs) <= 1e-6),
        }

    audit.update({
        "per_file": per_file,
        "paired": paired,
        "ranking": ranking,
        "same_seed": same_seed,
        "source_finding": {
            "file": "utmosv2/dataset/multi_spec.py:52-63, utmosv2/dataset/ssl.py:48, utmosv2/dataset/_utils.py:35-37",
            "detail": "random crop + mixup dùng np.random toàn cục, không seed; chạy ở cả prediction",
        },
    })
    (out / "audit12.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps({
        "per_file": {branch: {key: per_file[branch][key] for key in ("mean_range", "median_range", "max_range")}
                     for branch in BRANCHES},
        "paired": {name: {"mean_of_seed_means": paired[name]["mean_of_seed_means"],
                          "seed_range": paired[name]["seed_range"]} for name in paired},
        "ranking": ranking,
        "same_seed": same_seed,
    }, ensure_ascii=False, indent=1))

    if args.skip_rescore36:
        return

    # Chấm lại đúng 36 câu: trung bình 5 seed cho từng file, không render lại.
    print("rescore 36...", flush=True)
    rescore: dict[str, dict[str, float]] = {branch: {} for branch in BRANCHES}
    per_seed_scores: dict[str, dict[str, dict[int, float]]] = {branch: {} for branch in BRANCHES}
    video_of = {}
    for entry in blind:
        row_id = entry["row"]
        video_of[row_id] = video_id_of_row(rows[row_id])
        for branch in BRANCHES:
            path = Path(args.audio_root) / branch / f"{row_id}.wav"
            values = []
            per_seed_scores[branch][row_id] = {}
            for seed in SEEDS:
                value = round(score_file(model, path, seed, args.device), 4)
                values.append(value)
                per_seed_scores[branch][row_id][seed] = value
            rescore[branch][row_id] = round(float(np.mean(values)), 4)

    # BAK lấy từ v18.38 (cùng WAV), không chấm lại DNSMOS.
    old = json.loads(V1838_RESULT.read_text(encoding="utf-8"))
    old_judges = json.loads((AUDIO_ROOT / "judges.json").read_text(encoding="utf-8"))
    bak = {branch: {key: value["dnsmos"]["bak_mos"] for key, value in old_judges[branch].items()}
           for branch in BRANCHES}

    old_pairs = {
        "rec_vs_p2": old["paired_vs_p2_ema"]["rec"]["summary"]["d_utmos"]["median"],
        "legacy_vs_p2": old["paired_vs_p2_ema"]["gan_legacy"]["summary"]["d_utmos"]["median"],
        "fixed_vs_p2": old["paired_vs_p2_ema"]["gan_fixed"]["summary"]["d_utmos"]["median"],
        "fixed_vs_legacy": old["paired_between"]["gan_fixed_vs_legacy"]["summary"]["d_utmos"]["median"],
        "fixed_vs_rec": old["paired_between"]["gan_fixed_vs_rec"]["summary"]["d_utmos"]["median"],
    }

    def paired_rescore(ref: str, cand: str, old_value: float | None = None) -> dict:
        deltas = [rescore[cand][key] - rescore[ref][key] for key in rescore[ref]]
        return {
            "n": len(deltas),
            "median": round(float(np.median(deltas)), 4),
            "mean": round(float(np.mean(deltas)), 4),
            "wins": sum(1 for value in deltas if value > 0),
            "losses": sum(1 for value in deltas if value < 0),
            "old_v1838_median_d_utmos": old_value,
        }

    videos = sorted(set(video_of.values()))
    per_video = {
        branch: {
            video: round(float(np.mean([rescore[branch][key] for key in rescore[branch]
                                         if video_of[key] == video])), 4)
            for video in videos
        }
        for branch in BRANCHES
    }
    per_video_delta = {
        name: {
            video: round(float(np.mean([
                rescore[cand][key] - rescore[ref][key] for key in rescore[ref]
                if video_of[key] == video])), 4)
            for video in videos
        }
        for name, (ref, cand) in {
            "rec_vs_p2": ("p2_ema", "rec"),
            "legacy_vs_p2": ("p2_ema", "gan_legacy"),
            "fixed_vs_p2": ("p2_ema", "gan_fixed"),
            "fixed_vs_legacy": ("gan_legacy", "gan_fixed"),
            "fixed_vs_rec": ("rec", "gan_fixed"),
        }.items()
    }

    result = {
        "set": "36 câu v1832, WAV v18.38, text punctuation, sentence mode",
        "scoring": "UTMOSv2 fusion_stage3 fold0; 5 seed [0..4], mỗi seed 1 pass; trung bình 5 seed/file",
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "config_sha256": audit["config_sha256"],
        "utmosv2_version": audit["utmosv2_version"],
        "scores_seeded": rescore,
        "per_seed": per_seed_scores,
        "means": {branch: round(float(np.mean(list(rescore[branch].values()))), 4)
                  for branch in BRANCHES},
        "per_video": per_video,
        "per_video_delta": per_video_delta,
        "paired": {
            "rec_vs_p2": paired_rescore("p2_ema", "rec", old_pairs["rec_vs_p2"]),
            "legacy_vs_p2": paired_rescore("p2_ema", "gan_legacy", old_pairs["legacy_vs_p2"]),
            "fixed_vs_p2": paired_rescore("p2_ema", "gan_fixed", old_pairs["fixed_vs_p2"]),
            "fixed_vs_legacy": paired_rescore("gan_legacy", "gan_fixed", old_pairs["fixed_vs_legacy"]),
            "fixed_vs_rec": paired_rescore("rec", "gan_fixed", old_pairs["fixed_vs_rec"]),
        },
        "bak_reference": {branch: round(float(np.mean(list(bak[branch].values()))), 4)
                          for branch in BRANCHES},
        "note": "BAK lấy từ v18.38 cùng WAV; chỉ UTMOS được chấm lại; không render lại audio",
    }
    # Thêm biến thiên theo seed của delta trung bình
    for name, (ref, cand) in {
        "rec_vs_p2": ("p2_ema", "rec"),
        "legacy_vs_p2": ("p2_ema", "gan_legacy"),
        "fixed_vs_p2": ("p2_ema", "gan_fixed"),
        "fixed_vs_legacy": ("gan_legacy", "gan_fixed"),
        "fixed_vs_rec": ("rec", "gan_fixed"),
    }.items():
        per_seed = {}
        for seed in SEEDS:
            values = [per_seed_scores[cand][key][seed] - per_seed_scores[ref][key][seed]
                      for key in per_seed_scores[ref]]
            per_seed[str(seed)] = round(float(np.mean(values)), 4)
        result["paired"][name]["per_seed_mean"] = per_seed
        result["paired"][name]["seed_range"] = round(max(per_seed.values()) - min(per_seed.values()), 4)
    (out / "rescore36.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps({"means": result["means"], "paired": result["paired"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

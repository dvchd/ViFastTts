"""v18.37 — đọc tensor vuv trên tập TF nhỏ (KHÔNG ép vuv, KHÔNG train).

36 câu v18.32, P2 EMA, GT durations/seg/gaps/lead. Đo:
- theo role O/M/N/C: target voiced ratio, predicted logit/probability, BCE sau threshold;
- theo nhóm coda ở slot C: số frame thật, GT-voiced/GT-unvoiced (denominator buzz),
  predicted probability trung bình, BCE frame-level (predicted prob vs vuv frame).

Mục đích: kiểm tra denominator buzz C-mũi và mức tín hiệu vuv thực tế trước khi
cân nhắc bất kỳ can thiệp nào. Không sửa model, không ép C-mũi vô thanh.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

import v1828_gta_eval as E

from vifasttts.data.dataset_v2 import SlotDataset, collate_slot_batch
from vifasttts.data.v2_targets import segment_windows
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.models.factory import arch_of_checkpoint, build_model, load_model_state

CFG32 = "configs/small_codelung_v32.yaml"
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
VAL = "data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl"
ROLES = ("O", "M", "N", "C")


def coda_group(coda: str) -> str:
    if coda in ("M", "N", "NG", "NH"):
        return "nasal"
    if coda in ("P", "T", "C", "CH"):
        return "stop"
    if coda in ("W", "J"):
        return "glide"
    return "open"


def main() -> None:
    cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path("data/codelung/f0_stats_v2.json").read_text(encoding="utf-8"))
    dataset = SlotDataset(VAL, cfg, f0_stats, cache_dir="cache/codelung_v2", min_syllables=1)
    by_id = {row["id"]: index for index, row in enumerate(dataset.rows)}
    items = json.loads(Path("artifacts/listen_small_filt/v1832_blind/items.json")
                       .read_text(encoding="utf-8"))
    ids = [item["row"] for item in items if item["row"] in by_id]
    if len(ids) != len(items):
        missing = [item["row"] for item in items if item["row"] not in by_id]
        raise RuntimeError(f"thiếu row trong val: {missing[:3]}")

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
             len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    state = torch.load(P2, map_location="cpu", weights_only=False)
    model = build_model(cfg, sizes, arch=arch_of_checkpoint(state))
    load_model_state(model, state["model"])
    model.eval()

    role_stats = {role: defaultdict(float) for role in ROLES}
    role_frames = {role: defaultdict(int) for role in ROLES}
    coda_stats = {group: defaultdict(float) for group in ("open", "glide", "nasal", "stop")}
    coda_frames = {group: defaultdict(int) for group in ("open", "glide", "nasal", "stop")}
    rows_out = []

    for row_id in ids:
        index = by_id[row_id]
        row = dataset.rows[index]
        batch = collate_slot_batch([dataset[index]])
        with torch.no_grad():
            out = model(batch["features"], durations=batch["durations"],
                        seg_durations=batch["seg_durations"], gaps=batch["gaps"],
                        lead=batch["lead"])
        logits = out["seg_vuv_logits"][0]           # [N,4]
        target = batch["seg_vuv"][0]                 # [N,4] ratio
        prob = torch.sigmoid(logits)
        vuv = batch["vuv"][0]
        start, end = row["frame_range"]
        real, _ = sf.read(row["audio_path"], dtype="float32", start=start * E.HOP,
                          frames=(end - start) * E.HOP)
        autocorr_voiced = E.frame_analysis(np.asarray(real, dtype=np.float32))["voiced"]
        windows = segment_windows(row, relative=True)
        row_frames = {"id": row_id, "items": len(row["durations"]),
                      "coda_groups": defaultdict(lambda: {"frames": 0, "gt_unvoiced": 0,
                                                          "pred_prob": 0.0, "syllables": 0})}
        for syllable_index, windows_syllable in enumerate(windows):
            coda = coda_group(str(row["syllables"][syllable_index]["abstract"]["coda"]))
            for role, low, high in windows_syllable:
                count = max(0, high - low)
                if count == 0:
                    continue
                role_index = ROLES.index(role)
                voiced = int((vuv[low:high] > 0.5).sum())
                unvoiced = count - voiced
                p = float(prob[syllable_index, role_index])
                role_frames[role]["frames"] += count
                role_frames[role]["gt_voiced"] += voiced
                role_frames[role]["gt_unvoiced"] += unvoiced
                role_stats[role]["target_ratio"] += float(target[syllable_index, role_index]) * count
                role_stats[role]["pred_prob"] += p * count
                role_stats[role]["pred_logit"] += float(logits[syllable_index, role_index]) * count
                role_stats[role]["bce"] += float(F.binary_cross_entropy_with_logits(
                    logits[syllable_index, role_index].reshape(1),
                    (target[syllable_index, role_index] >= 0.5).float().reshape(1))) * count
                auto_role_unvoiced = int((~autocorr_voiced[low:high]).sum())
                role_stats[role]["autocorr_unvoiced"] += auto_role_unvoiced
                role_stats[role]["pred_pos_on_autocorr_unvoiced"] += (
                    p if p >= 0.5 else 0.0) * auto_role_unvoiced
                if role == "C":
                    coda_frames[coda]["frames"] += count
                    coda_frames[coda]["gt_voiced"] += voiced
                    coda_frames[coda]["gt_unvoiced"] += unvoiced
                    coda_stats[coda]["target_ratio"] += float(target[syllable_index, role_index]) * count
                    coda_stats[coda]["pred_prob"] += p * count
                    coda_stats[coda]["pred_logit"] += float(logits[syllable_index, role_index]) * count
                    coda_stats[coda]["bce"] += float(F.binary_cross_entropy_with_logits(
                        logits[syllable_index, role_index].reshape(1),
                        (target[syllable_index, role_index] >= 0.5).float().reshape(1))) * count
                    coda_stats[coda]["frame_bce"] += float(F.binary_cross_entropy(
                        torch.full((count,), p), (vuv[low:high] > 0.5).float(), reduction="sum"))
                    auto_slice = autocorr_voiced[low:high]
                    auto_unvoiced = int((~auto_slice).sum())
                    coda_stats[coda]["autocorr_unvoiced"] += auto_unvoiced
                    coda_stats[coda]["pred_pos_on_autocorr_unvoiced"] += (
                        p if p >= 0.5 else 0.0) * auto_unvoiced
                    group = row_frames["coda_groups"][coda]
                    group["frames"] += count
                    group["gt_unvoiced"] += unvoiced
                    group["pred_prob"] += p * count
                    group["syllables"] += 1
        row_frames["coda_groups"] = dict(row_frames["coda_groups"])
        rows_out.append(row_frames)

    def summarize(stats, frames):
        result = {}
        for key in stats:
            count = frames[key]["frames"]
            result[key] = {
                "frames": count,
                "gt_voiced": frames[key]["gt_voiced"],
                "gt_unvoiced": frames[key]["gt_unvoiced"],
                "target_ratio_mean": round(stats[key]["target_ratio"] / max(1, count), 4),
                "pred_logit_mean": round(stats[key]["pred_logit"] / max(1, count), 4),
                "pred_prob_mean": round(stats[key]["pred_prob"] / max(1, count), 4),
                "bce_weighted_by_frame": round(stats[key]["bce"] / max(1, count), 4),
                "frame_bce_weighted_by_frame": (
                    round(stats[key]["frame_bce"] / max(1, count), 4)
                    if "frame_bce" in stats[key] else None),
                "autocorr_unvoiced": int(stats[key].get("autocorr_unvoiced", 0)),
                "buzz_pct_segment_prob": round(
                    100.0 * stats[key].get("pred_pos_on_autocorr_unvoiced", 0.0)
                    / max(1, stats[key].get("autocorr_unvoiced", 0)), 2),
            }
        return result

    result = {
        "items": len(rows_out),
        "source": "36 câu v18.32, TF GT durations/seg/gaps/lead, P2 EMA",
        "roles": summarize(role_stats, role_frames),
        "coda_C": summarize(coda_stats, coda_frames),
        "buzz_denominator": {
            group: {
                "vuv_unvoiced": coda_frames[group]["gt_unvoiced"],
                "autocorr_unvoiced": int(coda_stats[group].get("autocorr_unvoiced", 0)),
            } for group in coda_frames
        },
        "rows": rows_out,
        "notes": [
            "buzz_denominator = số frame GT-unvoiced trong slot C (vuv<=0.5), không phải số âm tiết",
            "pred_prob là sigmoid(logit) cấp segment, lặp trên frame của slot C",
            "không ép vuv, không dùng ±10, không train",
        ],
    }
    out = Path("artifacts/bench/v1837_vuv_diag.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("items", "roles", "coda_C", "buzz_denominator")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

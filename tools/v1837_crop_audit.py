"""v18.37 — audit crop D trên sampler/collate/crop thực tế (không training).

Lấy 4 batch cố định (SameNBatchSampler, seed 42, batch 8), forward P2 EMA với GT
durations/seg/gaps/lead, gọi `sample_crops(..., return_windows=True)` của
`tools/finetune_adv.py`, rồi phân loại từng row:

  fully_valid / partial_padding / fully_padding / row_shorter_than_crop.

Ghi `artifacts/bench/v1837_crop_audit.json`. Không kết luận chất lượng, không probe.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from finetune_adv import gap_mask, sample_crops

from vifasttts.data.dataset_v2 import SameNBatchSampler, SlotDataset, collate_slot_batch
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.models.factory import arch_of_checkpoint, build_model, load_model_state

CFG32 = "configs/small_codelung_v32.yaml"
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
TRAIN = "data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl"
CACHE = "cache/codelung_v2"


def main() -> None:
    cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path("data/codelung/f0_stats_v2.json").read_text(encoding="utf-8"))
    dataset = SlotDataset(TRAIN, cfg, f0_stats, cache_dir=CACHE, min_syllables=2)
    sampler = SameNBatchSampler(dataset.rows, 8, shuffle=True, seed=42,
                                lengths=dataset.effective_lengths())
    batches = []
    for index, indices in enumerate(sampler):
        if index >= 4:
            break
        batches.append(collate_slot_batch([dataset[i] for i in indices]))
    if len(batches) != 4:
        raise RuntimeError(f"chỉ lấy được {len(batches)} batch")

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
             len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    state = torch.load(P2, map_location="cpu", weights_only=False)
    model = build_model(cfg, sizes, arch=arch_of_checkpoint(state))
    load_model_state(model, state["model"])
    model.eval()

    def audit_batch(batch: dict, label: str, *, use_lengths: bool) -> list[dict]:
        moved = {key: value for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(moved["features"], durations=moved["durations"],
                            seg_durations=moved["seg_durations"], gaps=moved["gaps"],
                            lead=moved["lead"])
        gap = gap_mask(outputs["slot_durations"])
        rng = random.Random(42)
        _real, fake, windows = sample_crops(
            moved["mel"], outputs["mel"], gap, rng,
            lengths=moved["frame_lengths"] if use_lengths else None,
            return_windows=True,
        )
        total = moved["mel"].size(1)
        rows = []
        for row_index, (start, end) in enumerate(windows):
            frame_length = int(moved["frame_lengths"][row_index])
            width = end - start
            valid = max(0, min(end, frame_length) - start)
            padding = width - valid
            if frame_length < width:
                category = "row_shorter_than_crop"
            elif padding == 0:
                category = "fully_valid"
            elif valid == 0:
                category = "fully_padding"
            else:
                category = "partial_padding"
            pad_start = max(start, frame_length)
            pad_end = min(end, total)
            pad_count = max(0, pad_end - pad_start)
            gap_pad = int(gap[row_index, pad_start:pad_end].sum()) if pad_count else 0
            real_pad = moved["mel"][row_index, pad_start:pad_end]
            fake_pad = fake[row_index, max(0, pad_start - start):max(0, pad_end - start)]
            rows.append({
                "audit": label, "crop_mode": "fixed" if use_lengths else "legacy",
                "row": row_index, "category": category,
                "frame_length": frame_length,
                "crop_start": int(start), "crop_end": int(end), "crop_width": int(width),
                "padding_frames": int(padding),
                "padding_ratio": round(padding / max(1, width), 4),
                "gap_padding_frames": gap_pad,
                "gap_padding_ratio": round(gap_pad / max(1, pad_count), 4),
                "real_pad_nonzero": int((real_pad != 0).sum()) if pad_count else 0,
                "fake_pad_nonzero": int((fake_pad != 0).sum()) if pad_count else 0,
                "fake_pad_max_abs": round(float(fake_pad.abs().max()), 6) if pad_count else 0.0,
            })
        return rows

    rows = []
    for batch_index, batch in enumerate(batches):
        for row in audit_batch(batch, f"random_{batch_index}", use_lengths=True):
            row["batch"] = batch_index
            rows.append(row)

    # Batch đích: 1 row ngắn (<64 frame) + các row dài **cùng N** — tái hiện lỗi crop cũ.
    from collections import defaultdict

    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        groups[len(row["durations"])].append(index)
    targeted = None
    targeted_meta = None
    for count, indices in sorted(groups.items()):
        short = [i for i in indices
                 if dataset.rows[i]["frame_range"][1] - dataset.rows[i]["frame_range"][0] < 64]
        long_rows = [i for i in indices
                     if dataset.rows[i]["frame_range"][1] - dataset.rows[i]["frame_range"][0] >= 128]
        if short and long_rows:
            chosen = [short[0]] + long_rows[:7]
            targeted = collate_slot_batch([dataset[i] for i in chosen])
            targeted_meta = {
                "N": count, "indices": chosen,
                "frame_lengths": [dataset.rows[i]["frame_range"][1] - dataset.rows[i]["frame_range"][0]
                                  for i in chosen],
            }
            break
    if targeted is not None:
        rows.extend(audit_batch(targeted, "targeted_legacy", use_lengths=False))
        rows.extend(audit_batch(targeted, "targeted_fixed", use_lengths=True))

    summary: dict[str, dict] = {}
    for audit in sorted({row["audit"] for row in rows}):
        for mode in sorted({row["crop_mode"] for row in rows if row["audit"] == audit}):
            subset = [row for row in rows if row["audit"] == audit and row["crop_mode"] == mode]
            summary[f"{audit}|{mode}"] = {
                "rows": len(subset),
                "partial_padding": sum(1 for row in subset if row["category"] == "partial_padding"),
                "fully_padding": sum(1 for row in subset if row["category"] == "fully_padding"),
                "row_shorter_than_crop": sum(1 for row in subset if row["category"] == "row_shorter_than_crop"),
                "padded_rows": sum(1 for row in subset if row["padding_frames"] > 0),
                "padding_frames": int(sum(row["padding_frames"] for row in subset)),
                "gap_padding_frames": int(sum(row["gap_padding_frames"] for row in subset)),
            }
    result = {
        "source": "sample_crops thực tế của tools/finetune_adv.py",
        "batches": len(batches), "row_count": len(rows),
        "targeted_batch": targeted_meta,
        "note": "legacy = không truyền lengths (hành vi cũ); fixed = truyền frame_lengths",
        "summary": summary,
        "rows": rows,
    }
    out = Path("artifacts/bench/v1837_crop_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"batches": len(batches), "rows": len(rows), "summary": summary},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

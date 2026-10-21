"""tools/fix_edge_labels.py — sửa nhãn đầu/cuối segment có chọn lọc (A0 v17, Việc A1/12b).

Vì sao: âm tiết CUỐI segment dài +50 % so với giữa (21 vs 14 frame) — một phần do nhãn nhét
lặng/hơi thở vào slot cuối. Tool chuyển các frame cuối (năng lượng thấp) sang gap/tail và các
frame đầu (slot O) sang lead, sàn 3 frame/slot. Không sửa audio, không đổi tổng frame.

Chạy:
  .venv/bin/python tools/fix_edge_labels.py --config configs/small_codelung_v32.yaml \
      --manifest data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl \
      --out data/codelung/v35/train_clean_v2_dur_seg_slot4_edgelabel.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

DB = 10.0 / np.log(10.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default="cache/codelung_v2")
    parser.add_argument("--threshold-db", type=float, default=25.0)
    parser.add_argument("--min-slot", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from vifasttts.data.dataset_v2 import SlotDataset

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json"))
                          .read_text(encoding="utf-8"))
    stats = json.loads(Path(config["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=1)
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]

    changed_last = changed_first = 0
    before_last, after_last = [], []
    before_first, after_first = [], []
    for index, row in enumerate(rows):
        durations = [int(v) for v in row["durations"]]
        gaps = [int(v) for v in row.get("gaps", [0] * len(durations))]
        while len(gaps) < len(durations):
            gaps.append(0)
        lead = int(row.get("lead_frames", 0))
        seg = [list(map(int, v)) for v in row["seg_durations"]]
        item = dataset[index]
        mel = np.asarray(item["mel"], dtype=np.float64) * std + mean
        level = mel.mean(axis=-1) * DB
        voiced = np.asarray(item["vuv"]).reshape(-1) >= 0.5
        ref = float(np.median(level[voiced])) if voiced.any() else float(np.median(level))
        low = level < (ref - args.threshold_db)

        # cuối segment: đếm run thấp ở cuối
        last = len(durations) - 1
        end = lead + sum(durations) + sum(gaps[: last])
        start = end - durations[last]
        run = 0
        frame = end - 1
        while frame >= start and low[frame] and run < durations[last] - args.min_slot:
            run += 1
            frame -= 1
        before_last.append(durations[last])
        if run > 0:
            slot = 3 if seg[last][3] >= seg[last][2] else 2
            take = min(run, max(0, seg[last][slot] - args.min_slot))
            if take > 0:
                seg[last][slot] -= take
                durations[last] -= take
                gaps[last] += take
                changed_last += 1
        after_last.append(durations[last])

        # đầu segment: run thấp ở đầu slot O
        run = 0
        frame = lead
        while frame < lead + durations[0] and low[frame] and run < durations[0] - args.min_slot:
            run += 1
            frame += 1
        before_first.append(durations[0])
        if run > 0:
            take = min(run, max(0, seg[0][0] - args.min_slot))
            if take > 0:
                seg[0][0] -= take
                durations[0] -= take
                lead += take
                changed_first += 1
        after_first.append(durations[0])

        row["durations"] = durations
        row["gaps"] = gaps
        row["lead_frames"] = lead
        row["seg_durations"] = seg

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"sửa cuối {changed_last}/{len(rows)} = {100*changed_last/len(rows):.1f}% · "
          f"đầu {changed_first}/{len(rows)} = {100*changed_first/len(rows):.1f}%")
    print(f"độ dài âm cuối: trước {np.median(before_last):.0f} → sau {np.median(after_last):.0f} frame · "
          f"âm đầu: {np.median(before_first):.0f} → {np.median(after_first):.0f}")
    print(f"ghi {out}")


if __name__ == "__main__":
    main()

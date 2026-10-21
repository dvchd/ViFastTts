"""tools/top_spike_inspect.py — kiểm 10 âm tiết "spike" nhất của thước năng lượng (A0 v10, Việc 5/6e).

Vì sao: các top spike (+15…+22 dB) giống hệt nhau ở mọi checkpoint ⇒ nghi frame "thật" của âm tiết
đó là lặng hoặc biên align lệch, chứ không phải model tự làm to. Tool chỉ ĐỌC dữ liệu, không sửa:
  - Vẽ mel THẬT quanh âm tiết + khoảng âm tiết (theo `durations`) + biên phone (TextGrid MFA).
  - In mức năng lượng thật của âm tiết so trung vị cả segment, và các phone phủ lên khoảng đó.

Chạy:
  .venv/bin/python tools/top_spike_inspect.py \
      --energy artifacts/bench/energy_v32_ref.json --manifest data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl \
      --config configs/small_codelung_v32.yaml --cache-dir cache/codelung_v2 \
      --textgrid-dir data/codelung/mfa_v2c_vn_aligned --out artifacts/top_spikes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_mfa_coda import _parse

DB = 10.0 / np.log(10.0)
HOP = 256


def _phones_in(tiers: dict[str, list[tuple[float, float, str]]], start: float, end: float) -> str:
    phones = tiers.get("phones", [])
    hits = [text for a, b, text in phones if b > start and a < end and text.strip()]
    return " ".join(hits) if hits else "(không có phone)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy", required=True, help="JSON của tools/adv_energy_audit.py")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", default="cache/codelung_v2")
    parser.add_argument("--textgrid-dir", default="data/codelung/mfa_v2c_vn_aligned")
    parser.add_argument("--alpha", default="0.0")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vifasttts.data.dataset_v2 import SlotDataset

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json"))
                          .read_text(encoding="utf-8"))
    stats = json.loads(Path(config["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)

    energy = json.loads(Path(args.energy).read_text(encoding="utf-8"))
    spikes = energy["summary"][args.alpha]["top_spikes"][: args.top]
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=1)
    index_by_id = {str(row["id"]): i for i, row in enumerate(dataset.rows)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for rank, spike in enumerate(spikes, 1):
        segment = str(spike["segment"])
        index = index_by_id.get(segment)
        if index is None:
            rows_out.append({**spike, "note": "không thấy segment trong manifest"})
            continue
        item = dataset[index]
        row = dataset.rows[index]
        mel = (np.asarray(item["mel"], dtype=np.float64) * std + mean)
        durations = [int(d) for d in np.asarray(item["durations"]).reshape(-1)]
        gaps = [int(g) for g in np.asarray(item["gaps"]).reshape(-1)]
        lead = int(item["lead"]) if np.ndim(item["lead"]) == 0 else int(np.asarray(item["lead"]).reshape(-1)[0])
        syl = int(spike["syllable"])
        start_frame = lead + sum(durations[:syl]) + sum(gaps[:syl])
        end_frame = start_frame + durations[syl]
        level = np.log(np.sum(np.exp(2 * mel), axis=-1) + 1e-12) * DB
        inside = float(np.mean(level[start_frame:end_frame])) if end_frame > start_frame else float("nan")
        med = float(np.median(level))
        # thời gian tuyệt đối trong clip cha (TextGrid dùng giây của cả clip)
        offset = float(row["frame_range"][0]) * HOP / 24000.0
        t0 = offset + start_frame * HOP / 24000.0
        t1 = offset + end_frame * HOP / 24000.0
        grid_path = Path(args.textgrid_dir) / f"{row['parent']}.TextGrid"
        phones = _phones_in(_parse(grid_path), t0, t1) if grid_path.exists() else "(thiếu TextGrid)"

        fig, ax = plt.subplots(figsize=(7, 3))
        lo = max(0, start_frame - 40)
        hi = min(mel.shape[0], end_frame + 40)
        ax.imshow(mel[lo:hi].T, origin="lower", aspect="auto", cmap="magma",
                  extent=[lo / (24000 / HOP), hi / (24000 / HOP), 0, 100])
        ax.axvline(start_frame / (24000 / HOP), color="#4fd1ff", lw=1.4)
        ax.axvline(end_frame / (24000 / HOP), color="#4fd1ff", lw=1.4)
        ax.set_title(f"#{rank} “{spike['word']}” · Δ {spike['delta_db']:+.1f} dB · thật {inside - med:+.1f} dB "
                     f"so trung vị · {phones}", fontsize=8)
        ax.set_xlabel("giây trong segment (frame thật)", fontsize=7)
        ax.set_ylabel("mel bin", fontsize=7)
        fig.tight_layout()
        png = out / f"{rank:02d}_{spike['word'].replace('/', '_') or 'syl'}.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        rows_out.append({**spike, "real_rel_db": round(inside - med, 2), "phones": phones,
                         "png": str(png), "t0": round(t0, 3), "t1": round(t1, 3)})

    (out / "summary.json").write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'#':>2} {'từ':12s} {'Δ dB':>7s} {'thật-median':>11s}  phone")
    for rank, r in enumerate(rows_out, 1):
        print(f"{rank:2d} {r.get('word', ''):12s} {r.get('delta_db', 0):7.1f} "
              f"{r.get('real_rel_db', float('nan')):11.1f}  {r.get('phones', '')}")


if __name__ == "__main__":
    main()

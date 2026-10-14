"""tools/copy_synth_clips.py — copy-synthesis theo DANH SÁCH clip (mel THẬT → vocoder).

Dùng cho cổng rè (§38–§45): cần wav copy-synth của đúng 25 clip giàu ngã/nặng để so
`tools/creak_rate.py` giữa các vocoder (pretrained vs fine-tune) — cùng mel thật, chỉ
đổi vocoder, nên mọi khác biệt creak quy về vocoder.

    .venv/bin/python tools/copy_synth_clips.py --clips artifacts/creak25/clips25.json \
        --manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
        --config configs/small_codelung_v2.yaml --out artifacts/creak25/copy_synth_stftc \
        --checkpoint runs/vocoder_vocos_stftc/last.pt

- `--checkpoint` bỏ trống ⇒ dùng Vocos pretrained trong config (`vocoder_training.pretrained_path`).
- `--device` mặc định `cpu` (đo song song với train mà không tranh MPS).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def main() -> None:
    """CLI: nạp dataset + vocoder, sinh wav cho từng clip trong danh sách."""
    import torch
    import yaml

    from vifasttts.data.dataset_v2 import SlotDataset
    from vifasttts.infer import write_wav
    from vifasttts.models.istftnet import build_generator, load_generator_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, help="JSON [{parent, row}, …]")
    parser.add_argument("--manifest", required=True, help="Manifest chứa các row (để lấy mel)")
    parser.add_argument("--config", required=True, help="Config (yaml hoặc resolved_config.json)")
    parser.add_argument("--out", required=True, help="Thư mục ra, tên <parent>.wav")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint vocoder (bỏ trống = pretrained)")
    parser.add_argument("--cache-dir", default="cache/codelung_v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    path = Path(args.config)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix in (".yaml", ".yml") \
        else json.loads(path.read_text(encoding="utf-8"))
    f0_stats = json.loads(Path("data/codelung/f0_stats_v2.json").read_text(encoding="utf-8"))
    dataset = SlotDataset(args.manifest, cfg, f0_stats, cache_dir=args.cache_dir)
    id_to_index = {}
    for index, line in enumerate(Path(args.manifest).read_text(encoding="utf-8").splitlines()):
        if line:
            id_to_index[json.loads(line)["id"]] = index

    device = torch.device(args.device)
    generator = build_generator(cfg.get("vocoder_training", {}), cfg["mel"]).to(device).eval()
    if args.checkpoint:
        load_generator_state(generator, args.checkpoint)
        print(f"vocoder: {args.checkpoint}")
    else:
        print("vocoder: pretrained (theo config)")

    clips = json.loads(Path(args.clips).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    with torch.no_grad():
        for entry in clips:
            row_id = entry["row"]["id"]
            if row_id not in id_to_index:
                missing += 1
                continue
            item = dataset[id_to_index[row_id]]
            mel = item["mel"].unsqueeze(0).to(device)
            waveform = generator(mel)[0]
            write_wav(out_dir / f"{entry['parent']}.wav", waveform.detach().cpu(),
                      int(cfg.get("sample_rate", 24000)))
    print(f"xong {len(clips) - missing}/{len(clips)} clip → {out_dir}")


if __name__ == "__main__":
    main()

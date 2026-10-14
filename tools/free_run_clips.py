"""tools/free_run_clips.py — free-run (text → acoustic → vocoder) theo DANH SÁCH clip.

Dùng cho cổng rè (§38–§45): đo creak trên bản model TỰ SINH (free-run) — khác copy-synth
ở chỗ thời lượng do model tự đặt ⇒ kèm `--windows-out` (cửa sổ nucleus theo
`slot_durations` của model) để `tools/creak_rate.py --windows` quy đúng âm tiết.

    .venv/bin/python tools/free_run_clips.py --clips artifacts/creak25/clips25.json \
        --config runs/acoustic_v2_E4a/resolved_config.json \
        --checkpoint runs/acoustic_v2_E4a/last.pt --out artifacts/creak25/free_E4a \
        --windows-out artifacts/creak25/free_E4a_windows.json --device mps

- `--checkpoint` bỏ trống ⇒ chỉ dùng để kiểm (sẽ lỗi vì thiếu trọng số model).
- Vocoder: theo config (`vocoder_training`) — pretrained nếu không truyền `--vocoder-checkpoint`.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HOP_SECONDS = 256 / 24000


def main() -> None:
    """CLI: sinh wav free-run + cửa sổ âm tiết (giây) cho từng clip."""
    import torch
    import yaml

    from vifasttts.frontend.pipeline import parse_text
    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.infer import write_wav
    from vifasttts.models.factory import build_model, load_model_state
    from vifasttts.models.istftnet import build_generator, load_generator_state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, help="JSON [{parent, row}, …] (row cần `text`)")
    parser.add_argument("--config", required=True, help="resolved_config.json của run acoustic")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint acoustic (.pt)")
    parser.add_argument("--out", required=True, help="Thư mục wav ra (<parent>.wav)")
    parser.add_argument("--windows-out", required=True, help="JSON cửa sổ âm tiết cho creak_rate")
    parser.add_argument("--vocoder-checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "mel" not in config:  # yaml cũng nhận
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path("data/codelung/f0_stats_v2.json").read_text(encoding="utf-8"))
    vocab = ComponentVocabs.default()
    sizes = [len(vocab.onset), len(vocab.medial), len(vocab.nucleus), len(vocab.coda),
             len(vocab.tone), len(vocab.boundary), len(vocab.punctuation)]
    device = torch.device(args.device)

    model = build_model(config, sizes, arch="v2_slot5")
    load_model_state(model, torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"])
    model.load_f0_stats(f0_stats)
    model.to(device).eval()
    generator = build_generator(config.get("vocoder_training", {}), config["mel"]).to(device).eval()
    if args.vocoder_checkpoint:
        load_generator_state(generator, args.vocoder_checkpoint)
        print(f"vocoder: {args.vocoder_checkpoint}")
    else:
        print("vocoder: pretrained (theo config)")

    clips = json.loads(Path(args.clips).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows: dict[str, list[dict]] = {}
    done = 0
    with torch.no_grad():
        for entry in clips:
            items = parse_text(entry["row"]["text"])
            if not items:
                continue
            features = torch.tensor([[vocab.encode(item) for item in items]], dtype=torch.long, device=device)
            output = model(features, speed=torch.tensor([args.speed], device=device))
            mel = output["mel"][0]
            waveform = generator(mel.unsqueeze(0))[0]
            write_wav(out_dir / f"{entry['parent']}.wav", waveform.detach().cpu(),
                      int(config.get("sample_rate", 24000)))
            slots = output["slot_durations"][0].cpu().tolist()
            cursor = slots[0]
            per_clip = []
            for index, item in enumerate(items):
                onset, medial, nucleus, coda, gap = slots[1 + 5 * index:6 + 5 * index]
                start = cursor + onset + medial
                per_clip.append({"tone": item.abstract.tone.name, "start": start * HOP_SECONDS,
                                 "end": (start + nucleus) * HOP_SECONDS})
                cursor += onset + medial + nucleus + coda + gap
            windows[entry["parent"]] = per_clip
            done += 1
    Path(args.windows_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.windows_out).write_text(json.dumps(windows, ensure_ascii=False), encoding="utf-8")
    print(f"xong {done}/{len(clips)} clip → {out_dir} · cửa sổ → {args.windows_out}")


if __name__ == "__main__":
    main()

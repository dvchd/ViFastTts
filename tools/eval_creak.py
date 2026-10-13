#!/usr/bin/env python
"""tools/eval_creak.py — đo độ "rè" (ồn/rung) của bản tổng hợp so với audio THẬT, theo thanh.

Lý do (§38): tai user nghe "rè" ở thanh NGÃ; đo ra: HNR synth thấp hơn thật 2,4 dB và jitter
+66% — tức model *tự sinh* độ rung/ồn cho ngã/nặng (không phải Vocos, không phải `sh25`; `sh25`
chỉ khuếch đại). Cần thước này để: (a) so checkpoint (18k/40k/…), (b) kiểm GTA Vocos có kéo về
mức thật không, (c) cảnh báo overfit-nhiễu khi train dài (gap phình = học cả tiếng ồn nguồn).

Cách đo: pool 3 kênh phonation (hnr · glottalization · jitter) trên cửa sổ **nucleus** — một bên
là cache v3 của audio thật, một bên là `extract_features` trên wav do model sinh (free-run, cửa
sổ theo `slot_durations` của model). Chỉ tính âm tiết có ≥ `--min-frames` frame và (với tổng hợp)
đủ frame hữu thanh; đếm + báo số bỏ.

Chạy:
  .venv/bin/python tools/eval_creak.py --config runs/acoustic_v2_base/resolved_config.json \\
      --checkpoint runs/acoustic_v2_base/last.pt --clips 40 --out artifacts/eval_acoustic/creak_base40k.json

Kết quả: JSON `{config, checkpoint, clips_used, groups: {TONE: {real: {...}, synth: {...}, n}}}`.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch

from vifasttts.data.dataset_v2 import SlotDataset, collate_slot_batch
from vifasttts.data.features import extract_features
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.infer import Synthesizer
from vifasttts.models.factory import build_model, load_model_state

CHANNELS = ("hnr", "glottalization", "jitter")
#: Thanh cần soi (ngã/nặng là nhóm creak) + đối chứng không creak.
DEFAULT_TONES = ("NGANG", "NGA", "NANG")


def _pool(features: dict, start: int, span: int) -> np.ndarray:
    """Trung bình 3 kênh phonation trên cửa sổ [start, start+span)."""
    window = slice(start, start + span)
    return np.array([float(features[name][window].mean()) for name in CHANNELS])


def _windows_manifest(item: dict) -> list[tuple[int, int]]:
    """Cửa sổ nucleus theo `seg_durations` + `gaps` + `lead` của row (toạ độ segment)."""
    seg = item["seg_durations"].numpy()
    lead = int(item["lead"])
    gaps = item["gaps"].numpy()
    windows, cursor = [], lead
    for k in range(len(seg)):
        onset, medial, nucleus, coda = (int(x) for x in seg[k])
        windows.append((cursor + onset + medial, nucleus))
        cursor += onset + medial + nucleus + coda
        if k < len(gaps):
            cursor += int(gaps[k])
    return windows


def _windows_model(slot_durations: list[int]) -> list[tuple[int, int]]:
    """Cửa sổ nucleus theo `slot_durations` phẳng `[lead, (O,M,N,C,G)×N]` của model."""
    windows, cursor = [], int(slot_durations[0])
    for k in range((len(slot_durations) - 1) // 5):
        onset, medial, nucleus, coda, gap = slot_durations[1 + 5 * k: 6 + 5 * k]
        windows.append((cursor + onset + medial, nucleus))
        cursor += onset + medial + nucleus + coda + gap
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Đo độ rè (HNR/jitter) synth vs thật theo thanh.")
    parser.add_argument("--config", required=True, help="resolved_config.json của run (JSON)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/codelung/val_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--cache-dir", default="cache/codelung_v2")
    parser.add_argument("--f0-stats", default="data/codelung/f0_stats_v2.json")
    parser.add_argument("--clips", type=int, default=40, help="số clip quét (chỉ dùng clip có thanh đích)")
    parser.add_argument("--tones", default=",".join(DEFAULT_TONES))
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path(args.f0_stats).read_text(encoding="utf-8"))
    vocab = ComponentVocabs.default()
    tone_names = list(vocab.tone)
    wanted = [t.strip().upper() for t in args.tones.split(",") if t.strip()]
    sizes = [len(vocab.onset), len(vocab.medial), len(vocab.nucleus), len(vocab.coda),
             len(vocab.tone), len(vocab.boundary), len(vocab.punctuation)]
    model = build_model(config, sizes, arch="v2_slot5")
    load_model_state(model, torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"])
    model.load_f0_stats(f0_stats)
    model.eval()
    synth = Synthesizer(config, args.checkpoint, device=args.device)
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir)
    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]

    groups: dict[str, dict[str, list[np.ndarray]]] = {t: {"real": [], "synth": []} for t in wanted}
    clips_used = 0
    tmp = Path(tempfile.mkdtemp(prefix="eval_creak_"))
    with torch.no_grad():
        for index in range(min(args.clips, len(dataset))):
            item = dataset[index]
            tones = item["features"][:, 4].numpy()
            if not any(int(t) < len(tone_names) and tone_names[int(t)] in wanted for t in tones):
                continue
            clips_used += 1
            real = {name: value.numpy() for name, value in
                    extract_features(rows[index]["audio_path"], config).items()}
            batch = collate_slot_batch([item])
            outputs = model(batch["features"], durations=batch["durations"],
                            seg_durations=batch["seg_durations"], gaps=batch["gaps"], lead=batch["lead"])
            waveform = synth.vocoder(outputs["mel"])[0].cpu().numpy()
            path = tmp / f"{index}.wav"
            sf.write(path, waveform, int(config.get("sample_rate", 24000)))
            fake = {name: value.numpy() for name, value in extract_features(path, config).items()}
            wins_real = _windows_manifest(item)
            wins_synth = _windows_model([int(x) for x in outputs["slot_durations"][0].tolist()])
            for k in range(min(len(wins_real), len(wins_synth), len(tones))):
                tone = tone_names[int(tones[k])]
                if tone not in groups:
                    continue
                span_real, span_synth = wins_real[k][1], wins_synth[k][1]
                if span_real < args.min_frames or span_synth < args.min_frames:
                    continue
                groups[tone]["real"].append(_pool(real, *wins_real[k]))
                groups[tone]["synth"].append(_pool(fake, *wins_synth[k]))

    report: dict = {"config": args.config, "checkpoint": args.checkpoint, "channels": list(CHANNELS),
                    "clips_used": clips_used, "clips_scanned": min(args.clips, len(dataset)), "groups": {}}
    print(f"clip dùng: {clips_used}/{min(args.clips, len(dataset))} · kênh: {' · '.join(CHANNELS)}")
    print(f"{'thanh':<7}{'nguồn':<9}{'hnr':>9}{'glottal':>10}{'jitter':>9}{'n':>6}")
    for tone, bucket in groups.items():
        if not bucket["real"]:
            continue
        entry: dict = {}
        for source in ("real", "synth"):
            values = np.stack(bucket[source])
            entry[source] = {name: float(values[:, i].mean()) for i, name in enumerate(CHANNELS)}
            entry[source]["n"] = int(values.shape[0])
            print(f"{tone if source == 'real' else '':<7}{source:<9}"
                  f"{entry[source]['hnr']:>9.2f}{entry[source]['glottalization']:>10.4f}"
                  f"{entry[source]['jitter']:>9.4f}{entry[source]['n']:>6}")
        entry["delta_synth_minus_real"] = {
            name: entry["synth"][name] - entry["real"][name] for name in CHANNELS}
        report["groups"][tone] = entry
        delta = entry["delta_synth_minus_real"]
        print(f"       Δ(synth−thật): hnr {delta['hnr']:+.2f} · jitter {delta['jitter']:+.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"đã ghi {args.out}")


if __name__ == "__main__":
    main()

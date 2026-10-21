"""tools/adv_checkpoints.py — bảng thước cho checkpoint adversarial (A0 lượt 5, §2.2–2.3).

Đo trên N segment val cố định (TF forward, batch 1), **mel z sau postnet, TRƯỚC làm nét α**:
1. `val_mel`      — L1 mel z toàn row (như mọi lần).
2. `val_voiced`   — L1 chỉ trên frame hữu thanh (vuv ≥ 0,5, bỏ frame gap).
3. `gv`, `gv26`   — GV hữu thanh (median var_pred/var_real) + dải 2–6 kHz.
4. `flux_ratio`   — |Δmel| trung bình quanh **biên âm tiết** (cửa sổ cursor−1…cursor+2),
                    tỉ lệ synth/real (thước cũ: thật = 1,0; v3.2 ≈ 0,63 — phải tăng về 1).
5. `smooth_ratio` — |Δmel| trung bình giữa 2 frame liền kề **trong vùng hữu thanh**, synth/real.
                    > 1,2 ⇒ model đang rung/nhiễu (A0: loại).

Chạy:
  .venv/bin/python tools/adv_checkpoints.py --config configs/small_codelung_v32.yaml \
      --checkpoint artifacts/adv_grid/P1/ckpt_2k_ema.pt \
      --manifest data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl \
      --cache-dir cache/codelung_v2 --segments 40 --out artifacts/bench/ckpt_p1_2k.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def _gap_mask(slot_durations: torch.Tensor) -> torch.Tensor:
    """[B,1+5N] → [B,T] True ở frame G_lead/slot G (cùng mẹo cumsum của trainer)."""
    batch, slots = slot_durations.shape
    gap_slot = torch.cat(
        [torch.ones(batch, 1, dtype=torch.bool, device=slot_durations.device),
         (torch.arange(slots - 1, device=slot_durations.device) % 5 == 4).view(1, -1).expand(batch, -1)],
        dim=1)
    steps = torch.arange(slot_durations.sum(dim=1).max().item(), device=slot_durations.device)
    index = (slot_durations.cumsum(dim=1).unsqueeze(2) <= steps.view(1, 1, -1)).sum(dim=1)
    return torch.gather(gap_slot, 1, index.clamp(max=slots - 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--segments", type=int, default=40)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from vifasttts.data.dataset_v2 import SlotDataset
    from vifasttts.infer import Synthesizer

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json")).read_text(encoding="utf-8"))
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    device = engine.device
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=1)

    l1_all, l1_voiced = [], []
    num_v = np.zeros(100)
    den_v = np.zeros(100)
    flux_num = flux_den = 0.0
    smooth_num = smooth_den = 0.0
    for index in range(min(args.segments, len(dataset))):
        item = dataset[index]
        features = item["features"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = engine.model(features,
                               durations=item["durations"].unsqueeze(0).to(device),
                               seg_durations=item["seg_durations"].unsqueeze(0).to(device),
                               gaps=item["gaps"].unsqueeze(0).to(device),
                               lead=item["lead"].reshape(1).to(device))
        pred = out["mel"][0].float().cpu().numpy()
        real = item["mel"].numpy()
        frames = min(len(pred), len(real))
        pred, real = pred[:frames], real[:frames]
        valid = np.ones(frames, dtype=bool)  # item["mel"] đã cắt đúng độ dài segment
        gap = _gap_mask(out["slot_durations"])[0, :frames].cpu().numpy()
        vuv = item["vuv"][:frames].numpy()
        voice = (~gap) & (vuv >= 0.5) & valid
        diff = np.abs(pred - real)
        l1_all.append(float(diff[valid].mean()) if valid.any() else float("nan"))
        if voice.sum() >= 3:
            l1_voiced.append(float(diff[voice].mean()))
            num_v += pred[voice].var(axis=0) * int(voice.sum())
            den_v += real[voice].var(axis=0) * int(voice.sum())
        # flux ở biên âm tiết (cùng công thức roughness_dashboard)
        durations = [int(d) for d in item["durations"].tolist()]
        cursor = int(item["lead"])
        for i, dur in enumerate(durations):
            if i > 0:
                lo, hi = max(0, cursor - 1), min(frames, cursor + 2)
                if hi - lo >= 2:
                    flux_num += float(np.abs(np.diff(pred[lo:hi], axis=0)).mean())
                    flux_den += float(np.abs(np.diff(real[lo:hi], axis=0)).mean())
            cursor += dur + int(item["gaps"][i])
        # độ mượt thời gian trong vùng hữu thanh
        pair = voice[1:] & voice[:-1]
        if pair.any():
            smooth_num += float(np.abs(np.diff(pred, axis=0))[pair].mean()) * int(pair.sum())
            smooth_den += float(np.abs(np.diff(real, axis=0))[pair].mean()) * int(pair.sum())

    ratio = num_v / np.maximum(den_v, 1e-9)
    from vifasttts.data.features import mel_filterbank

    mel_cfg = config["mel"]
    bank = mel_filterbank(int(config.get("sample_rate", 24000)), int(mel_cfg["n_fft"]), int(mel_cfg["n_mels"]),
                          float(mel_cfg.get("f_min", 0)), float(mel_cfg.get("f_max", 12000)),
                          norm=mel_cfg.get("norm", "slaney")).numpy()
    freqs = np.linspace(0, int(config.get("sample_rate", 24000)) / 2, bank.shape[1])
    centers = np.array([freqs[bank[i] > 0].mean() if (bank[i] > 0).any() else 0 for i in range(bank.shape[0])])
    band = (centers >= 2000) & (centers <= 6000)
    result = {"checkpoint": args.checkpoint, "tag": args.tag, "segments": min(args.segments, len(dataset)),
              "val_mel": float(np.nanmean(l1_all)),
              "val_voiced": float(np.nanmean(l1_voiced)) if l1_voiced else float("nan"),
              "gv": float(np.median(ratio)),
              "gv26": float(np.median(ratio[band])),
              "flux_ratio": flux_num / max(flux_den, 1e-9),
              "smooth_ratio": smooth_num / max(smooth_den, 1e-9)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.tag or Path(args.checkpoint).name:24s} val {result['val_mel']:.4f} · voiced {result['val_voiced']:.4f} "
          f"· GV {result['gv']:.3f} · GV26 {result['gv26']:.3f} · flux {result['flux_ratio']:.3f} "
          f"· smooth {result['smooth_ratio']:.3f}")


if __name__ == "__main__":
    main()

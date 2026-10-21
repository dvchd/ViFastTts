"""tools/finetune_adv.py — Adversarial fine-tune decoder+postnet từ checkpoint (spec A0 21/10).

Spec (Việc 2):
- **Đóng băng**: embedding, encoder, duration, slot_expander/ratio/variance, pause, các projection.
  **Train**: `decoder`, `mel` (đầu ra), `postnet`.
- G loss: mel L1 1.0 + mel_pre 0.2 + mel_sharp 1.0 + **adv 0.1** (ramp 0→0.1 trong `ramp_steps`)
  + feature matching **2.0**. Các loss khác không tính.
- D: `MultiScaleMelDiscriminator` + spectral norm, **hinge**, input = crop ngẫu nhiên **64 frame**
  với ≥80% frame **vùng tiếng** (bỏ lặng). Mel z sau postnet.
- G AdamW lr **2e-5**, D lr 1e-4, betas (0.5, 0.9), grad clip 1.0. D warmup 1000 (G đứng yên).
- **EMA 0.999** cho G; eval/checkpoint dùng trọng số EMA.
- Dừng khẩn: D loss < 0.05 liên tục 500 bước · val mel L1 tăng > 10% · NaN.
- History mỗi `--log-every` bước: d_loss, adv, mel L1, **GV hữu thanh** (tổng + dải 2–6 kHz).

Chạy probe 1k:
  .venv/bin/python tools/finetune_adv.py --config configs/small_codelung_v32.yaml \
      --checkpoint runs/acoustic_v32/best.pt \
      --train-manifest data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl \
      --val-manifest data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl \
      --cache-dir cache/codelung_v2 --output artifacts/adv_probe --max-steps 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from vifasttts.data.dataset_v2 import SameNBatchSampler, SlotDataset, collate_slot_batch
from vifasttts.data.features import mel_filterbank
from vifasttts.models.discriminators import (
    MultiScaleMelDiscriminator,
    feature_matching_loss,
)
from vifasttts.models.factory import arch_of_checkpoint, build_model, load_model_state
from vifasttts.models.v2_losses import DEFAULT_WEIGHTS, slot_losses

TRAIN_PREFIXES = ("decoder.", "mel.", "postnet.")


def gap_mask(slot_durations: torch.Tensor) -> torch.Tensor:
    """[B,1+5N] → [B,T] True ở frame G_lead/slot G (cùng mẹo cumsum của trainer)."""
    batch, slots = slot_durations.shape
    gap_slot = torch.cat(
        [torch.ones(batch, 1, dtype=torch.bool, device=slot_durations.device),
         (torch.arange(slots - 1, device=slot_durations.device) % 5 == 4).view(1, -1).expand(batch, -1)],
        dim=1)
    steps = torch.arange(slot_durations.sum(dim=1).max().item(), device=slot_durations.device)
    index = (slot_durations.cumsum(dim=1).unsqueeze(2) <= steps.view(1, 1, -1)).sum(dim=1)
    return torch.gather(gap_slot, 1, index.clamp(max=slots - 1))


def sample_crops(real: torch.Tensor, fake: torch.Tensor, gap: torch.Tensor, rng: random.Random,
                 frames: int = 64, min_speech: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop cùng vị trí cho real/fake; ưu tiên cửa sổ ≥80% frame vùng tiếng."""
    batch, total, _bins = real.shape
    width = min(frames, total)
    out_r, out_f = [], []
    for b in range(batch):
        best, best_frac = 0, -1.0
        for _ in range(8):
            start = rng.randint(0, max(0, total - width))
            frac = 1.0 - float(gap[b, start:start + width].float().mean())
            if frac >= min_speech:
                best, best_frac = start, frac
                break
            if frac > best_frac:
                best, best_frac = start, frac
        end = best + width
        out_r.append(real[b, best:end])
        out_f.append(fake[b, best:end])
    return torch.stack(out_r), torch.stack(out_f)


def hinge_d(real_outs, fake_outs) -> torch.Tensor:
    total = torch.zeros((), device=real_outs[0][0].device)
    for (r, _), (f, _) in zip(real_outs, fake_outs):
        total = total + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
    return total / max(len(real_outs), 1)


def hinge_g(fake_outs) -> torch.Tensor:
    total = torch.zeros((), device=fake_outs[0][0].device)
    for f, _ in fake_outs:
        total = total - f.mean()
    return total / max(len(fake_outs), 1)


@torch.no_grad()
def evaluate(model, batches, device, centers, band_2_6) -> dict:
    """val mel L1 (toàn row) + GV hữu thanh (median + dải 2-6 kHz) trên các batch cho trước."""
    model.eval()
    l1_sum, l1_n = 0.0, 0
    num_v = np.zeros(100); den_v = np.zeros(100)
    for batch in batches:
        moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(moved["features"], durations=moved["durations"],
                    seg_durations=moved["seg_durations"], gaps=moved["gaps"], lead=moved["lead"])
        pred, real = out["mel"], moved["mel"]
        frames = min(pred.size(1), real.size(1))
        pred, real = pred[:, :frames], real[:, :frames]
        steps = torch.arange(frames, device=device).view(1, -1)
        valid = steps < moved["frame_lengths"].unsqueeze(1)
        l1_sum += float(((pred - real).abs().mean(-1) * valid).sum())
        l1_n += int(valid.sum())
        gap = gap_mask(out["slot_durations"])[:, :frames]
        vuv = moved["vuv"][:, :frames]
        voice = (~gap) & (vuv >= 0.5) & valid
        if int(voice.sum()) < 3:
            continue
        num_v += pred[voice].var(dim=0).detach().cpu().numpy() * int(voice.sum())
        den_v += real[voice].var(dim=0).detach().cpu().numpy() * int(voice.sum())
    model.train()
    ratio = num_v / np.maximum(den_v, 1e-9)
    return {"val_mel": l1_sum / max(l1_n, 1),
            "gv_voiced": float(np.median(ratio)),
            "gv_2_6k": float(np.median(ratio[band_2_6]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--g-lr", type=float, default=2e-5)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--adv-weight", type=float, default=0.1)
    parser.add_argument("--fm-weight", type=float, default=2.0)
    parser.add_argument("--ramp-steps", type=int, default=2000)
    parser.add_argument("--d-warmup", type=int, default=1000)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json")).read_text(encoding="utf-8"))

    train_data = SlotDataset(args.train_manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=2)
    val_data = SlotDataset(args.val_manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=2)
    loader = DataLoader(train_data, batch_sampler=SameNBatchSampler(train_data.rows, args.batch_size,
                                                                    shuffle=True, seed=args.seed),
                        collate_fn=collate_slot_batch, num_workers=0)
    val_batches = [collate_slot_batch([val_data[i]]) for i in range(min(args.val_batches, len(val_data)))]
    val_batches = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()} for b in val_batches]

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from vifasttts.frontend.vocab import ComponentVocabs

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
             len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    model = build_model(config, sizes, arch=arch_of_checkpoint(state if isinstance(state, dict) else {}))
    load_model_state(model, state.get("model", state))
    if hasattr(model, "load_f0_stats") and isinstance(state, dict) and state.get("f0_stats"):
        model.load_f0_stats(dict(state["f0_stats"]))
    model.to(device)
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(TRAIN_PREFIXES))
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() for p in trainable)/1e6:.2f}M / {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    mel_cfg = config["mel"]
    bank = mel_filterbank(int(config.get("sample_rate", 24000)), int(mel_cfg["n_fft"]), int(mel_cfg["n_mels"]),
                          float(mel_cfg.get("f_min", 0)), float(mel_cfg.get("f_max", 12000)),
                          norm=mel_cfg.get("norm", "slaney")).numpy()
    freqs = np.linspace(0, int(config.get("sample_rate", 24000)) / 2, bank.shape[1])
    centers = np.array([freqs[bank[i] > 0].mean() if (bank[i] > 0).any() else 0 for i in range(100)])
    band = (centers >= 2000) & (centers <= 6000)

    adversarial = config.get("adversarial", {})
    discriminator = MultiScaleMelDiscriminator(
        n_mels=int(mel_cfg["n_mels"]), scales=tuple(int(s) for s in adversarial.get("scales", (1, 2, 4))),
        channels=int(adversarial.get("channels", 64)), layers=int(adversarial.get("layers", 3)),
        spectral_norm=bool(adversarial.get("spectral_norm", True))).to(device)
    opt_g = torch.optim.AdamW(trainable, lr=args.g_lr, betas=(0.5, 0.9), weight_decay=0.0)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.9), weight_decay=0.0)

    ema = {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}
    weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
    weights.update({"mel": 1.0, "mel_pre": 0.2, "mel_sharp": 1.0})

    base = evaluate(model, val_batches, device, centers, band)
    print(f"baseline (EMA=live): val_mel {base['val_mel']:.4f} · GV voiced {base['gv_voiced']:.3f} · 2-6k {base['gv_2_6k']:.3f}")
    history = out / "history.csv"
    with history.open("w", newline="") as fh:
        csv.writer(fh).writerow(["step", "g_loss", "mel", "mel_pre", "mel_sharp", "adv", "fm", "d_loss",
                                 "val_mel", "gv_voiced", "gv_2_6k"])

    step = 0
    bad_d = 0
    model.train(); discriminator.train()
    rng = random.Random(args.seed)
    t0 = time.time()
    while step < args.max_steps:
        for batch in loader:
            step += 1
            moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(moved["features"], durations=moved["durations"], seg_durations=moved["seg_durations"],
                            gaps=moved["gaps"], lead=moved["lead"])
            loss, parts = slot_losses(outputs, moved, weights, shape_mean=model.shape_mean,
                                      shape_std=model.shape_std, tone_prior=model.tone_prior_z)
            gap = gap_mask(outputs["slot_durations"])[:, :outputs["mel"].size(1)]
            real_c, fake_c = sample_crops(moved["mel"], outputs["mel"], gap, rng)
            # ── D step (D học NGAY từ đầu; G đứng yên trong `d_warmup`) ──
            opt_d.zero_grad(set_to_none=True)
            d_loss = hinge_d(discriminator(real_c), discriminator(fake_c.detach()))
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            opt_d.step()
            d_value = float(d_loss.detach())
            bad_d = bad_d + 1 if d_value < 0.05 else 0
            parts["d"] = torch.tensor(d_value)
            # ── G step ──
            if step > args.d_warmup:
                for p in discriminator.parameters():
                    p.requires_grad_(False)
                g_adv = hinge_g(discriminator(fake_c))
                g_fm = feature_matching_loss(discriminator(real_c), discriminator(fake_c))
                for p in discriminator.parameters():
                    p.requires_grad_(True)
                ramp = min(1.0, (step - args.d_warmup) / max(1, args.ramp_steps))
                loss = loss + ramp * (args.adv_weight * g_adv + args.fm_weight * g_fm)
                parts["adv"] = g_adv.detach(); parts["fm"] = g_fm.detach()
            else:
                g_adv = torch.zeros(()); g_fm = torch.zeros(())
            if step > args.d_warmup:   # D warmup: G đứng yên
                opt_g.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt_g.step()
                else:
                    print(f"[{step}] NaN — dừng khẩn"); raise SystemExit(2)
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if torch.is_tensor(v) and k in ema:
                        ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
            if step % args.log_every == 0:
                # evaluate với EMA (đổi tạm trọng số trainable)
                live = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if torch.is_tensor(v) and k.split(".")[0] in ("decoder", "mel", "postnet")}
                model.load_state_dict({**model.state_dict(), **{k: ema[k] for k in live}}, strict=False)
                ev = evaluate(model, val_batches, device, centers, band)
                model.load_state_dict({**model.state_dict(), **live}, strict=False)
                if ev["val_mel"] > base["val_mel"] * 1.10:
                    print(f"[{step}] val mel L1 tăng >10% — dừng khẩn"); raise SystemExit(3)
                with history.open("a", newline="") as fh:
                    csv.writer(fh).writerow([step, f"{float(loss):.4f}", f"{float(parts['mel']):.4f}",
                                             f"{float(parts['mel_pre']):.4f}", f"{float(parts['mel_sharp']):.4f}",
                                             f"{float(g_adv):.4f}", f"{float(g_fm):.4f}", f"{d_value:.4f}",
                                             f"{ev['val_mel']:.4f}", f"{ev['gv_voiced']:.4f}", f"{ev['gv_2_6k']:.4f}"])
                print(f"step {step}: mel {float(parts['mel']):.3f} · d {d_value:.3f} · adv {float(g_adv):.3f} "
                      f"· val {ev['val_mel']:.4f} · GV {ev['gv_voiced']:.3f} (2-6k {ev['gv_2_6k']:.3f}) "
                      f"· {time.time()-t0:.0f}s")
                if bad_d >= 500:
                    print("D loss < 0.05 liên tục 500 bước — dừng khẩn"); raise SystemExit(4)
            if args.save_every and step % args.save_every == 0:
                torch.save({"model": model.state_dict(), "ema": ema, "step": step, "ema_decay": args.ema},
                           out / f"ckpt_{step}.pt")
            if step >= args.max_steps:
                break
    torch.save({"model": model.state_dict(), "ema": ema, "step": step, "ema_decay": args.ema}, out / "last.pt")
    print(f"xong {step} bước → {out}")


if __name__ == "__main__":
    main()

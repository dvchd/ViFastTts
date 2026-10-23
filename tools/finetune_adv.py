"""tools/finetune_adv.py — Adversarial fine-tune decoder+postnet từ checkpoint (spec A0 21/10 lượt 3+4).

Spec:
- **Đóng băng**: embedding, encoder, duration, slot_expander/ratio/variance, pause, các projection.
  **Train**: `decoder`, `mel` (đầu ra), `postnet` (2,82M/7,05M).
- L_rec = mel 1.0 + mel_pre 0.2 + mel_sharp 1.0.
- G: L_rec + w_adv·L_adv + w_fm·L_fm (hinge, crop 64 frame ≥80% vùng tiếng, D đóng băng khi train G).
- D: `MultiScaleMelDiscriminator` + spectral norm, **hinge**, D học ngay từ bước 1; **G đứng yên**
  trong `--d-warmup` bước đầu; **không ramp** (adv đủ ngay sau warmup, `--ramp-steps 0`).
- **Adaptive (P3/P4)**: λ = ||grad L_rec|| / (||grad L_adv|| + 1e-4) trên **Conv cuối postnet**,
  clamp [0, 10], detach, ×0,5. (Tính lại mỗi `--lambda-every` bước — điểm log — và giữ giữa các lần;
  khác spec "mỗi bước" để giảm 2 lượt backward/bước, ghi trong báo cáo.)
- Log mỗi `--log-every`: D loss, **D acc**, **grad ratio adv/rec trên decoder**, λ.
- Val mỗi `--val-every` bằng **CẢ trọng số sống và EMA**, cùng 10 batch val cố định.
- EMA 0.999; dừng khẩn chỉ khi bật `--hard-stops` (D<0,05 liên tục 500 · val +10% · NaN).

Ví dụ lưới probe 2k (A0 lượt 4):
  for cfg in P0 P1 P2 P3 P4; do ... --max-steps 2000 --d-warmup 300 ... ; done
  P0: --adv-weight 0 --fm-weight 0   · P1: --adv-weight 0.1 --fm-weight 0
  P2: --adv-weight 0.3 --fm-weight 0 · P3: --adv-mode adaptive --fm-weight 0
  P4: --adv-mode adaptive --fm-weight 0.5
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

TRAIN_PREFIXES = ("decoder.", "mel.", "postnet.", "gate_bias.")


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
                 frames: int = 64, min_speech: float = 0.8,
                 lengths: torch.Tensor | None = None,
                 return_windows: bool = False) -> tuple:
    """Crop cùng vị trí cho real/fake; ưu tiên cửa sổ ≥80% frame vùng tiếng.

    `lengths` (v18.37): độ dài thật từng row. Khi truyền, hàm **không bao giờ crop vào
    padding**: `width = min(frames, min(lengths))` và mỗi row chỉ lấy start trong
    `[0, length-width]`. Nếu bỏ qua, giữ hành vi cũ (dùng Tmax của batch) — chỉ dùng
    cho test/đối chứng cũ, không dùng trong train.

    `return_windows=True` trả thêm `[(start,end)]` để audit.
    """
    batch, total, _bins = real.shape
    min_length = int(lengths.min()) if lengths is not None else total
    width = min(frames, total, min_length)
    out_r, out_f = [], []
    windows: list[tuple[int, int]] = []
    for b in range(batch):
        row_total = int(lengths[b]) if lengths is not None else total
        row_total = max(width, min(row_total, total))
        best, best_frac = 0, -1.0
        for _ in range(8):
            start = rng.randint(0, max(0, row_total - width))
            frac = 1.0 - float(gap[b, start:start + width].float().mean())
            if frac >= min_speech:
                best = start
                break
            if frac > best_frac:
                best, best_frac = start, frac
        end = best + width
        out_r.append(real[b, best:end])
        out_f.append(fake[b, best:end])
        windows.append((int(best), int(end)))
    if return_windows:
        return torch.stack(out_r), torch.stack(out_f), windows
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


def d_accuracy(real_outs, fake_outs) -> float:
    """Tỉ lệ D(real) > D(fake), trung bình qua các scale."""
    values = []
    for (r, _), (f, _) in zip(real_outs, fake_outs):
        values.append(float((r > f).float().mean()))
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def evaluate(model, batches, band_2_6k) -> dict:
    """val mel L1 (toàn row) + GV hữu thanh (median + dải 2-6 kHz) trên các batch cho trước."""
    was_training = model.training
    model.eval()
    l1_sum, l1_n = 0.0, 0
    num_v = np.zeros(100)
    den_v = np.zeros(100)
    for batch in batches:
        out = model(batch["features"], durations=batch["durations"],
                    seg_durations=batch["seg_durations"], gaps=batch["gaps"], lead=batch["lead"],
                    gate=batch.get("gate"))
        pred, real = out["mel"], batch["mel"]
        frames = min(pred.size(1), real.size(1))
        pred, real = pred[:, :frames], real[:, :frames]
        steps = torch.arange(frames, device=pred.device).view(1, -1)
        valid = steps < batch["frame_lengths"].unsqueeze(1)
        l1_sum += float(((pred - real).abs().mean(-1) * valid).sum())
        l1_n += int(valid.sum())
        gap = gap_mask(out["slot_durations"])[:, :frames]
        vuv = batch["vuv"][:, :frames]
        voice = (~gap) & (vuv >= 0.5) & valid
        if int(voice.sum()) < 3:
            continue
        num_v += pred[voice].var(dim=0).detach().cpu().numpy() * int(voice.sum())
        den_v += real[voice].var(dim=0).detach().cpu().numpy() * int(voice.sum())
    model.train(was_training)
    ratio = num_v / np.maximum(den_v, 1e-9)
    return {"val_mel": l1_sum / max(l1_n, 1),
            "gv_voiced": float(np.median(ratio)),
            "gv_2_6k": float(np.median(ratio[band_2_6k]))}


def evaluate_live_and_ema(model, batches, band, ema: dict) -> dict:
    """Đo val bằng trọng số SỐNG rồi EMA (đổi tạm rồi trả lại) — A0 lượt 4 yêu cầu cả hai."""
    live = {k: v.detach().clone() for k, v in model.state_dict().items()
            if torch.is_tensor(v) and k.split(".")[0] in ("decoder", "mel", "postnet")}
    ev_live = evaluate(model, batches, band)
    if ema:
        model.load_state_dict({**model.state_dict(), **{k: ema[k] for k in live}}, strict=False)
        ev_ema = evaluate(model, batches, band)
        model.load_state_dict({**model.state_dict(), **live}, strict=False)
    else:
        ev_ema = dict(ev_live)
    return {"live": ev_live, "ema": ev_ema}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--g-lr", type=float, default=2e-5)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--adv-mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--adv-weight", type=float, default=0.1)
    parser.add_argument("--adv-scale", type=float, default=0.5, help="hệ số nhân λ adaptive")
    parser.add_argument("--fm-weight", type=float, default=0.0)
    parser.add_argument("--energy-weight", type=float, default=0.0,
                        help="E1: L1 trên log tổng năng lượng mel tuyến tính (neo mức to tuyệt đối)")
    parser.add_argument("--lowband-weight", type=float, default=0.0,
                        help="E2: hệ số nhân L1 dải 0–1 kHz (mặc định 2.0 khi bật)")
    parser.add_argument("--ramp-steps", type=int, default=0)
    parser.add_argument("--d-warmup", type=int, default=300)
    parser.add_argument("--lambda-every", type=int, default=50)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--hard-stops", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    f0_stats = json.loads(Path(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json")).read_text(encoding="utf-8"))

    train_data = SlotDataset(args.train_manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=2)
    val_data = SlotDataset(args.val_manifest, config, f0_stats, cache_dir=args.cache_dir, min_syllables=2)
    # augmentation đổi N sau __getitem__ ⇒ sampler phải gộp theo N SAU cắt (v3.5)
    loader = DataLoader(train_data, batch_sampler=SameNBatchSampler(train_data.rows, args.batch_size,
                                                                    shuffle=True, seed=args.seed,
                                                                    lengths=train_data.effective_lengths()),
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
    dec_params = [p for n, p in model.named_parameters() if n.startswith("decoder.") and p.requires_grad]
    post_head = model.postnet[-1].weight
    print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.2f}M / "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    mel_cfg = config["mel"]
    bank = mel_filterbank(int(config.get("sample_rate", 24000)), int(mel_cfg["n_fft"]), int(mel_cfg["n_mels"]),
                          float(mel_cfg.get("f_min", 0)), float(mel_cfg.get("f_max", 12000)),
                          norm=mel_cfg.get("norm", "slaney")).numpy()
    freqs = np.linspace(0, int(config.get("sample_rate", 24000)) / 2, bank.shape[1])
    centers = np.array([freqs[bank[i] > 0].mean() if (bank[i] > 0).any() else 0 for i in range(100)])
    band = (centers >= 2000) & (centers <= 6000)
    lowband_idx = torch.tensor(np.where(centers <= 1000.0)[0], device=device, dtype=torch.long)
    mel_stats = json.loads(Path("data/codelung/mel_stats.json").read_text(encoding="utf-8"))
    mel_mean_t = torch.tensor(mel_stats["mean"], device=device, dtype=torch.float32)
    mel_std_t = torch.tensor(mel_stats["std"], device=device, dtype=torch.float32)

    adversarial = config.get("adversarial", {})
    discriminator = MultiScaleMelDiscriminator(
        n_mels=int(mel_cfg["n_mels"]), scales=tuple(int(s) for s in adversarial.get("scales", (1, 2, 4))),
        channels=int(adversarial.get("channels", 64)), layers=int(adversarial.get("layers", 3)),
        spectral_norm=bool(adversarial.get("spectral_norm", True))).to(device)
    use_d = args.adv_mode == "adaptive" or args.adv_weight > 0 or args.fm_weight > 0
    opt_g = torch.optim.AdamW(trainable, lr=args.g_lr, betas=(0.5, 0.9), weight_decay=0.0)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.9), weight_decay=0.0)

    ema = {k: v.detach().clone() for k, v in model.state_dict().items() if torch.is_tensor(v)}
    weights = {k: 0.0 for k in DEFAULT_WEIGHTS}
    weights.update({"mel": 1.0, "mel_pre": 0.2, "mel_sharp": 1.0})

    base = evaluate_live_and_ema(model, val_batches, band, ema)
    print(f"baseline live: val {base['live']['val_mel']:.4f} · GV {base['live']['gv_voiced']:.3f} "
          f"(2-6k {base['live']['gv_2_6k']:.3f})")
    run = {"args": vars(args), "baseline": base,
           "adv_mode": args.adv_mode, "adv_weight": args.adv_weight, "fm_weight": args.fm_weight}
    (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    train_log = out / "train_log.csv"
    val_log = out / "val_log.csv"
    with train_log.open("w", newline="") as fh:
        csv.writer(fh).writerow(["step", "rec", "mel", "mel_pre", "mel_sharp", "adv", "fm", "adv_eff", "lam",
                                 "grad_ratio", "d_loss", "d_acc", "energy", "lowband", "sec"])
    with val_log.open("w", newline="") as fh:
        csv.writer(fh).writerow(["step", "live_val", "live_gv", "live_gv26", "ema_val", "ema_gv", "ema_gv26"])

    step = 0
    bad_d = 0
    lam = None
    model.train()
    discriminator.train()
    rng = random.Random(args.seed)
    t0 = time.time()
    while step < args.max_steps:
        for batch in loader:
            step += 1
            moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(moved["features"], durations=moved["durations"], seg_durations=moved["seg_durations"],
                            gaps=moved["gaps"], lead=moved["lead"], gate=moved.get("gate"))
            rec, parts = slot_losses(outputs, moved, weights, shape_mean=model.shape_mean,
                                     shape_std=model.shape_std, tone_prior=model.tone_prior_z)
            # E1/E2: neo mức to TUYỆT ĐỐI của từng frame (D chỉ nhìn crop 64 frame nên không thấy).
            energy_loss = torch.zeros((), device=device)
            lowband_loss = torch.zeros((), device=device)
            if args.energy_weight > 0 or args.lowband_weight > 0:
                frames_e = min(outputs["mel"].size(1), moved["mel"].size(1))
                steps_e = torch.arange(frames_e, device=device).view(1, -1)
                valid_e = steps_e < moved["frame_lengths"].unsqueeze(1)
                voice_e = ((~gap_mask(outputs["slot_durations"])[:, :frames_e])
                           & (moved["vuv"][:, :frames_e] >= 0.5) & valid_e)
                if int(voice_e.sum()) >= 3:
                    pred_z = outputs["mel"][:, :frames_e]
                    real_z = moved["mel"][:, :frames_e]
                    e_pred = torch.logsumexp(pred_z * mel_std_t + mel_mean_t, dim=-1)
                    e_real = torch.logsumexp(real_z * mel_std_t + mel_mean_t, dim=-1)
                    energy_loss = (e_pred - e_real).abs()[voice_e].mean()
                    lowband_loss = (pred_z[..., lowband_idx] - real_z[..., lowband_idx]).abs().mean(-1)[voice_e].mean()
            gap = gap_mask(outputs["slot_durations"])[:, :outputs["mel"].size(1)]
            real_c, fake_c = sample_crops(moved["mel"], outputs["mel"], gap, rng,
                                          lengths=moved["frame_lengths"])
            # ── D step (D học ngay từ bước 1) ──
            d_value, d_acc = float("nan"), float("nan")
            if use_d:
                real_out = discriminator(real_c)
                fake_out = discriminator(fake_c.detach())
                opt_d.zero_grad(set_to_none=True)
                d_loss = hinge_d(real_out, fake_out)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                opt_d.step()
                d_value = float(d_loss.detach())
                d_acc = d_accuracy(real_out, fake_out)
                bad_d = bad_d + 1 if d_value < 0.05 else 0
            # ── G step (chỉ sau warmup) ──
            g_adv = torch.zeros((), device=device)
            g_fm = torch.zeros((), device=device)
            grad_ratio = float("nan")
            if step > args.d_warmup:
                for p in discriminator.parameters():
                    p.requires_grad_(False)
                if use_d:
                    g_adv = hinge_g(discriminator(fake_c))
                    if args.fm_weight > 0:
                        g_fm = feature_matching_loss(discriminator(real_c), discriminator(fake_c))
                for p in discriminator.parameters():
                    p.requires_grad_(True)
                every = args.lambda_every if args.adv_mode == "adaptive" else args.log_every
                measure = (lam is None) if args.adv_mode == "adaptive" else (step % every == 0)
                if measure and use_d:
                    # đo grad rec và grad adv riêng trên decoder + Conv cuối postnet
                    opt_g.zero_grad(set_to_none=True)
                    rec.backward(retain_graph=True)
                    g_rec_dec = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in dec_params if p.grad is not None)))
                    g_rec_post = float(post_head.grad.detach().norm())
                    opt_g.zero_grad(set_to_none=True)
                    g_adv.backward(retain_graph=True)
                    g_adv_dec = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in dec_params if p.grad is not None)))
                    g_adv_post = float(post_head.grad.detach().norm())
                    opt_g.zero_grad(set_to_none=True)
                    # LƯU Ý: đây là tỉ lệ gradient **THÔ** của hinge_g / rec,
                    # CHƯA nhân adv_weight/eff. Với P2 fixed 0,3, tỉ lệ hiệu dụng
                    # xấp xỉ 0,3 × grad_ratio; loss lớn không tự chứng minh gradient chi phối.
                    grad_ratio = g_adv_dec / (g_rec_dec + 1e-8)
                    if args.adv_mode == "adaptive":
                        lam = float(min(10.0, max(0.0, g_rec_post / (g_adv_post + 1e-4)))) * args.adv_scale
                    elif lam is None:
                        lam = 0.0
                ramp = 1.0 if args.ramp_steps <= 0 else min(1.0, (step - args.d_warmup) / args.ramp_steps)
                weight = args.adv_weight if args.adv_mode == "fixed" else (lam or 0.0)
                eff = ramp * weight
                total = (rec + eff * g_adv + ramp * args.fm_weight * g_fm
                         + args.energy_weight * energy_loss + args.lowband_weight * lowband_loss)
                opt_g.zero_grad(set_to_none=True)
                if torch.isfinite(total):
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt_g.step()
                else:
                    print(f"[{step}] NaN — dừng khẩn")
                    raise SystemExit(2)
            else:
                eff = 0.0
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if torch.is_tensor(v) and k in ema:
                        ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
            if step % args.log_every == 0:
                rec_value = float(rec.detach())
                with train_log.open("a", newline="") as fh:
                    csv.writer(fh).writerow([step, f"{rec_value:.5f}", f"{float(parts['mel']):.5f}",
                                             f"{float(parts['mel_pre']):.5f}", f"{float(parts['mel_sharp']):.5f}",
                                             f"{float(g_adv):.4f}", f"{float(g_fm):.4f}", f"{eff:.4f}",
                                             f"{(lam or 0.0):.4f}", f"{grad_ratio:.4f}", f"{d_value:.4f}", f"{d_acc:.4f}",
                                             f"{float(energy_loss):.4f}", f"{float(lowband_loss):.4f}",
                                             f"{time.time() - t0:.0f}"])
            if step % args.val_every == 0 or step >= args.max_steps:
                ev = evaluate_live_and_ema(model, val_batches, band, ema)
                with val_log.open("a", newline="") as fh:
                    csv.writer(fh).writerow([step, f"{ev['live']['val_mel']:.5f}", f"{ev['live']['gv_voiced']:.4f}",
                                             f"{ev['live']['gv_2_6k']:.4f}", f"{ev['ema']['val_mel']:.5f}",
                                             f"{ev['ema']['gv_voiced']:.4f}", f"{ev['ema']['gv_2_6k']:.4f}"])
                print(f"step {step}: rec {rec_value:.3f} · d {d_value:.3f} acc {d_acc:.2f} · eff {eff:.3f} "
                      f"· ratio {grad_ratio:.3f} · LIVE val {ev['live']['val_mel']:.4f} GV {ev['live']['gv_voiced']:.3f} "
                      f"· EMA val {ev['ema']['val_mel']:.4f} GV {ev['ema']['gv_voiced']:.3f} · {time.time() - t0:.0f}s")
                if args.hard_stops and ev["live"]["val_mel"] > base["live"]["val_mel"] * 1.10:
                    print(f"[{step}] val mel L1 tăng >10% — dừng khẩn")
                    raise SystemExit(3)
            if args.hard_stops and bad_d >= 500:
                print("D loss < 0.05 liên tục 500 bước — dừng khẩn")
                raise SystemExit(4)
            if args.save_every and step % args.save_every == 0:
                torch.save({"model": model.state_dict(), "ema": ema, "step": step, "ema_decay": args.ema,
                            "arch": getattr(model, "arch", "v1_syl")}, out / f"ckpt_{step}.pt")
            if step >= args.max_steps:
                break
    torch.save({"model": model.state_dict(), "ema": ema, "step": step, "ema_decay": args.ema,
                "arch": getattr(model, "arch", "v1_syl")}, out / "last.pt")
    print(f"xong {step} bước → {out}")


if __name__ == "__main__":
    main()

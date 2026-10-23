"""v18.38 — probe crop D có đối chứng (300 D warm-up + 200 G update).

Ba nhánh từ **cùng P2 EMA**, cùng batch order / seed / optimizer G / EMA:
  rec          — mel + mel_pre + mel_sharp, không adversarial (200 G update)
  gan_legacy   — crop cũ (không truyền lengths), 300 D + 200 G
  gan_fixed    — crop đã sửa (truyền frame_lengths), 300 D + 200 G

Không đổi production, không train dài, không ép vuv. Chỉ lưu checkpoint/JSON/CSV
dưới `runs/v1838_crop_probe/`. Đánh giá cuối nằm ở `tools/v1838_eval.py`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from finetune_adv import d_accuracy, gap_mask, hinge_d, hinge_g, sample_crops

from vifasttts.data.dataset_v2 import SameNBatchSampler, SlotDataset, collate_slot_batch
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.models.discriminators import MultiScaleMelDiscriminator
from vifasttts.models.factory import arch_of_checkpoint, build_model, load_model_state
from vifasttts.models.v2_losses import DEFAULT_WEIGHTS, slot_losses

CFG32 = "configs/small_codelung_v32.yaml"
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
TRAIN = "data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl"
CACHE = "cache/codelung_v2"
OUT = Path("runs/v1838_crop_probe")
BRANCHES = ("rec", "gan_legacy", "gan_fixed")
TRAIN_PREFIXES = ("decoder.", "mel.", "postnet.", "gate_bias.")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha(state: dict) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()[:16]


def maybe_finite(name: str, value, step: int) -> float:
    number = float(value.detach()) if torch.is_tensor(value) else float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"NaN/Inf ở {name} step {step}: {number}")
    return number


def make_model(cfg: dict, sizes: list[int]) -> torch.nn.Module:
    state = torch.load(P2, map_location="cpu", weights_only=False)
    model = build_model(cfg, sizes, arch=arch_of_checkpoint(state))
    load_model_state(model, state["model"])
    if hasattr(model, "load_f0_stats") and state.get("f0_stats"):
        model.load_f0_stats(dict(state["f0_stats"]))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(TRAIN_PREFIXES))
    return model


def make_discriminator(cfg: dict, seed: int) -> MultiScaleMelDiscriminator:
    torch.manual_seed(seed)
    adv = cfg.get("adversarial", {})
    return MultiScaleMelDiscriminator(
        n_mels=int(cfg["mel"]["n_mels"]),
        scales=tuple(int(s) for s in adv.get("scales", (1, 2, 4))),
        channels=int(adv.get("channels", 64)),
        layers=int(adv.get("layers", 3)),
        spectral_norm=bool(adv.get("spectral_norm", True)),
    )


def crop_rows(windows, lengths) -> list[dict]:
    rows = []
    for row_index, (start, end) in enumerate(windows):
        length = int(lengths[row_index])
        width = end - start
        valid = max(0, min(end, length) - start)
        padding = width - valid
        rows.append({
            "row": row_index, "frame_length": length,
            "crop_start": int(start), "crop_end": int(end), "crop_width": int(width),
            "padding_frames": int(padding),
            "padding_ratio": round(padding / max(1, width), 4),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CFG32)
    parser.add_argument("--checkpoint", default=P2)
    parser.add_argument("--train-manifest", default=TRAIN)
    parser.add_argument("--cache-dir", default=CACHE)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-warmup", type=int, default=300)
    parser.add_argument("--g-steps", type=int, default=200)
    parser.add_argument("--g-lr", type=float, default=2e-5)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--adv-weight", type=float, default=0.3)
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = json.loads(Path("data/codelung/f0_stats_v2.json").read_text(encoding="utf-8"))
    dataset = SlotDataset(args.train_manifest, cfg, f0_stats, cache_dir=args.cache_dir, min_syllables=2)
    sampler = SameNBatchSampler(dataset.rows, args.batch_size, shuffle=True, seed=args.seed,
                                lengths=dataset.effective_lengths())
    total_steps = args.d_warmup + args.g_steps
    batch_indices = []
    for index, indices in enumerate(sampler):
        if index >= total_steps:
            break
        batch_indices.append(list(indices))
    if len(batch_indices) != total_steps:
        raise RuntimeError(f"không đủ batch: {len(batch_indices)} < {total_steps}")
    batch_hash = hashlib.sha256(json.dumps(batch_indices).encode()).hexdigest()

    lengths = [row["frame_range"][1] - row["frame_range"][0] for row in dataset.rows]
    short_rows = sum(1 for value in lengths if value < 64)
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                   capture_output=True, text=True, check=False).stdout.strip()
    git_status = subprocess.run(["git", "status", "--short"], cwd=_ROOT,
                                capture_output=True, text=True, check=False).stdout.strip().splitlines()

    vocab = ComponentVocabs.default()
    sizes = [len(vocab.onset), len(vocab.medial), len(vocab.nucleus), len(vocab.coda),
             len(vocab.tone), len(vocab.boundary), len(vocab.punctuation)]
    device = torch.device(args.device)
    d_template = make_discriminator(cfg, args.seed).to(device)
    d_template_state = copy.deepcopy(d_template.state_dict())
    initial_model = make_model(cfg, sizes)
    g_init_sha = state_sha(initial_model.state_dict())
    d_init_sha = state_sha(d_template_state)
    trainable_names = [name for name, p in initial_model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in initial_model.parameters() if p.requires_grad)

    # drop template model khỏi RAM; mỗi nhánh tự nạp P2.
    del initial_model

    weights = {key: 0.0 for key in DEFAULT_WEIGHTS}
    weights.update({"mel": 1.0, "mel_pre": 0.2, "mel_sharp": 1.0})

    resolved = {
        "probe": "v18.38 crop D có đối chứng",
        "source_commit": source_commit,
        "git_status_at_start": git_status,
        "config": cfg,
        "args": vars(args),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest": args.train_manifest,
        "manifest_sha256": sha256(args.train_manifest),
        "batch_order_hash": batch_hash,
        "batch_count": len(batch_indices),
        "seed": args.seed,
        "d_warmup": args.d_warmup,
        "g_steps": args.g_steps,
        "total_steps": total_steps,
        "g_init_sha": g_init_sha,
        "d_init_sha": d_init_sha,
        "trainable_names": trainable_names,
        "trainable_count": trainable_count,
        "dataset_rows": len(dataset.rows),
        "short_rows_lt64": short_rows,
        "short_rows_ratio": round(short_rows / max(1, len(dataset.rows)), 4),
        "branches": BRANCHES,
        "note": "EMA chỉ cập nhật ở bước G; D warm-up không cập nhật EMA",
    }
    (out / "resolved.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    (out / "batch_indices.json").write_text(json.dumps(batch_indices), encoding="utf-8")
    start_rng = random.Random(args.seed)
    torch.save({"python_global": random.getstate(), "crop": start_rng.getstate(),
                "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
               out / "rng_start.pt")

    crop_file = (out / "crop_trace.csv").open("w", newline="", encoding="utf-8")
    crop_writer = csv.writer(crop_file)
    crop_writer.writerow(["branch", "step", "g_step", "row", "frame_length", "crop_start",
                          "crop_end", "crop_width", "padding_frames", "padding_ratio"])

    def load_batch(step: int) -> dict:
        return collate_slot_batch([dataset[i] for i in batch_indices[step]])

    def move(batch: dict) -> dict:
        return {key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()}

    def run_branch(name: str) -> dict:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        model = make_model(cfg, sizes).to(device)
        model.train()
        opt_g = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.g_lr, betas=(0.5, 0.9), weight_decay=0.0)
        ema = {key: value.detach().clone() for key, value in model.state_dict().items()
               if torch.is_tensor(value)}
        discriminator = None
        opt_d = None
        if name != "rec":
            discriminator = make_discriminator(cfg, args.seed).to(device)
            discriminator.load_state_dict(copy.deepcopy(d_template_state))
            discriminator.train()
            opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr,
                                      betas=(0.5, 0.9), weight_decay=0.0)
        rng = random.Random(args.seed)
        dec_params = [p for n, p in model.named_parameters() if n.startswith("decoder.") and p.requires_grad]
        csv_path = out / f"train_{name}.csv"
        handle = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(handle)
        writer.writerow(["step", "g_step", "phase", "rec", "mel", "mel_pre", "mel_sharp",
                         "adv_raw", "adv_weighted", "grad_ratio_raw", "grad_ratio_eff",
                         "grad_g_norm", "grad_d_norm", "d_loss", "d_acc",
                         "crop_padding_ratio", "sec"])
        torch.manual_seed(args.seed + args.d_warmup)
        initial_batch = move(load_batch(args.d_warmup))
        with torch.no_grad():
            initial_out = model(initial_batch["features"], durations=initial_batch["durations"],
                                seg_durations=initial_batch["seg_durations"],
                                gaps=initial_batch["gaps"], lead=initial_batch["lead"])
            initial_rec, initial_parts = slot_losses(
                initial_out, initial_batch, weights, shape_mean=model.shape_mean,
                shape_std=model.shape_std, tone_prior=model.tone_prior_z)
        milestones = {0: {
            "step": args.d_warmup, "g_step": 0,
            "rec": float(initial_rec.detach()), "mel": float(initial_parts["mel"].detach()),
            "mel_pre": float(initial_parts["mel_pre"].detach()),
            "mel_sharp": float(initial_parts["mel_sharp"].detach()),
            "adv_raw": 0.0, "adv_weighted": 0.0, "grad_ratio_raw": float("nan"),
            "grad_ratio_eff": float("nan"), "grad_g_norm": 0.0, "grad_d_norm": 0.0,
            "d_loss": float("nan"), "d_acc": float("nan"), "crop_padding_ratio": float("nan"),
        }}
        started = time.time()
        step = 0
        for step in range(1, total_steps + 1):
            if name == "rec" and step <= args.d_warmup:
                continue
            if step == args.d_warmup + 1:
                # Căn RNG dropout của generator ở đầu pha G cho cả 3 nhánh.
                # REC không chạy 300 forward D warm-up nên nếu không reset sẽ lệch mask.
                torch.manual_seed(args.seed + args.d_warmup + 1)
            batch = move(load_batch(step - 1))
            outputs = model(batch["features"], durations=batch["durations"],
                            seg_durations=batch["seg_durations"], gaps=batch["gaps"],
                            lead=batch["lead"])
            rec, parts = slot_losses(outputs, batch, weights, shape_mean=model.shape_mean,
                                     shape_std=model.shape_std, tone_prior=model.tone_prior_z)
            for key in ("mel", "mel_pre", "mel_sharp"):
                maybe_finite(key, parts[key], step)
            maybe_finite("rec", rec, step)

            g_step = step - args.d_warmup
            phase = "d_warmup" if g_step <= 0 else "g"
            adv_raw = 0.0
            adv_weighted = 0.0
            grad_ratio_raw = float("nan")
            grad_ratio_eff = float("nan")
            grad_g_norm = 0.0
            grad_d_norm = 0.0
            d_loss_value = float("nan")
            d_acc_value = float("nan")
            crop_padding_ratio = float("nan")

            if discriminator is not None:
                gap = gap_mask(outputs["slot_durations"])[:, :outputs["mel"].size(1)]
                lengths_t = batch["frame_lengths"]
                use_fixed = name == "gan_fixed"
                real_c, fake_c, windows = sample_crops(
                    batch["mel"], outputs["mel"], gap, rng,
                    lengths=lengths_t if use_fixed else None,
                    return_windows=True,
                )
                rows = crop_rows(windows, lengths_t)
                for row in rows:
                    crop_writer.writerow([name, step, max(0, g_step), row["row"],
                                          row["frame_length"], row["crop_start"], row["crop_end"],
                                          row["crop_width"], row["padding_frames"],
                                          row["padding_ratio"]])
                crop_padding_ratio = float(np.mean([row["padding_ratio"] for row in rows]))
                real_out = discriminator(real_c)
                fake_out = discriminator(fake_c.detach())
                opt_d.zero_grad(set_to_none=True)
                d_loss = hinge_d(real_out, fake_out)
                d_loss_value = maybe_finite("d_loss", d_loss, step)
                d_acc_value = d_accuracy(real_out, fake_out)
                d_loss.backward()
                grad_d_norm = float(torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0))
                opt_d.step()
                if phase == "g":
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(False)
                    g_adv = hinge_g(discriminator(fake_c))
                    adv_raw = maybe_finite("adv_raw", g_adv, step)
                    adv_weighted = args.adv_weight * adv_raw
                    if g_step in (1, 100, 200) or g_step % args.log_every == 0:
                        opt_g.zero_grad(set_to_none=True)
                        rec.backward(retain_graph=True)
                        rec_dec = float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                                       for p in dec_params if p.grad is not None)))
                        opt_g.zero_grad(set_to_none=True)
                        g_adv.backward(retain_graph=True)
                        adv_dec = float(torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                                       for p in dec_params if p.grad is not None)))
                        opt_g.zero_grad(set_to_none=True)
                        grad_ratio_raw = adv_dec / (rec_dec + 1e-8)
                        grad_ratio_eff = args.adv_weight * grad_ratio_raw
                    total = rec + args.adv_weight * g_adv
                    opt_g.zero_grad(set_to_none=True)
                    maybe_finite("total", total, step)
                    total.backward()
                    grad_g_norm = float(torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0))
                    opt_g.step()
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(True)
                    with torch.no_grad():
                        for key, value in model.state_dict().items():
                            if torch.is_tensor(value) and key in ema:
                                ema[key].mul_(args.ema).add_(value.detach(), alpha=1 - args.ema)
            else:
                opt_g.zero_grad(set_to_none=True)
                rec.backward()
                grad_g_norm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0))
                opt_g.step()
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        if torch.is_tensor(value) and key in ema:
                            ema[key].mul_(args.ema).add_(value.detach(), alpha=1 - args.ema)

            if g_step in (1, 100, 200):
                milestones[g_step] = {
                    "step": step, "g_step": g_step, "rec": float(rec.detach()),
                    "mel": float(parts["mel"].detach()), "mel_pre": float(parts["mel_pre"].detach()),
                    "mel_sharp": float(parts["mel_sharp"].detach()),
                    "adv_raw": adv_raw, "adv_weighted": adv_weighted,
                    "grad_ratio_raw": grad_ratio_raw, "grad_ratio_eff": grad_ratio_eff,
                    "grad_g_norm": grad_g_norm, "grad_d_norm": grad_d_norm,
                    "d_loss": d_loss_value, "d_acc": d_acc_value,
                    "crop_padding_ratio": crop_padding_ratio,
                }
            if g_step > 0 and (g_step % args.log_every == 0 or g_step in (1, 100, 200)):
                elapsed = time.time() - started
                writer.writerow([step, g_step, phase, float(rec.detach()), float(parts["mel"].detach()),
                                 float(parts["mel_pre"].detach()), float(parts["mel_sharp"].detach()),
                                 adv_raw, adv_weighted, grad_ratio_raw, grad_ratio_eff,
                                 grad_g_norm, grad_d_norm, d_loss_value, d_acc_value,
                                 crop_padding_ratio, round(elapsed, 1)])
                handle.flush()
            if step % 50 == 0 and device.type == "mps":
                torch.mps.empty_cache()
            if g_step in (100, 200):
                save_state(name, step, g_step, model, ema, opt_g, opt_d, discriminator,
                           rng, batch_hash, args, out)
        handle.close()
        final = save_state(name, step, max(0, step - args.d_warmup), model, ema, opt_g, opt_d,
                           discriminator, rng, batch_hash, args, out)
        return {"branch": name, "steps": step, "g_updates": max(0, step - args.d_warmup),
                "d_updates": step if discriminator is not None else 0,
                "milestones": milestones, "final": final,
                "mode_snapshot": {
                    "model_training": model.training,
                    "top_level_training": {child_name: child.training
                                           for child_name, child in model.named_children()},
                    "dropout_p": float(cfg["model"].get("dropout", 0.0)),
                    "note": "frozen module vẫn ở train mode ⇒ dropout của chúng vẫn hoạt động",
                },
                "seconds": round(time.time() - started, 1)}

    def save_state(name, step, g_step, model, ema, opt_g, opt_d, discriminator, rng,
                   batch_hash, args, out_dir) -> dict:
        payload = {
            "branch": name, "step": step, "g_updates": g_step,
            "model": model.state_dict(), "ema": ema,
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict() if opt_d is not None else None,
            "discriminator": discriminator.state_dict() if discriminator is not None else None,
            "rng": {"python_global": random.getstate(), "crop": rng.getstate(),
                    "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
            "config": cfg, "source_commit": source_commit,
            "checkpoint_sha256": sha256(P2), "batch_order_hash": batch_hash,
            "ema_decay": args.ema, "adv_weight": args.adv_weight,
            "crop_mode": "legacy" if name == "gan_legacy" else ("fixed" if name == "gan_fixed" else "none"),
            "weights": weights,
        }
        path = out_dir / f"{name}_g{g_step:03d}.pt"
        torch.save(payload, path)
        return {"path": str(path), "sha256": sha256(path), "g_step": g_step}

    summaries = {}
    for branch in BRANCHES:
        print(f"=== {branch} ===", flush=True)
        summaries[branch] = run_branch(branch)
    crop_file.close()
    result = {"resolved": resolved, "branches": summaries}
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""Train acoustic v2 (`arch="v2_slot5"`) — warm-start từ B0 + giai đoạn A/B (docs/ACOUSTIC_V2.md §11).

- Giai đoạn A (~`phase_a_steps`): **đóng băng** `embedding/encoder/duration`, LR = base ×
  `phase_a_lr_scale` (0,25), grad clip 1.0; log **mel tách G vs hữu thanh**.
- Giai đoạn B: mở khóa toàn bộ, warm-up LR lên base trong `phase_b_warmup_steps` rồi cosine.
- Optimizer **mới hoàn toàn**; checkpoint ghi `arch`, `seg_method`/`seg_method_version`/
  `seg_weights_hash` (từ manifest), `f0_stats` + hash, `warm_start_from`.
- Val dùng chế độ `tf_seg` (d_syl/seg/gaps/lead thật); `best.pt` chọn theo **mel hữu thanh**.
- Dataloader: `SlotDataset` + `collate_slot_batch` — yêu cầu cùng N trong batch (mặc định batch 1).

Chạy:
    vifasttts train-v2 --config configs/small_codelung_v2.yaml \
        --manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
        --val-manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
        --output runs/v2_slot5 --warm-start runs/small_codelung_v2/best.pt
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from vifasttts.train import _mel_hash, _resolve_cache_dir, set_seed

FROZEN_PREFIXES = ("embedding.", "encoder.", "duration.")


def _sha16(payload: bytes) -> str:
    """16 ký tự đầu sha256 của bytes (ghi vào checkpoint để truy vết B0)."""
    return hashlib.sha256(payload).hexdigest()[:16]


def _f0_stats_hash(stats: dict) -> str:
    """Hash 16 ký tự của `f0_stats_v2` (đóng băng thang chuẩn hoá — §6)."""
    return _sha16(json.dumps(stats, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _gap_frame_mask(slot_durations) -> Any:
    """`slot_durations [B,1+5N]` → mask [B,T] True ở frame thuộc `G_lead`/slot G."""
    import torch

    batch, slots = slot_durations.shape
    gap_slot = torch.cat([
        torch.ones(batch, 1, dtype=torch.bool, device=slot_durations.device),
        (torch.arange(slots - 1, device=slot_durations.device) % 5 == 4).view(1, -1).expand(batch, -1),
    ], dim=1)
    frames = int(slot_durations.sum(dim=1).max())
    steps = torch.arange(frames, device=slot_durations.device)
    index = (slot_durations.cumsum(dim=1).unsqueeze(2) <= steps.view(1, 1, -1)).sum(dim=1)
    return torch.gather(gap_slot, 1, index.clamp(max=slots - 1))


def _mel_split(outputs: dict, batch: dict) -> tuple[Any, Any]:
    """(mel L1 trung bình trên frame slot G, trên frame hữu thanh) — §11 log tách."""
    import torch

    error = (outputs["mel"] - batch["mel"]).abs().mean(dim=-1)  # [B,T]
    mask = _gap_frame_mask(outputs["slot_durations"])
    valid = torch.arange(error.size(1), device=error.device).view(1, -1) < batch["frame_lengths"].unsqueeze(1)
    gap = mask & valid
    voiced = (~mask) & valid
    mel_gap = (error * gap).sum() / gap.sum().clamp_min(1)
    mel_voiced = (error * voiced).sum() / voiced.sum().clamp_min(1)
    return mel_gap, mel_voiced


def _move_batch(batch: dict, device) -> dict:
    """Chuyển tensor trong batch lên device (giữ `id` list)."""
    import torch

    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def _validate(model, loader, weights: dict, device, max_batches: int = 0) -> dict[str, float]:
    """Val chế độ `tf_seg`: trả {'total', 'mel', 'mel_voiced'} (best.pt theo mel hữu thanh)."""
    import torch

    from vifasttts.models.v2_losses import slot_losses

    model.eval()
    totals: dict[str, float] = {"total": 0.0, "mel": 0.0, "mel_voiced": 0.0}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batches += 1
            moved = _move_batch(batch, device)
            outputs = model(moved["features"], durations=moved["durations"],
                            seg_durations=moved["seg_durations"], gaps=moved["gaps"],
                            lead=moved["lead"])
            loss, parts = slot_losses(outputs, moved, weights,
                                      shape_mean=model.shape_mean, shape_std=model.shape_std)
            _gap, mel_voiced = _mel_split(outputs, moved)
            totals["total"] += float(loss.detach())
            totals["mel"] += float(parts["mel"].detach())
            totals["mel_voiced"] += float(mel_voiced)
            if max_batches and batches >= max_batches:
                break
    model.train()
    return {key: value / max(1, batches) for key, value in totals.items()}


def train_slot_v2(
    config: dict,
    train_manifest: str,
    val_manifest: str,
    output: str,
    *,
    warm_start: str | None = None,
    max_steps: int | None = None,
    resume: str | None = None,
    device: str | None = None,
) -> None:
    """Vòng train v2 (xem docstring module); `device` ghi đè `training.device`."""
    from vifasttts.config import validate_config

    validate_config(config, "acoustic")
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from vifasttts.data.dataset import ManifestDataset
    from vifasttts.data.dataset_v2 import SlotDataset, collate_slot_batch
    from vifasttts.device import resolve_device
    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.models.factory import (
        ARCH_V1,
        ARCH_V2,
        arch_of_checkpoint,
        build_model,
        warm_start_v2_from_v1,
    )
    from vifasttts.models.v2_losses import slot_losses

    training: dict[str, Any] = dict(config.get("training", {}))
    set_seed(int(config.get("seed", 42)))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for path in (train_manifest, val_manifest):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    (out / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                              encoding="utf-8")

    stats_path = Path(str(config.get("f0_stats_path", "data/codelung/f0_stats_v2.json")))
    if not stats_path.exists():
        raise FileNotFoundError(f"Thiếu f0_stats_v2: {stats_path} (đặt khóa `f0_stats_path`)")
    f0_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    cache_dir = _resolve_cache_dir(training, out)
    train_data = SlotDataset(train_manifest, config, f0_stats, cache_dir=cache_dir, min_syllables=2)
    val_data = SlotDataset(val_manifest, config, f0_stats, cache_dir=cache_dir)
    batch_size = int(training.get("batch_size", 1))
    if batch_size != 1:
        print(f"[v2] batch_size={batch_size}: collate yêu cầu **cùng N** — sẽ raise nếu khác N.")
    dropped = len(ManifestDataset(train_manifest).rows) - len(train_data.rows)
    if dropped:
        print(f"[v2] Lọc {dropped} row 1 âm tiết khỏi train (MPS abort conv backward với N=1 — §25).")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=int(training.get("num_workers", 0)),
                              collate_fn=collate_slot_batch, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=int(training.get("num_workers", 0)),
                            collate_fn=collate_slot_batch, drop_last=False)

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
             len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    model = build_model(config, sizes, arch=ARCH_V2)
    model.load_f0_stats(f0_stats)

    warm_hash: str | None = None
    if warm_start:
        checkpoint = torch.load(warm_start, map_location="cpu", weights_only=False)
        source_arch = arch_of_checkpoint(checkpoint if isinstance(checkpoint, dict) else {})
        if source_arch != ARCH_V1:
            raise ValueError(f"Warm-start cần checkpoint v1 ({ARCH_V1}), nhận {source_arch!r}")
        report = warm_start_v2_from_v1(model, checkpoint)
        warm_hash = _sha16(Path(warm_start).read_bytes())
        print(f"Warm-start {warm_start}: copy {len(report['loaded'])} tensor "
              f"(bỏ {len(report['skipped'])}), hash {warm_hash}")

    resolved_device = torch.device(device or resolve_device(training.get("device", "auto")))
    model.to(resolved_device)
    model.train()

    base_lr = float(training.get("learning_rate", 2e-4))
    phase_a_lr = base_lr * float(training.get("phase_a_lr_scale", 0.25))
    phase_a_steps = int(training.get("phase_a_steps", 2000))
    phase_b_warmup = int(training.get("phase_b_warmup_steps", 2000))
    grad_clip = float(training.get("grad_clip", 1.0))
    log_every = int(training.get("log_every", 50))
    validate_every = int(training.get("validate_every", 1000))
    save_every = int(training.get("save_every", 1000))
    min_lr_ratio = float(training.get("min_lr_ratio", 0.05))
    weights = config.get("loss_weights", {})

    optimizer = AdamW(model.parameters(), lr=phase_a_lr,
                      weight_decay=float(training.get("weight_decay", 1e-4)))
    start_step = 0
    best_val = float("inf")
    if resume:
        state = torch.load(resume, map_location=resolved_device, weights_only=False)
        if arch_of_checkpoint(state) != ARCH_V2:
            raise ValueError(f"Resume cần checkpoint {ARCH_V2}, nhận {arch_of_checkpoint(state)!r}")
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0))
        best_val = float(state.get("best_val", float("inf")))
        print(f"Tiếp tục từ {resume} (bước {start_step}, best_val {best_val:.4f})")

    first_row = json.loads(Path(train_manifest).read_text(encoding="utf-8").splitlines()[0])
    seg_meta = {key: first_row.get(key) for key in ("seg_method", "seg_method_version", "seg_weights_hash")}
    steps = max_steps if max_steps is not None else int(training.get("max_steps", 20000))

    frozen = start_step < phase_a_steps

    def set_frozen(value: bool) -> None:
        for name, parameter in model.named_parameters():
            if name.startswith(FROZEN_PREFIXES):
                parameter.requires_grad = not value

    set_frozen(frozen)

    def lr_at(step: int) -> float:
        if step <= phase_a_steps:
            return phase_a_lr
        if step <= phase_a_steps + phase_b_warmup:
            ratio = (step - phase_a_steps) / max(1, phase_b_warmup)
            return phase_a_lr + (base_lr - phase_a_lr) * ratio
        progress = min(1.0, (step - phase_a_steps - phase_b_warmup)
                       / max(1, steps - phase_a_steps - phase_b_warmup))
        factor = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * factor

    history_path = out / "history.csv"
    append_history = bool(resume) and history_path.exists() and history_path.stat().st_size > 0
    step = start_step
    started = time.perf_counter()
    with history_path.open("a" if append_history else "w", encoding="utf-8", newline="") as history:
        writer = csv.writer(history)
        if not append_history:
            writer.writerow(["step", "phase", "train_total", "train_mel", "mel_gap", "mel_voiced",
                             "val_total", "val_mel_voiced", "lr"])
        while step < steps:
            for batch in train_loader:
                step += 1
                if frozen and step > phase_a_steps:  # mở khóa — giai đoạn B (§11)
                    frozen = False
                    set_frozen(False)
                    print(f"step {step}: mở khóa toàn bộ (hết giai đoạn A)")
                learning_rate = lr_at(step)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                moved = _move_batch(batch, resolved_device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(moved["features"], durations=moved["durations"],
                                seg_durations=moved["seg_durations"], gaps=moved["gaps"],
                                lead=moved["lead"])
                loss, parts = slot_losses(outputs, moved, weights,
                                          shape_mean=model.shape_mean, shape_std=model.shape_std)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                mel_gap = mel_voiced = float("nan")
                if step % log_every == 0:
                    mel_gap_tensor, mel_voiced_tensor = _mel_split(outputs, moved)
                    mel_gap = float(mel_gap_tensor.detach())
                    mel_voiced = float(mel_voiced_tensor.detach())
                    speed = (time.perf_counter() - started) / max(1, step - start_step)
                    memory = (f" mps={torch.mps.driver_allocated_memory() / 2**30:.1f}G"
                              if resolved_device.type == "mps" else "")
                    print(f"step {step} [{'A' if step <= phase_a_steps else 'B'}]: "
                          f"loss={loss.detach().item():.4f} mel={float(parts['mel'].detach()):.4f} "
                          f"G={mel_gap:.4f} tho={mel_voiced:.4f} "
                          f"lr={learning_rate:.2e} ({speed:.3f}s/bước){memory}")
                if resolved_device.type == "mps" and step % 500 == 0:
                    torch.mps.empty_cache()
                if step % validate_every == 0:
                    metrics = _validate(model, val_loader, weights, resolved_device,
                                        max_batches=int(training.get("validate_batches", 0)))
                    if resolved_device.type == "mps":
                        torch.mps.empty_cache()
                    print(f"step {step}: val={metrics['total']:.4f} "
                          f"(mel hữu thanh {metrics['mel_voiced']:.4f})")
                    writer.writerow([step, "A" if step <= phase_a_steps else "B",
                                     float(loss), float(parts["mel"]), float(mel_gap),
                                     float(mel_voiced),
                                     metrics["total"], metrics["mel_voiced"], learning_rate])
                    history.flush()
                    state = {
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": config, "step": step, "best_val": best_val,
                        "mel_hash": _mel_hash(config), "arch": ARCH_V2,
                        "f0_stats": f0_stats, "f0_stats_hash": _f0_stats_hash(f0_stats),
                        "warm_start_from": warm_hash, **seg_meta,
                    }
                    torch.save(state, out / "last.pt")
                    if metrics["mel_voiced"] < best_val:
                        best_val = metrics["mel_voiced"]
                        torch.save(state, out / "best.pt")
                elif step % save_every == 0:
                    torch.save({
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": config, "step": step, "best_val": best_val,
                        "mel_hash": _mel_hash(config), "arch": ARCH_V2,
                        "f0_stats": f0_stats, "f0_stats_hash": _f0_stats_hash(f0_stats),
                        "warm_start_from": warm_hash, **seg_meta,
                    }, out / "last.pt")
                if step >= steps:
                    break
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": config, "step": step, "best_val": best_val,
        "mel_hash": _mel_hash(config), "arch": ARCH_V2,
        "f0_stats": f0_stats, "f0_stats_hash": _f0_stats_hash(f0_stats),
        "warm_start_from": warm_hash, **seg_meta,
    }, out / "last.pt")
    print(f"Đã huấn luyện v2 {step} bước, checkpoint tại {out}/last.pt "
          f"(best mel hữu thanh {best_val:.4f})")

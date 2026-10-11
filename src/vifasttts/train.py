from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _build_loaders(config: dict[str, Any], train_manifest: str, val_manifest: str, training: dict[str, Any]):
    from torch.utils.data import DataLoader

    from vifasttts.data.dataset import TtsDataset, collate_batch

    train_dataset = TtsDataset(train_manifest, config)
    val_dataset = TtsDataset(val_manifest, config)
    if len(train_dataset) == 0:
        raise ValueError("train manifest rỗng")
    if len(val_dataset) == 0:
        raise ValueError("val manifest rỗng")
    batch_size = int(training.get("batch_size", 16))
    num_workers = int(training.get("num_workers", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_batch, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_batch, num_workers=num_workers)
    return train_loader, val_loader


def _to_device(batch: dict[str, Any], device):
    import torch
    moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    moved["tone_ids"] = moved["features"][..., 4]
    return moved


def _run_validation(model, loader, config, device) -> float:
    import torch

    from vifasttts.losses import acoustic_losses

    model.eval()
    total_loss, batches = 0.0, 0
    weights = config.get("loss_weights", {})
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(batch["features"], durations=batch["durations"], input_lengths=batch["syllable_lengths"])
            loss, _ = acoustic_losses(outputs, batch, weights)
            total_loss += float(loss.item())
            batches += 1
    model.train()
    return total_loss / max(1, batches)


def train_reference(
    config: dict,
    train_manifest: str,
    val_manifest: str,
    output: str,
    *,
    max_steps: int | None = None,
) -> None:
    """Vòng huấn luyện acoustic model tham chiếu.

    Yêu cầu manifest có audio_path trỏ tới WAV 24 kHz (trường "durations"
    tùy chọn từ forced alignment). Mel/energy/F0 được trích tại chỗ; nếu
    chưa có alignment, duration chia đều theo frame và được ghi rõ trong
    docs/TRAINING.md — đó là placeholder, không phải chất lượng production.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Cần cài extra 'train' (torch) để huấn luyện") from exc

    from torch.optim import AdamW

    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.losses import acoustic_losses
    from vifasttts.models.vifasttts import ViFastTtsModel

    training: dict[str, Any] = dict(config.get("training", {}))
    set_seed(int(config.get("seed", 42)))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for path in (train_manifest, val_manifest):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    (out / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda), len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    model = ViFastTtsModel(config, sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_loader, val_loader = _build_loaders(config, train_manifest, val_manifest, training)
    optimizer = AdamW(model.parameters(), lr=float(training.get("learning_rate", 2e-4)),
                      weight_decay=float(training.get("weight_decay", 1e-4)))
    steps = max_steps if max_steps is not None else int(training.get("max_steps", 100_000))
    warmup = max(1, int(training.get("warmup_steps", 1000)))
    grad_clip = float(training.get("grad_clip", 1.0))
    log_every = int(training.get("log_every", 50))
    validate_every = int(training.get("validate_every", 2000))
    save_every = int(training.get("save_every", 5000))
    base_lr = float(training.get("learning_rate", 2e-4))
    use_amp = bool(training.get("mixed_precision", False)) and device.type == "cuda"
    weights = config.get("loss_weights", {})

    history_path = out / "history.csv"
    best_val = float("inf")
    step = 0
    with history_path.open("w", encoding="utf-8", newline="") as history:
        writer = csv.writer(history)
        writer.writerow(["step", "train_total", "train_mel", "val_total"])
        while step < steps:
            for batch in train_loader:
                step += 1
                lr = base_lr * min(1.0, step / warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                batch = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    # Teacher forcing cho phép mở rộng: dùng duration thật để
                    # decoder sinh mel đúng độ dài; duration predictor được học
                    # qua log_duration loss (chuẩn FastSpeech2).
                    outputs = model(batch["features"], durations=batch["durations"], input_lengths=batch["syllable_lengths"])
                loss, parts = acoustic_losses(outputs, batch, weights)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                if step % log_every == 0:
                    print(f"step {step}: loss={loss.detach().item():.4f} mel={parts['mel'].detach().item():.4f} lr={lr:.2e}")
                if step % validate_every == 0:
                    val_loss = _run_validation(model, val_loader, config, device)
                    print(f"step {step}: val={val_loss:.4f}")
                    writer.writerow([step, float(loss), float(parts["mel"]), val_loss])
                    history.flush()
                    state = {"model": model.state_dict(), "config": config, "step": step}
                    torch.save(state, out / "last.pt")
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save(state, out / "best.pt")
                elif step % save_every == 0:
                    torch.save({"model": model.state_dict(), "config": config, "step": step}, out / "last.pt")
                if step >= steps:
                    break
    torch.save({"model": model.state_dict(), "config": config, "step": step}, out / "last.pt")
    print(f"Đã huấn luyện {step} bước, checkpoint tại {out}/last.pt")

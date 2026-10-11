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


def _build_loaders(config: dict[str, Any], train_manifest: str, val_manifest: str, training: dict[str, Any], cache_dir: str | None = None):
    from torch.utils.data import DataLoader

    from vifasttts.data.cache import precompute_features
    from vifasttts.data.dataset import TtsDataset, collate_batch

    if cache_dir is not None:
        # Tính trước ở tiến trình chính để tránh tranh chấp ghi cache giữa workers.
        precompute_features(train_manifest, config, cache_dir)
        precompute_features(val_manifest, config, cache_dir)
    train_dataset = TtsDataset(train_manifest, config, cache_dir=cache_dir)
    val_dataset = TtsDataset(val_manifest, config, cache_dir=cache_dir)
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


def _resolve_cache_dir(training: dict[str, Any], out: Path) -> str | None:
    """cache_dir tường minh, hoặc bật `cache_features: true` -> mặc định trong run."""
    cache_dir = training.get("cache_dir")
    if cache_dir:
        return str(cache_dir)
    if training.get("cache_features", False):
        return str(out / "feature_cache")
    return None


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
    from vifasttts.config import validate_config

    validate_config(config, "acoustic")
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

    train_loader, val_loader = _build_loaders(config, train_manifest, val_manifest, training, cache_dir=_resolve_cache_dir(training, out))
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


def train_vocoder_reference(
    config: dict,
    manifest: str,
    output: str,
    *,
    max_steps: int | None = None,
    vocoder_checkpoint: str | None = None,
) -> None:
    """Huấn luyện vocoder tham chiếu: iSTFTNetLite + MPD-lite.

    Loss generator = mel L1 + feature matching + adversarial (tinh thần
    HiFi-GAN). Checkpoint lưu khóa `vocoder` để `synthesize` dùng trực tiếp.
    Chất lượng production cần cấu hình lớn hơn — xem docs/TRAINING.md.
    """
    from vifasttts.config import validate_config

    validate_config(config, "vocoder")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Cần cài extra 'train' (torch) để huấn luyện vocoder") from exc

    from torch.optim import Adam
    from torch.utils.data import DataLoader

    from vifasttts.data.dataset import VocoderDataset, collate_vocoder_batch
    from vifasttts.data.features import extract_mel_energy_batched
    from vifasttts.models.discriminators import (
        MultiPeriodDiscriminatorLite,
        discriminator_loss,
        feature_matching_loss,
        generator_adversarial_loss,
    )
    from vifasttts.models.istftnet import ISTFTNetLite

    vt: dict[str, Any] = dict(config.get("vocoder_training", {}))
    training: dict[str, Any] = dict(config.get("training", {}))
    set_seed(int(config.get("seed", 42)))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if not Path(manifest).exists():
        raise FileNotFoundError(manifest)
    (out / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    mel_config = dict(config["mel"])
    mel_config.setdefault("sample_rate", int(config.get("sample_rate", 24000)))
    # Vocoder độc lập với acoustic model; mặc định 256 khớp `infer.synthesize`.
    hidden = int(vt.get("hidden", 256))
    upsample_ratio = int(vt.get("upsample_ratio", 4))
    generator = ISTFTNetLite(
        n_mels=int(mel_config["n_mels"]), hidden=hidden,
        n_fft=int(mel_config["n_fft"]), hop_length=int(mel_config["hop_length"]),
        upsample_ratio=upsample_ratio,
    )
    if vocoder_checkpoint and Path(vocoder_checkpoint).exists():
        state = torch.load(vocoder_checkpoint, map_location="cpu")
        generator.load_state_dict(state.get("vocoder", state))
    discriminator = MultiPeriodDiscriminatorLite(channels=int(vt.get("discriminator_channels", 16)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    dataset = VocoderDataset(manifest, config, cache_dir=_resolve_cache_dir(training, out))
    loader = DataLoader(
        dataset,
        batch_size=int(vt.get("batch_size", training.get("batch_size", 8))),
        shuffle=True,
        collate_fn=collate_vocoder_batch,
        num_workers=int(vt.get("num_workers", training.get("num_workers", 0))),
    )
    if len(dataset) == 0:
        raise ValueError("manifest rỗng")

    learning_rate = float(vt.get("learning_rate", training.get("learning_rate", 2e-4)))
    opt_g = Adam(generator.parameters(), lr=learning_rate, betas=(0.8, 0.99))
    opt_d = Adam(discriminator.parameters(), lr=learning_rate, betas=(0.8, 0.99))
    mel_weight = float(vt.get("mel_weight", 45.0))
    fm_weight = float(vt.get("feature_matching_weight", 2.0))
    adv_weight = float(vt.get("adversarial_weight", 1.0))
    steps = max_steps if max_steps is not None else int(vt.get("max_steps", 100_000))
    log_every = int(vt.get("log_every", 50))
    save_every = int(vt.get("save_every", 5000))

    history_path = out / "history_vocoder.csv"
    best = float("inf")
    step = 0
    with history_path.open("w", encoding="utf-8", newline="") as history:
        writer = csv.writer(history)
        writer.writerow(["step", "g_total", "g_adv", "g_mel", "d_loss"])
        while step < steps:
            for batch in loader:
                step += 1
                real = batch["wav"].to(device)
                mel_real = batch["mel"].to(device)
                fake = generator(mel_real)
                length = min(real.size(-1), fake.size(-1))
                real_crop, fake_crop = real[..., :length], fake[..., :length]

                opt_d.zero_grad(set_to_none=True)
                real_outputs = discriminator(real_crop)
                fake_outputs = discriminator(fake_crop.detach())
                d_loss = discriminator_loss(real_outputs, fake_outputs)
                d_loss.backward()
                opt_d.step()

                opt_g.zero_grad(set_to_none=True)
                fake_outputs = discriminator(fake_crop)
                g_adv = generator_adversarial_loss(fake_outputs)
                g_fm = feature_matching_loss(real_outputs, fake_outputs)
                mel_fake, _ = extract_mel_energy_batched(fake_crop, mel_config)
                frames = min(mel_fake.size(1), mel_real.size(1))
                g_mel = (mel_fake[:, :frames] - mel_real[:, :frames]).abs().mean()
                g_loss = adv_weight * g_adv + fm_weight * g_fm + mel_weight * g_mel
                g_loss.backward()
                opt_g.step()

                if step % log_every == 0:
                    print(f"step {step}: g={g_loss.detach().item():.4f} adv={g_adv.detach().item():.4f} "
                          f"mel={g_mel.detach().item():.4f} d={d_loss.detach().item():.4f}")
                if step % save_every == 0 or g_loss.detach().item() < best:
                    best = min(best, float(g_loss.detach().item()))
                    state = {"vocoder": generator.state_dict(), "discriminator": discriminator.state_dict(),
                             "config": config, "step": step}
                    torch.save(state, out / "last.pt")
                    writer.writerow([step, float(g_loss.detach().item()), float(g_adv.detach().item()),
                                     float(g_mel.detach().item()), float(d_loss.detach().item())])
                    history.flush()
                if step >= steps:
                    break
    torch.save({"vocoder": generator.state_dict(), "discriminator": discriminator.state_dict(),
                "config": config, "step": step}, out / "last.pt")
    print(f"Đã huấn luyện vocoder {step} bước, checkpoint tại {out}/last.pt")

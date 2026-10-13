"""Huấn luyện: acoustic model (teacher forcing) và vocoder tham chiếu (GAN).

Luồng chi tiết: `docs/MODEL_FLOW.md` §4–§5; cảnh báo alignment: `docs/TRAINING.md`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Cố định seed cho random/numpy/torch (kể cả CUDA) để tái lập thí nghiệm."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _mel_hash(config: dict) -> str:
    """Hash biểu diễn mel (canonical) để ghi vào checkpoint — rev.5 §1a."""
    from vifasttts.data.features import canonical_mel_config, mel_config_hash

    return mel_config_hash(canonical_mel_config(config))


def _build_loaders(config: dict[str, Any], train_manifest: str, val_manifest: str, training: dict[str, Any], cache_dir: str | None = None):
    """Dựng DataLoader train/val; tính trước cache (nếu bật) để tránh tranh chấp worker."""
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
    """Chuyển tensor trong batch lên thiết bị; thêm `tone_ids` cho loss phân loại thanh."""
    import torch
    moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    moved["tone_ids"] = moved["features"][..., 4]
    return moved


def _run_validation(model, loader, config, device, max_batches: int = 0) -> float:
    """Chạy validation không backward; trả loss trung bình (model về lại train()).

    `max_batches > 0` giới hạn số batch (MPS: validation dài làm bộ nhớ Metal phình
    → giới hạn + `empty_cache` định kỳ để không OOM).
    """
    import torch

    from vifasttts.losses import acoustic_losses

    model.eval()
    total_loss, batches = 0.0, 0
    weights = config.get("loss_weights", {})
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(batch["features"], durations=batch["spans"], input_lengths=batch["syllable_lengths"])
            loss, _ = acoustic_losses(outputs, batch, weights)
            total_loss += float(loss.item())
            batches += 1
            if device.type == "mps" and batches % 100 == 0:
                torch.mps.empty_cache()
            if max_batches and batches >= max_batches:
                break
    if device.type == "mps":
        torch.mps.empty_cache()
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


def _learning_rate(step: int, warmup: int, steps: int, base_lr: float,
                   schedule: str = "constant", min_ratio: float = 0.05,
                   decay_steps: int | None = None) -> float:
    """LR theo bước: warmup tuyến tính → hằng số hoặc cosine giảm dần.

    - `constant`: warmup rồi giữ `base_lr` (mặc định, hành vi cũ).
    - `cosine`: sau warmup giảm cos từ `base_lr` xuống `base_lr × min_ratio`
      tại bước `decay_steps` (mặc định = `steps`) — quan trọng khi chạy theo khối:
      `steps` là mốc cuối CỦA KHỐI, còn `decay_steps` là mốc cuối TOÀN CỤC.
    """
    total = int(decay_steps) if decay_steps else int(steps)
    if step <= warmup:
        return base_lr * max(1e-8, step / max(1, warmup))
    if schedule != "cosine":
        return base_lr
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    factor = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * factor


def _sub_discriminator_weights(discriminator: Any, mrd_weight: Any) -> list[float] | None:
    """Trọng số từng sub-discriminator khi train vocoder: MPD 1.0, MRD `mrd_weight`.

    Vocos gốc dùng `mrd_loss_coeff = 0.1`; None (mặc định) giữ hành vi cũ (đều 1.0).
    """
    if mrd_weight is None:
        return None
    mpd = getattr(discriminator, "mpd", None)
    mrd = getattr(discriminator, "mrd", None)
    if mpd is None or mrd is None:
        return None
    return [1.0] * len(mpd.discriminators) + [float(mrd_weight)] * len(mrd.discriminators)


def _pretrained_sha256(vocoder_training: dict[str, Any]) -> str | None:
    """sha256 của checkpoint pretrained (nếu có) — ghi vào checkpoint fine-tune để truy vết.

    Nếu config ghi `pretrained_sha256` thì bắt buộc khớp file (phát hiện tải nhầm/gãy).
    """
    path = vocoder_training.get("pretrained_path")
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.exists():
        return None
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    expected = str(vocoder_training.get("pretrained_sha256", "") or "")
    if expected and expected != digest:
        raise ValueError(
            f"pretrained_sha256 không khớp: config {expected[:12]}… vs file {digest[:12]}… ({candidate})"
        )
    return digest


def _resume_history_state(path: Path) -> tuple[float, int]:
    """Đọc `history.csv` để khôi phục (best_val, số lần val không cải thiện).

    Cần cho early-stop khi chạy theo khối: mỗi khối là tiến trình mới nhưng phải
    "nhớ" val của các khối trước để đếm đúng số lần chững.
    """
    if not path.exists():
        return float("inf"), 0
    best, no_improve = float("inf"), 0
    try:
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError:
        return float("inf"), 0
    for row in rows[1:]:
        if len(row) < 4:
            continue
        try:
            val = float(row[3])
        except ValueError:
            continue
        if val < best:
            best, no_improve = val, 0
        else:
            no_improve += 1
    return best, no_improve


def train_reference(
    config: dict,
    train_manifest: str,
    val_manifest: str,
    output: str,
    *,
    max_steps: int | None = None,
    resume: str | None = None,
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
    from vifasttts.device import resolve_device

    device = torch.device(resolve_device(training.get("device", "auto")))
    model.to(device)
    model.train()

    train_loader, val_loader = _build_loaders(config, train_manifest, val_manifest, training, cache_dir=_resolve_cache_dir(training, out))
    optimizer = AdamW(model.parameters(), lr=float(training.get("learning_rate", 2e-4)),
                      weight_decay=float(training.get("weight_decay", 1e-4)))
    start_step = 0
    resume_state: dict[str, Any] | None = None
    if resume:
        # Tiếp tục từ checkpoint (model + optimizer + bước) — job dài không mất tiến độ.
        resume_state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        start_step = int(resume_state.get("step", 0))
        print(f"Tiếp tục từ {resume} (bước {start_step})")
    steps = max_steps if max_steps is not None else int(training.get("max_steps", 100_000))
    warmup = max(1, int(training.get("warmup_steps", 1000)))
    grad_clip = float(training.get("grad_clip", 1.0))
    log_every = int(training.get("log_every", 50))
    validate_every = int(training.get("validate_every", 2000))
    save_every = int(training.get("save_every", 5000))
    base_lr = float(training.get("learning_rate", 2e-4))
    lr_schedule = str(training.get("lr_schedule", "constant")).lower()
    lr_min_ratio = float(training.get("min_lr_ratio", 0.05))
    # Mốc cuối TOÀN CỤC cho cosine (chạy theo khối thì `steps` chỉ là mốc cuối khối).
    lr_decay_steps = int(training.get("lr_decay_steps", 0)) or None
    # Đo trên Mac mini M6: fp16 autocast trên MPS không nhanh hơn (0,52 vs 0,48 s/bước)
    # → chỉ bật AMP cho CUDA; MPS giữ fp32.
    use_amp = bool(training.get("mixed_precision", False)) and device.type == "cuda"
    weights = config.get("loss_weights", {})

    early_stop_patience = int(training.get("early_stop_patience", 0))

    history_path = out / "history.csv"
    # Resume: ghi tiếp vào history cũ (không xoá lịch sử val của các khối trước).
    append_history = bool(resume) and history_path.exists() and history_path.stat().st_size > 0
    best_val = float("inf")
    no_improve = 0
    if resume:
        if resume_state is not None:
            best_val = float(resume_state.get("best_val", float("inf")))
        hist_best, no_improve = _resume_history_state(history_path)
        best_val = min(best_val, hist_best)
    step = start_step
    started = time.perf_counter()
    early_stopped = False
    with history_path.open("a" if append_history else "w", encoding="utf-8", newline="") as history:
        writer = csv.writer(history)
        if not append_history:
            writer.writerow(["step", "train_total", "train_mel", "val_total"])
        while step < steps:
            for batch in train_loader:
                step += 1
                lr = _learning_rate(step, warmup, steps, base_lr, lr_schedule, lr_min_ratio,
                                    decay_steps=lr_decay_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                batch = _to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    # Teacher forcing cho phép mở rộng: dùng span thật (duration + gap) để
                    # decoder sinh mel đúng độ dài; duration predictor được học trên
                    # phần hữu thanh (batch["durations"]) qua log_duration loss (chuẩn FastSpeech2).
                    outputs = model(batch["features"], durations=batch["spans"], input_lengths=batch["syllable_lengths"])
                loss, parts = acoustic_losses(outputs, batch, weights)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                if step % log_every == 0:
                    speed = (time.perf_counter() - started) / max(1, step - start_step)
                    mps_mem = (f" mps={torch.mps.driver_allocated_memory() / 2**30:.1f}G"
                               if device.type == "mps" else "")
                    print(f"step {step}: loss={loss.detach().item():.4f} mel={parts['mel'].detach().item():.4f} "
                          f"lr={lr:.2e} ({speed:.3f}s/bước){mps_mem}")
                if device.type == "mps" and step % 500 == 0:
                    # MPS giữ cache block theo shape; batch dài ngắn khác nhau làm cache phình
                    # tới hàng chục GB (đo thật: 40GB → chậm/OOM). E2: giãn còn mỗi ~500 bước
                    # (đủ thưa để tái dùng buffer) + sau mỗi lần validation.
                    torch.mps.empty_cache()
                if step % validate_every == 0:
                    val_loss = _run_validation(model, val_loader, config, device,
                                               max_batches=int(training.get("validate_batches", 0)))
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    print(f"step {step}: val={val_loss:.4f}")
                    writer.writerow([step, float(loss), float(parts["mel"]), val_loss])
                    history.flush()
                    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "config": config, "step": step, "best_val": best_val,
                             "mel_hash": _mel_hash(config)}
                    torch.save(state, out / "last.pt")
                    if val_loss < best_val:
                        best_val = val_loss
                        no_improve = 0
                        torch.save(state, out / "best.pt")
                    else:
                        no_improve += 1
                    if early_stop_patience and no_improve >= early_stop_patience:
                        (out / "early_stop.json").write_text(json.dumps({
                            "step": step, "best_val": best_val,
                            "no_improve": no_improve, "patience": early_stop_patience,
                        }, ensure_ascii=False), encoding="utf-8")
                        print(f"Dừng sớm: val không cải thiện {no_improve} mốc liên tiếp "
                              f"(patience={early_stop_patience}), best_val={best_val:.4f}")
                        early_stopped = True
                        break
                elif step % save_every == 0:
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "config": config, "step": step, "best_val": best_val,
                                "mel_hash": _mel_hash(config)}, out / "last.pt")
                if early_stopped or step >= steps:
                    break
            if early_stopped:
                break
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "step": step, "best_val": best_val,
                "mel_hash": _mel_hash(config)}, out / "last.pt")
    print(f"Đã huấn luyện {step} bước, checkpoint tại {out}/last.pt")


def train_vocoder_reference(
    config: dict,
    manifest: str,
    output: str,
    *,
    max_steps: int | None = None,
    vocoder_checkpoint: str | None = None,
    resume: str | None = None,
    seed: int | None = None,
    lr_scale: float = 1.0,
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
        DEFAULT_MRD_RESOLUTIONS,
        MultiPeriodDiscriminatorLite,
        MultiPeriodMultiResolutionDiscriminator,
        discriminator_loss,
        feature_matching_loss,
        generator_adversarial_loss,
        multi_resolution_stft_loss,
    )
    from vifasttts.models.istftnet import build_generator, load_generator_state

    vt: dict[str, Any] = dict(config.get("vocoder_training", {}))
    training: dict[str, Any] = dict(config.get("training", {}))
    discriminator_kind = str(vt.get("discriminator", "lite"))
    pretrained_sha256 = _pretrained_sha256(vt)
    if pretrained_sha256:
        print(f"Checkpoint pretrained sha256: {pretrained_sha256[:12]}…")
    set_seed(int(seed if seed is not None else config.get("seed", 42)))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if not Path(manifest).exists():
        raise FileNotFoundError(manifest)
    (out / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    mel_config = dict(config["mel"])
    mel_config.setdefault("sample_rate", int(config.get("sample_rate", 24000)))
    # Vocoder độc lập với acoustic model; generator chọn theo `vocoder_training.generator`.
    generator = build_generator(vt, mel_config)
    print(f"Generator {type(generator).__name__}: "
          f"{sum(p.numel() for p in generator.parameters()) / 1e6:.2f}M tham số")
    if vocoder_checkpoint and Path(vocoder_checkpoint).exists():
        load_generator_state(generator, vocoder_checkpoint)
    resolutions_raw = vt.get("mrd_resolutions") or DEFAULT_MRD_RESOLUTIONS
    resolutions: tuple[tuple[int, int, int], ...] = tuple(
        (int(item[0]), int(item[1]), int(item[2])) for item in resolutions_raw
    )
    discriminator: torch.nn.Module
    if discriminator_kind in ("mpd_mrd", "full"):
        # Nâng cấp theo REVISION_PLAN_V2 §10: MPD đầy đủ 2/3/5/7/11 + MRD.
        discriminator = MultiPeriodMultiResolutionDiscriminator(
            periods=tuple(int(period) for period in vt.get("mpd_periods", (2, 3, 5, 7, 11))),
            channels=int(vt.get("discriminator_channels", 32)),
            resolutions=resolutions,
        )
    else:
        discriminator = MultiPeriodDiscriminatorLite(channels=int(vt.get("discriminator_channels", 16)))

    from vifasttts.device import resolve_device

    requested = vt.get("device") or training.get("device") or "auto"
    if (str(requested) == "auto" and discriminator_kind in ("mpd_mrd", "full")
            and torch.backends.mps.is_available()):
        # Đo lại sau khi nâng discriminator (MPD đầy đủ 2/3/5/7/11 + MRD): MPS nhanh
        # hơn CPU 3,8–4,7× (REVISION_PLAN_V2 §10) — kết luận "CPU nhanh hơn" đã đảo.
        print("Vocoder MPD+MRD: dùng MPS (nhanh hơn CPU khi discriminator lớn).")
        resolved = "mps"
    else:
        resolved = resolve_device(requested)
        if (resolved == "mps" and str(requested) == "auto"
                and discriminator_kind not in ("mpd_mrd", "full")):
            # Đo thực tế trên Mac Apple Silicon: vocoder discriminator nhỏ chậm hơn CPU.
            print("Ghi chú: MPS chậm hơn CPU cho vocoder discriminator nhỏ → dùng CPU "
                  "(ép bằng vocoder_training.device=mps nếu muốn).")
            resolved = "cpu"
    device = torch.device(resolved)
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
        drop_last=True,  # batch đủ → một shape duy nhất trên MPS (E4)
    )
    if len(dataset) == 0:
        raise ValueError("manifest rỗng")

    learning_rate = float(vt.get("learning_rate", training.get("learning_rate", 2e-4)))
    disc_learning_rate = float(vt.get("discriminator_learning_rate", learning_rate))
    lr_base = learning_rate * float(lr_scale)
    lr_base_d = disc_learning_rate * float(lr_scale)
    lr_schedule = str(vt.get("lr_schedule", "constant")).lower()
    lr_min_ratio = float(vt.get("lr_min_ratio", 0.05))
    lr_decay_steps = int(vt.get("lr_decay_steps", 0)) or None
    lr_warmup = int(vt.get("warmup_steps", 0))
    opt_g = Adam(generator.parameters(), lr=lr_base, betas=(0.8, 0.99))
    opt_d = Adam(discriminator.parameters(), lr=lr_base_d, betas=(0.8, 0.99))
    start_step = 0
    if resume:
        # Tiếp tục vocoder (G + D + optimizer) — xem train_reference.
        state = torch.load(resume, map_location=device, weights_only=False)
        generator.load_state_dict(state["vocoder"])
        discriminator.load_state_dict(state["discriminator"])
        if "opt_g" in state:
            opt_g.load_state_dict(state["opt_g"])
        if "opt_d" in state:
            opt_d.load_state_dict(state["opt_d"])
        start_step = int(state.get("step", 0))
        print(f"Tiếp tục vocoder từ {resume} (bước {start_step})")
    if lr_scale != 1.0:
        print(f"lr_scale={lr_scale}: G {learning_rate:.2e} → {lr_base:.2e} | D {disc_learning_rate:.2e} → {lr_base_d:.2e} "
              "(hạ tiếp nếu NaN lặp lại — E4)")
    mel_weight = float(vt.get("mel_weight", 45.0))
    fm_weight = float(vt.get("feature_matching_weight", 2.0))
    adv_weight = float(vt.get("adversarial_weight", 1.0))
    stft_weight = float(vt.get("stft_weight", 0.0))
    grad_clip = float(vt.get("grad_clip", 5.0))
    adv_mode = str(vt.get("adv_loss", "bce")).lower()
    d_warmup = max(0, int(vt.get("d_warmup_steps", 0)))
    sub_weights = _sub_discriminator_weights(discriminator, vt.get("mrd_weight"))
    # Loss mel của vocoder sống trong không gian của generator (Vocos: log-magnitude
    # thô, khử z nội bộ; hybrid: như config) — xem docs/VOCODER_VOCOS.md §2.
    mel_cfg_loss = generator.loss_mel_config(mel_config) if hasattr(generator, "loss_mel_config") else mel_config
    steps = max_steps if max_steps is not None else int(vt.get("max_steps", 100_000))
    log_every = int(vt.get("log_every", 50))
    save_every = int(vt.get("save_every", 5000))
    save_every = max(10, (save_every // 10) * 10)  # rev.6 §1a: bội số của 10 (chỉ lưu sau kiểm NaN)

    history_path = out / "history_vocoder.csv"
    append_history = bool(resume) and history_path.exists() and history_path.stat().st_size > 0
    best = float("inf")
    step = start_step
    started = time.perf_counter()
    finite_ok = torch.ones((), dtype=torch.bool, device=device)
    window_min = torch.full((), float("inf"), device=device)
    last_clean_loss = float("inf")

    def save_clean(state: dict, marker_loss: float) -> None:
        """Lưu checkpoint 'sạch' kiểu atomic + xoay bản cũ sang prev.pt (rev.6 §1a)."""
        state = {**state, "clean": True, "last_loss": marker_loss}
        tmp_path = out / "last.pt.tmp"
        torch.save(state, tmp_path)
        target_path = out / "last.pt"
        if target_path.exists():
            target_path.replace(out / "prev.pt")
        tmp_path.replace(target_path)
    with history_path.open("a" if append_history else "w", encoding="utf-8", newline="") as history:
        writer = csv.writer(history)
        if not append_history:
            writer.writerow(["step", "g_total", "g_adv", "g_mel", "g_stft", "d_loss"])
        while step < steps:
            for batch in loader:
                step += 1
                train_g = step > d_warmup
                lr_now = _learning_rate(step, lr_warmup, steps, lr_base, lr_schedule, lr_min_ratio,
                                        decay_steps=lr_decay_steps)
                lr_now_d = _learning_rate(step, lr_warmup, steps, lr_base_d, lr_schedule, lr_min_ratio,
                                          decay_steps=lr_decay_steps)
                for group in opt_g.param_groups:
                    group["lr"] = lr_now
                for group in opt_d.param_groups:
                    group["lr"] = lr_now_d
                real = batch["wav"].to(device)
                mel_real = batch["mel"].to(device)
                # Random crop ngắn (chuẩn GAN vocoder): giữ bước train nhanh và ổn định
                # dù clip dài 3–20 giây; crop cắt cả mel lẫn wav theo đúng hop.
                crop_frames = int(vt.get("crop_frames", 0))
                if crop_frames > 0 and mel_real.size(1) > crop_frames:
                    hop = int(mel_config["hop_length"])
                    start = random.randrange(mel_real.size(1) - crop_frames)
                    mel_real = mel_real[:, start:start + crop_frames]
                    real = real[:, start * hop:(start + crop_frames) * hop]
                if train_g:
                    fake = generator(mel_real)
                else:
                    # D warm-up: G đóng băng — sinh fake không grad (đỡ bộ nhớ/thời gian).
                    with torch.no_grad():
                        fake = generator(mel_real)
                length = min(real.size(-1), fake.size(-1))
                real_crop, fake_crop = real[..., :length], fake[..., :length]

                opt_d.zero_grad(set_to_none=True)
                real_outputs = discriminator(real_crop)
                fake_outputs = discriminator(fake_crop.detach())
                d_loss = discriminator_loss(real_outputs, fake_outputs, weights=sub_weights, mode=adv_mode)
                finite_ok = finite_ok & torch.isfinite(d_loss)  # E4: không sync mỗi bước
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                opt_d.step()

                g_loss: torch.Tensor | None = None
                g_adv: torch.Tensor | None = None
                g_fm: torch.Tensor | None = None
                g_mel: torch.Tensor | None = None
                g_stft: torch.Tensor | None = None
                if train_g:
                    opt_g.zero_grad(set_to_none=True)
                    fake_outputs = discriminator(fake_crop)
                    g_adv = generator_adversarial_loss(fake_outputs, weights=sub_weights, mode=adv_mode)
                    g_fm = feature_matching_loss(real_outputs, fake_outputs, weights=sub_weights)
                    # Loss mel sống trong không gian của generator (Vocos: log-magnitude thô,
                    # đã khử z nội bộ). stft/FFT chạy trực tiếp trên MPS được (torch 2.14).
                    mel_target = generator.denormalize(mel_real) if hasattr(generator, "denormalize") else mel_real
                    mel_fake, _ = extract_mel_energy_batched(fake_crop, mel_cfg_loss)
                    frames = min(mel_fake.size(1), mel_target.size(1))
                    g_mel = (mel_fake[:, :frames] - mel_target[:, :frames]).abs().mean()
                    g_stft = torch.zeros((), device=fake_crop.device)
                    if stft_weight > 0:
                        g_stft = multi_resolution_stft_loss(fake_crop, real_crop, resolutions)
                    g_loss = adv_weight * g_adv + fm_weight * g_fm + mel_weight * g_mel + stft_weight * g_stft
                    finite_ok = finite_ok & torch.isfinite(g_loss)
                    window_min = torch.minimum(window_min, g_loss.detach())
                check_passed, window_value = False, float("inf")
                if step % 10 == 0:
                    # E4 + rev.6 §1a: kiểm gộp mỗi 10 bước (một sync). NaN → raise NGAY, không ghi gì;
                    # checkpoint chỉ được ghi tại bước vừa kiểm sạch.
                    window_value = float(window_min.item())
                    if not bool(finite_ok.item()):
                        raise RuntimeError(
                            f"NaN/Inf trong 10 bước gần nhất (bước {step}) — runner sẽ resume "
                            "checkpoint sạch gần nhất + đổi seed/hạ LR nếu lặp"
                        )
                    check_passed = True
                    last_clean_loss = window_value
                    finite_ok = torch.ones((), dtype=torch.bool, device=device)
                    window_min = torch.full((), float("inf"), device=device)
                if train_g:
                    assert g_loss is not None  # chỉ None khi đang ở D warm-up
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
                    opt_g.step()

                if step % log_every == 0:
                    speed = (time.perf_counter() - started) / max(1, step - start_step)
                    g_bits = ""
                    if g_loss is not None and g_adv is not None and g_mel is not None and g_stft is not None:
                        g_bits = (f"g={g_loss.detach().item():.4f} adv={g_adv.detach().item():.4f} "
                                  f"mel={g_mel.detach().item():.4f} stft={g_stft.detach().item():.4f} ")
                    print(f"step {step}: {g_bits}d={d_loss.detach().item():.4f} "
                          f"lr={lr_now:.2e}/{lr_now_d:.2e} ({speed:.3f}s/bước)")
                if device.type == "mps" and step % 500 == 0:
                    torch.mps.empty_cache()  # E2: thưa hơn để tái dùng buffer
                if (check_passed and window_value < float("inf")
                        and (step % save_every == 0 or window_value < best)):
                    best = min(best, window_value)
                    save_clean({"vocoder": generator.state_dict(), "discriminator": discriminator.state_dict(),
                                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                                "config": config, "step": step, "mel_hash": _mel_hash(config),
                                "pretrained_sha256": pretrained_sha256},
                               marker_loss=window_value)
                    writer.writerow([step, window_value,
                                     float(g_adv.detach().item()) if g_adv is not None else 0.0,
                                     float(g_mel.detach().item()) if g_mel is not None else 0.0,
                                     float(g_stft.detach().item()) if g_stft is not None else 0.0,
                                     float(d_loss.detach().item())])
                    history.flush()
                if step >= steps:
                    break
    if device.type == "mps":
        torch.mps.empty_cache()
    if bool(finite_ok.item()):
        finish_loss = last_clean_loss if last_clean_loss < float("inf") else (best if best < float("inf") else 0.0)
        save_clean({"vocoder": generator.state_dict(), "discriminator": discriminator.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "config": config, "step": step, "mel_hash": _mel_hash(config),
                    "pretrained_sha256": pretrained_sha256},
                   marker_loss=finish_loss)
        print(f"Đã huấn luyện vocoder {step} bước, checkpoint sạch tại {out}/last.pt")
    else:
        print(f"Cửa sổ hiện tại có NaN/Inf — KHÔNG lưu cuối, giữ checkpoint sạch gần nhất ({out}/last.pt)")

"""tools/train_vocoder_gta.py — GTA fine-tune Vocos: cặp (**mel MODEL → audio THẬT**).

Bối cảnh (docs/ACOUSTIC_V2.md §32.2 · §33.5 · §37.1): mel do model acoustic sinh **mượt hơn
mel thật 58–64 %**; bản "làm nét bằng lọc cứng" (`unsharp` α=2,5) được tai user xác nhận tốt hơn.
GTA là **bản học** của cùng ý tưởng: fine-tune Vocos trên cặp (mel do acoustic sinh
teacher-forcing → audio thật của chính row) để vocoder học nghịch đảo độ mờ hệ thống.

Phương án B (được phép): vòng train riêng, nhưng **tái dùng nguyên** generator/discriminator/loss
và công thức LR của `vifasttts.train.train_vocoder_reference`; khác duy nhất ở dataset — đọc mel
từ cache GTA (`tools/build_gta_mel.py`) và **cắt audio theo `frame_range`** của row (VocoderDataset
sẵn có lấy mel thật của cả clip nên không dùng được cho GTA). Mọi siêu tham số lấy từ
`vocoder_training` trong config (batch 8 · crop 64 · G lr 5e-5 · D lr 2e-4 · cosine · mel 45 ·
lsgan · MPD+MRD ch16); `--d-warmup` mặc định 300 = 10 % số bước (cùng tỉ lệ 1000/10k của config).

Chạy:
    .venv/bin/python tools/train_vocoder_gta.py --config configs/small_codelung_v2_base.yaml \
        --manifest artifacts/vocoder_gta/train_gta.jsonl --mel-cache cache/gta_base/train \
        --output artifacts/vocoder_gta --steps 3000 --save-at 2000,3000 --device mps
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # cho phép chạy `python tools/train_vocoder_gta.py`
    sys.path.insert(0, str(_ROOT))


def _row_frames(row: dict[str, Any]) -> int:
    """Số frame của row (khớp `build_gta_mel._row_frames`)."""
    frame_range = row.get("frame_range")
    if frame_range:
        return int(frame_range[1]) - int(frame_range[0])
    return int(sum(row.get("durations") or [])) + int(sum(row.get("gaps") or [])) + int(row.get("lead_frames") or 0)


class GtaMelDataset:
    """Mel **của model** (cache GTA) + audio THẬT cắt theo `frame_range` của row."""

    def __init__(self, manifest: str | Path, mel_cache: str | Path, sample_rate: int, hop: int) -> None:
        """Nạp manifest; mỗi row cần `<mel_cache>/<id>.pt` (do `build_gta_mel.py` sinh)."""
        from vifasttts.data.dataset import ManifestDataset

        self.rows = ManifestDataset(manifest).rows
        self.cache = Path(mel_cache)
        self.sample_rate = int(sample_rate)
        self.hop = int(hop)
        self.frames = [_row_frames(row) for row in self.rows]
        missing = [row["id"] for row in self.rows if not (self.cache / f"{row['id']}.pt").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} row thiếu mel GTA (vd {missing[:3]}) trong {self.cache}")

    def __len__(self) -> int:
        """Số row."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """mel [T,100] (model) + wav [T*hop] (audio thật, cắt đúng `frame_range`)."""


        row = self.rows[index]
        mel = torch_load(self.cache / f"{row['id']}.pt")["mel"]
        frames = mel.size(0)
        start = int(row["frame_range"][0]) if row.get("frame_range") else 0
        # M1 (v18.20): CHỈ đọc đúng đoạn row bằng seek — trước đây load_wav nạp CẢ clip cha
        # (tới ~30 phút) cho mỗi mẫu ⇒ 6 worker × clip cha = chục GB ⇒ MPS OOM.
        import soundfile as _sf
        import torch as _torch

        begin = start * self.hop
        with _sf.SoundFile(row["audio_path"]) as _f:
            _f.seek(begin)
            data = _f.read(frames * self.hop, dtype="float32", always_2d=False)
        chunk = _torch.from_numpy(data.copy()) if data.ndim == 1 else _torch.from_numpy(data[:, 0].copy())
        target = frames * self.hop
        if chunk.numel() < target:  # clip ngắn hơn mel (hiếm): pad 0 cho đủ — đúng như collate
            import torch

            chunk = torch.nn.functional.pad(chunk, (0, target - chunk.numel()))
        return {"mel": mel, "wav": chunk[:target]}


def torch_load(path: Path):
    """`torch.load` CPU (tách hàm để dataset gọn + tránh import torch toàn cục)."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


class BucketSampler:
    """Batch theo độ dài giống nhau (giảm pad khi crop) — xáo thứ tự batch mỗi epoch."""

    def __init__(self, frames: list[int], batch_size: int, seed: int = 42) -> None:
        """`frames`: số frame từng row (để sắp xếp); `batch_size` = batch train."""
        order = sorted(range(len(frames)), key=lambda index: frames[index])
        self.batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
        self.batches = [batch for batch in self.batches if len(batch) == batch_size]  # drop_last
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        """Số batch mỗi epoch."""
        return len(self.batches)

    def __iter__(self):
        """Yield batch (list index) sau khi xáo thứ tự — tái lập theo epoch."""
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.batches)))
        rng.shuffle(order)
        for index in order:
            yield self.batches[index]


def main() -> None:
    """Fine-tune Vocos trên cặp (mel model → audio thật); lưu checkpoint generator-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True, help="Manifest con (row đã có mel GTA)")
    parser.add_argument("--mel-cache", required=True, help="Thư mục mel GTA (<id>.pt)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save-at", default="", help="Danh sách mốc lưu generator-only, vd 2000,3000")
    parser.add_argument("--d-warmup", type=int, default=300, help="Số bước đầu chỉ train D")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr-decay-steps", type=int, default=0,
                        help="Mốc cosine TOÀN CỤC (0 = lấy config lr_decay_steps; nên = --steps cho run ngắn)")
    parser.add_argument("--resume", default=None, help="Checkpoint đầy đủ để resume (last_full.pt)")
    args = parser.parse_args()

    import torch
    import yaml
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    from vifasttts.data.dataset import collate_vocoder_batch
    from vifasttts.data.features import extract_mel_energy_batched
    from vifasttts.models.discriminators import (
        DEFAULT_MRD_RESOLUTIONS,
        MultiPeriodMultiResolutionDiscriminator,
        discriminator_loss,
        feature_matching_loss,
        generator_adversarial_loss,
        multi_resolution_stft_loss,
    )
    from vifasttts.models.istftnet import build_generator, load_generator_state
    from vifasttts.train import (
        _learning_rate,
        _mel_hash,
        _pretrained_sha256,
        _sub_discriminator_weights,
        set_seed,
    )

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    vt: dict[str, Any] = dict(config.get("vocoder_training", {}))
    set_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    pretrained_sha256 = _pretrained_sha256(vt)
    mel_config = dict(config["mel"])
    mel_config.setdefault("sample_rate", int(config.get("sample_rate", 24000)))
    generator = build_generator(vt, mel_config)
    load_generator_state(generator, str(vt["pretrained_path"]))  # G khởi đầu = Vocos pretrained
    print(f"G {sum(p.numel() for p in generator.parameters()) / 1e6:.2f}M · pretrained {vt['pretrained_path']} "
          f"(sha {str(pretrained_sha256)[:12]}…)")
    resolutions = tuple((int(item[0]), int(item[1]), int(item[2]))
                        for item in (vt.get("mrd_resolutions") or DEFAULT_MRD_RESOLUTIONS))
    discriminator = MultiPeriodMultiResolutionDiscriminator(
        periods=tuple(int(period) for period in vt.get("mpd_periods", (2, 3, 5, 7, 11))),
        channels=int(vt.get("discriminator_channels", 16)),
        resolutions=resolutions,
    )
    print(f"D {sum(p.numel() for p in discriminator.parameters()) / 1e6:.2f}M · {device}")
    generator.to(device).train()
    discriminator.to(device).train()

    dataset = GtaMelDataset(args.manifest, args.mel_cache, int(config.get("sample_rate", 24000)),
                            int(mel_config["hop_length"]))
    batch_size = int(vt.get("batch_size", 8))
    sampler = BucketSampler(dataset.frames, batch_size, seed=args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_vocoder_batch,
                        num_workers=int(vt.get("num_workers", 6)))
    hours = sum(dataset.frames) * int(mel_config["hop_length"]) / int(config.get("sample_rate", 24000)) / 3600
    print(f"Dataset: {len(dataset)} row · {sum(dataset.frames)} frame · {len(loader)} batch/epoch "
          f"({hours:.2f} giờ audio)")

    lr_base = float(vt.get("learning_rate", 5e-5)) * args.lr_scale
    lr_base_d = float(vt.get("discriminator_learning_rate", 2e-4)) * args.lr_scale
    opt_g = Adam(generator.parameters(), lr=lr_base, betas=(0.8, 0.99))
    opt_d = Adam(discriminator.parameters(), lr=lr_base_d, betas=(0.8, 0.99))
    start_step = 0
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        generator.load_state_dict(state["vocoder"])
        discriminator.load_state_dict(state["discriminator"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        start_step = int(state.get("step", 0))
        print(f"Resume từ {args.resume} (bước {start_step})")

    mel_weight = float(vt.get("mel_weight", 45.0))
    fm_weight = float(vt.get("feature_matching_weight", 1.0))
    adv_weight = float(vt.get("adversarial_weight", 1.0))
    stft_weight = float(vt.get("stft_weight", 0.0))
    grad_clip = float(vt.get("grad_clip", 5.0))
    adv_mode = str(vt.get("adv_loss", "lsgan")).lower()
    lr_schedule = str(vt.get("lr_schedule", "cosine")).lower()
    lr_min_ratio = float(vt.get("lr_min_ratio", 0.1))
    crop_frames = int(vt.get("crop_frames", 64))
    sub_weights = _sub_discriminator_weights(discriminator, vt.get("mrd_weight"))
    mel_cfg_loss = generator.loss_mel_config(mel_config)
    decay_steps = args.lr_decay_steps or int(vt.get("lr_decay_steps", 0)) or None
    if args.lr_decay_steps:
        print(f"Cosine theo --lr-decay-steps={args.lr_decay_steps} (anneal hết trong run này)")
    elif decay_steps:
        print(f"Cosine theo mốc toàn cục lr_decay_steps={decay_steps} (steps chạy {args.steps})")
    hop = int(mel_config["hop_length"])
    save_at = {int(item) for item in args.save_at.split(",") if item.strip()}

    def save_generator(step: int, metrics: dict[str, float]) -> None:
        """Lưu **generator-only** (`vocoder` + meta) — dùng trực tiếp cho synthesize/eval."""
        path = out / f"gta_{step // 1000}k.pt" if step % 1000 == 0 else out / f"gta_{step}.pt"
        torch.save({"vocoder": generator.state_dict(), "config": config, "step": step,
                    "mel_hash": _mel_hash(config), "pretrained_sha256": pretrained_sha256,
                    "gta": {"mel_cache": args.mel_cache, "manifest": args.manifest,
                            "acoustic_config": args.config, **metrics}},
                   path)
        print(f"  → lưu {path}")

    history_path = out / "history_gta.csv"
    step = start_step
    started = time.perf_counter()
    finite_ok = torch.ones((), dtype=torch.bool, device=device)
    window_min = torch.full((), float("inf"), device=device)
    with history_path.open("w", encoding="utf-8", newline="") as history:
        writer = csv.writer(history)
        writer.writerow(["step", "g_total", "g_adv", "g_mel", "d_loss", "lr_g"])
        while step < args.steps:
            for batch in loader:
                step += 1
                train_g = step > args.d_warmup
                lr_now = _learning_rate(step, 0, args.steps, lr_base, lr_schedule, lr_min_ratio, decay_steps=decay_steps)
                lr_now_d = _learning_rate(step, 0, args.steps, lr_base_d, lr_schedule, lr_min_ratio, decay_steps=decay_steps)
                for group in opt_g.param_groups:
                    group["lr"] = lr_now
                for group in opt_d.param_groups:
                    group["lr"] = lr_now_d
                real = batch["wav"].to(device)
                mel_real = batch["mel"].to(device)  # mel CỦA MODEL (cache GTA)
                if crop_frames > 0 and mel_real.size(1) > crop_frames:
                    start = random.randrange(mel_real.size(1) - crop_frames)  # cùng crop mel+wav (§ train gốc)
                    mel_real = mel_real[:, start:start + crop_frames]
                    real = real[:, start * hop:(start + crop_frames) * hop]
                if train_g:
                    fake = generator(mel_real)
                else:
                    with torch.no_grad():
                        fake = generator(mel_real)
                length = min(real.size(-1), fake.size(-1))
                real_crop, fake_crop = real[..., :length], fake[..., :length]

                opt_d.zero_grad(set_to_none=True)
                real_outputs = discriminator(real_crop)
                fake_outputs = discriminator(fake_crop.detach())
                d_loss = discriminator_loss(real_outputs, fake_outputs, weights=sub_weights, mode=adv_mode)
                finite_ok = finite_ok & torch.isfinite(d_loss)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
                opt_d.step()

                g_loss = g_adv = g_mel = None
                if train_g:
                    opt_g.zero_grad(set_to_none=True)
                    fake_outputs = discriminator(fake_crop)
                    g_adv = generator_adversarial_loss(fake_outputs, weights=sub_weights, mode=adv_mode)
                    g_fm = feature_matching_loss(real_outputs, fake_outputs, weights=sub_weights)
                    # v18.23 SỬA: đích mel loss phải là mel của AUDIO THẬT, không phải mel GTA
                    # (trước đây lấy `denormalize(mel_real)` = chính input ⇒ Vocos học tái tạo
                    # mel P2, tức học luôn rè). Cùng một extractor với mel_fake.
                    mel_target, _ = extract_mel_energy_batched(real_crop, mel_cfg_loss)
                    mel_fake, _ = extract_mel_energy_batched(fake_crop, mel_cfg_loss)
                    frames = min(mel_fake.size(1), mel_target.size(1))
                    g_mel = (mel_fake[:, :frames] - mel_target[:, :frames]).abs().mean()
                    g_loss = adv_weight * g_adv + fm_weight * g_fm + mel_weight * g_mel
                    if stft_weight > 0:
                        g_loss = g_loss + stft_weight * multi_resolution_stft_loss(fake_crop, real_crop, resolutions)
                    finite_ok = finite_ok & torch.isfinite(g_loss)
                    window_min = torch.minimum(window_min, g_loss.detach())
                check_passed = False
                if step % 10 == 0:
                    if torch.backends.mps.is_available() and \
                            torch.mps.driver_allocated_memory() > 30 * 2**30:
                        save_generator(step, {"stop": "mps_mem"})
                        print(f"step {step}: DỪNG AN TOÀN (driver > 30G)")
                        break
                    if not bool(finite_ok.item()):
                        raise RuntimeError(f"NaN/Inf trong 10 bước gần nhất (bước {step})")
                    finite_ok = torch.ones((), dtype=torch.bool, device=device)
                    window_min = torch.full((), float("inf"), device=device)
                    check_passed = True
                if train_g:
                    assert g_loss is not None
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
                    opt_g.step()

                if step % args.log_every == 0:
                    speed = (time.perf_counter() - started) / max(1, step - start_step)
                    mem = (f" mps={torch.mps.driver_allocated_memory() / 2**30:.1f}G"
                           if device.type == "mps" else "")
                    g_bits = (f"g={g_loss.detach().item():.4f} adv={g_adv.detach().item():.4f} "
                              f"mel={g_mel.detach().item():.4f} " if g_loss is not None else "d-warmup ")
                    _drv = (torch.mps.driver_allocated_memory() / 2**30
                            if torch.backends.mps.is_available() else 0.0)
                    print(f"step {step}: mps={_drv:.1f}G {g_bits}d={d_loss.detach().item():.4f} "
                          f"lr={lr_now:.2e} ({speed:.3f}s/bước){mem}", flush=True)
                if device.type == "mps" and step % 500 == 0:
                    torch.mps.empty_cache()
                if check_passed and (step in save_at or step % 1000 == 0 or step == args.steps):
                    save_generator(step, {"g_mel": float(g_mel.detach()) if g_mel is not None else None,
                                          "d_loss": float(d_loss.detach())})
                    writer.writerow([step, float(g_loss.detach()) if g_loss is not None else 0.0,
                                     float(g_adv.detach()) if g_adv is not None else 0.0,
                                     float(g_mel.detach()) if g_mel is not None else 0.0,
                                     float(d_loss.detach()), lr_now])
                    history.flush()
                    torch.save({"vocoder": generator.state_dict(), "discriminator": discriminator.state_dict(),
                                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                                "config": config, "step": step, "mel_hash": _mel_hash(config),
                                "pretrained_sha256": pretrained_sha256}, out / "last_full.pt")
                if step >= args.steps:
                    break
    print(f"Xong {step} bước · {time.perf_counter() - started:.0f}s → {out}")


if __name__ == "__main__":
    main()

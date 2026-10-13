"""Loss acoustic v2 (docs/ACOUSTIC_V2.md §3, §4, §6) — mirror quy ước `losses.acoustic_losses`.

Thành phần: mel L1 (mask frame, chia số bin) · mel_pre (0.2) · duration MSE(log1p) ·
**ratio CE** (soft target, mask inactive + `d_syl < 3` — B5) · **F0 hệ số hình dạng** L1 trong
không gian z-score (mask Addendum 6 × `f0_k_mask` cho hệ số thiếu) · vuv BCE theo segment ·
energy L1 · phonation L1 (mask voiced ≥ 0,5) · tone CE · **pause** gate BCE + log_gap MSE
(chỉ khi gate target = 1).

Trọng số mặc định ở `DEFAULT_WEIGHTS`, config `loss_weights` ghi đè từng key.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vifasttts.losses import masked_l1

DEFAULT_WEIGHTS: dict[str, float] = {
    "mel": 1.0, "mel_pre": 0.2, "mel_sharp": 1.0, "duration": 1.0, "ratio": 1.0,
    "f0": 1.0, "f0_delta": 0.5, "f0_prior": 0.1, "tone_disc": 0.0,
    "vuv": 0.5, "energy": 0.5, "phonation": 0.5,
    "tone": 0.1, "pause_gate": 0.5, "pause_gap": 0.5,
}

#: Nhiệt độ của CE phân biệt thanh (§33.4.3): logits = −d²/T với d = khoảng cách tới prior.
TONE_DISC_TEMPERATURE = 1.0

#: Vị trí chuẩn hoá để đo "độ dốc" contour (độc lập độ dài — B1/§6).
DELTA_POSITIONS = (0.25, 0.5, 0.75)


def shape_deltas(shape: torch.Tensor) -> torch.Tensor:
    """Đạo hàm contour theo vị trí chuẩn hoá tại `DELTA_POSITIONS` → `[.., 3]` (T3.2).

    `df/dpos = −√2·π·Σ_{k≥1} k·s_k·sin(πk·pos)` — dùng cho **delta loss** (khớp *hướng đi* của
    cao độ, không phụ thuộc mức trung bình `s0` hay độ dài segment).
    """
    positions = torch.tensor(DELTA_POSITIONS, device=shape.device, dtype=shape.dtype)
    order = torch.arange(shape.size(-1), device=shape.device, dtype=shape.dtype)
    slope = (-math.sqrt(2.0) * math.pi * order.view(-1, 1)
             * torch.sin(math.pi * order.view(-1, 1) * positions.view(1, -1)))
    return shape @ slope


def structural_active(features: torch.Tensor) -> torch.Tensor:
    """[B,N,7] long → [B,N,4] float {0,1} — cùng quy tắc `ViFastTtsModelV2._active_mask`."""
    onset = (features[..., 0] != 0).to(torch.float32)
    medial = (features[..., 1] != 0).to(torch.float32)
    coda = (features[..., 3] != 0).to(torch.float32)
    return torch.stack([onset, medial, torch.ones_like(onset), coda], dim=-1)


def slot_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict | None = None,
    *,
    shape_mean: torch.Tensor,
    shape_std: torch.Tensor,
    tone_prior: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Trả (loss tổng đã nhân trọng số, dict từng thành phần để log).

    `shape_mean`/`shape_std` là buffer chuẩn hoá của model (từ `f0_stats_v2`) — target `s_k`
    được z-score cùng thang trước khi so. `tone_prior` [6,4] (z-space, `model.tone_prior_z`)
    bật **prior mềm theo thanh** (T3.2).
    """
    merged = {**DEFAULT_WEIGHTS, **(weights or {})}
    losses: dict[str, torch.Tensor] = {}
    active = structural_active(batch["features"])

    losses["mel"] = masked_l1(outputs["mel"], batch["mel"], batch.get("frame_lengths"),
                              mean_over_bins=True)
    if "mel_pre" in outputs:
        losses["mel_pre"] = masked_l1(outputs["mel_pre"], batch["mel"], batch.get("frame_lengths"),
                                      mean_over_bins=True)
    # T4.1 (biến thể ổn định): khớp độ **gồ ghề** của mel (chênh lệch liền kề thời gian + tần số).
    # Đo thật: mel model mượt hơn mel thật 62% (thời gian) / 69% (tần số) — đây là "mờ" cần sửa;
    # GAN (adv) ở batch 1 phân kỳ (xem §28) nên chọn loss tất định này làm đòn chính.
    fake_mel = outputs["mel"]
    real_mel = batch["mel"]
    frames = min(fake_mel.size(1), real_mel.size(1))
    fake_mel, real_mel = fake_mel[:, :frames], real_mel[:, :frames]
    frame_mask = torch.ones(fake_mel.size(0), frames, dtype=torch.bool, device=fake_mel.device)
    if batch.get("frame_lengths") is not None:
        steps = torch.arange(frames, device=fake_mel.device).view(1, -1)
        frame_mask = steps < batch["frame_lengths"].unsqueeze(1)
    pair_mask = (frame_mask[:, 1:] & frame_mask[:, :-1]).to(torch.float32)
    delta_time = ((fake_mel[:, 1:] - fake_mel[:, :-1]) - (real_mel[:, 1:] - real_mel[:, :-1])).abs().mean(dim=-1)
    delta_freq = ((fake_mel[:, :, 1:] - fake_mel[:, :, :-1]) - (real_mel[:, :, 1:] - real_mel[:, :, :-1])).abs().mean(dim=-1)
    mask_float = frame_mask.to(torch.float32)
    losses["mel_sharp"] = (
        (delta_time * pair_mask).sum() / pair_mask.sum().clamp_min(1)
        + (delta_freq * mask_float).sum() / mask_float.sum().clamp_min(1)
    )

    losses["duration"] = F.mse_loss(outputs["log_duration"], torch.log1p(batch["durations"].float()))

    ratio_mask = batch["seg_ratio_mask"]
    # mask slot inactive NGAY TRƯỚC softmax (khớp model) — nếu không, logit slot chết làm lệch
    # log p của slot hợp lệ (đã đo bằng test). CE gộp theo âm tiết (như v1), mask B5.
    masked_logits = outputs["seg_ratio_logits"].masked_fill(active < 0.5, -1e9)
    log_prob = F.log_softmax(masked_logits, dim=-1)
    syllable_valid = ratio_mask.any(dim=-1)
    ce = -(batch["seg_ratio"] * log_prob).sum(dim=-1)  # [B,N]
    losses["ratio"] = (ce * syllable_valid).sum() / syllable_valid.sum().clamp_min(1)

    target_z = (batch["f0_shape"] - shape_mean.view(1, 1, 1, -1)) / shape_std.view(1, 1, 1, -1).clamp_min(1e-6)
    f0_mask = batch["f0_weight"].unsqueeze(-1) * batch["f0_k_mask"].to(torch.float32)
    losses["f0"] = ((outputs["seg_shape"] - target_z).abs() * f0_mask).sum() / f0_mask.sum().clamp_min(1)
    # T3.2 — delta loss: khớp *độ dốc contour* (không phụ thuộc s0/độ dài); mask theo trọng số segment
    delta_mask = batch["f0_weight"].unsqueeze(-1)
    delta_error = (shape_deltas(outputs["seg_shape"]) - shape_deltas(target_z)).abs()
    losses["f0_delta"] = (delta_error * delta_mask).sum() / (delta_mask.sum() * len(DELTA_POSITIONS)).clamp_min(1)
    # T3.2 — prior mềm theo thanh: kéo nucleus về hình dạng trung bình của thanh (z-score)
    if tone_prior is not None:
        tone = batch["features"][..., 4]
        prior = tone_prior[tone]  # [B,N,4]
        nucleus_weight = batch["f0_weight"][..., 2]
        prior_error = ((outputs["seg_shape"][..., 2, :] - prior) ** 2).sum(dim=-1)
        losses["f0_prior"] = (prior_error * nucleus_weight).sum() / nucleus_weight.sum().clamp_min(1)
        # T6.1 (§33.4.3) — CE *phân biệt* thanh từ chính contour dự đoán: logits = −d²/T với
        # d = khoảng cách (bình quân trên hệ số hợp lệ) tới prior của 6 thanh. Khác `f0_prior`
        # (chỉ *kéo về* prior đúng): số hạng này *đẩy ra* khỏi prior sai ⇒ thưởng biên phân biệt
        # (NGÃ vs SẮC/NẶNG — đo §33.3: model sai 1,5–2 nửa cung dù basis chỉ cần 13–65 cent).
        temperature = float(merged.get("tone_disc_temperature", TONE_DISC_TEMPERATURE))
        nucleus = outputs["seg_shape"][..., 2, :]                      # [B,N,4] — chỉ slot N
        kmask_n = batch["f0_k_mask"][..., 2, :].to(torch.float32)      # [B,N,4] hệ số hợp lệ
        distance = (((nucleus.unsqueeze(-2) - tone_prior.view(1, 1, -1, 4)) ** 2)
                    * kmask_n.unsqueeze(-2)).sum(-1) / kmask_n.sum(-1, keepdim=True).clamp_min(1.0)
        disc_mask = (nucleus_weight > 0) & kmask_n.any(dim=-1)
        if disc_mask.any():
            logits = -distance / max(temperature, 1e-3)
            losses["tone_disc"] = F.cross_entropy(logits[disc_mask], tone[disc_mask])
        else:
            losses["tone_disc"] = distance.sum() * 0.0

    losses["vuv"] = (
        F.binary_cross_entropy_with_logits(outputs["seg_vuv_logits"],
                                           (batch["seg_vuv"] >= 0.5).to(torch.float32),
                                           reduction="none") * active
    ).sum() / active.sum().clamp_min(1)
    losses["energy"] = ((outputs["seg_energy"] - batch["seg_energy"]).abs() * active).sum() / active.sum().clamp_min(1)

    if "seg_phonation" in outputs and "seg_phonation" in batch:
        pho_mask = batch["seg_phonation_mask"].unsqueeze(-1).to(torch.float32)
        denominator = pho_mask.sum() * outputs["seg_phonation"].size(-1)
        losses["phonation"] = (
            (outputs["seg_phonation"] - batch["seg_phonation"]).abs() * pho_mask
        ).sum() / denominator.clamp_min(1)

    losses["tone"] = F.cross_entropy(outputs["tone_logits"].transpose(1, 2), batch["features"][..., 4])

    pause_mask = batch["pause_mask"].to(torch.float32)
    losses["pause_gate"] = (
        F.binary_cross_entropy_with_logits(outputs["pause_logits"], batch["pause_gate"],
                                           reduction="none") * pause_mask
    ).sum() / pause_mask.sum().clamp_min(1)
    gap_mask = pause_mask * (batch["pause_gate"] >= 0.5).to(torch.float32)
    losses["pause_gap"] = (
        (outputs["pause_log_gap"] - batch["pause_log_gap"]) ** 2 * gap_mask
    ).sum() / gap_mask.sum().clamp_min(1)

    total = torch.zeros((), device=next(iter(losses.values())).device)
    for key, value in losses.items():
        total = total + value * float(merged.get(key, 1.0))
    return total, losses

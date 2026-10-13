"""Loss acoustic v2 (docs/ACOUSTIC_V2.md §3, §4, §6) — mirror quy ước `losses.acoustic_losses`.

Thành phần: mel L1 (mask frame, chia số bin) · mel_pre (0.2) · duration MSE(log1p) ·
**ratio CE** (soft target, mask inactive + `d_syl < 3` — B5) · **F0 hệ số hình dạng** L1 trong
không gian z-score (mask Addendum 6 × `f0_k_mask` cho hệ số thiếu) · vuv BCE theo segment ·
energy L1 · phonation L1 (mask voiced ≥ 0,5) · tone CE · **pause** gate BCE + log_gap MSE
(chỉ khi gate target = 1).

Trọng số mặc định ở `DEFAULT_WEIGHTS`, config `loss_weights` ghi đè từng key.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vifasttts.losses import masked_l1

DEFAULT_WEIGHTS: dict[str, float] = {
    "mel": 1.0, "mel_pre": 0.2, "duration": 1.0, "ratio": 1.0,
    "f0": 1.0, "vuv": 0.5, "energy": 0.5, "phonation": 0.5,
    "tone": 0.1, "pause_gate": 0.5, "pause_gap": 0.5,
}


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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Trả (loss tổng đã nhân trọng số, dict từng thành phần để log).

    `shape_mean`/`shape_std` là buffer chuẩn hoá của model (từ `f0_stats_v2`) — target `s_k`
    được z-score cùng thang trước khi so.
    """
    merged = {**DEFAULT_WEIGHTS, **(weights or {})}
    losses: dict[str, torch.Tensor] = {}
    active = structural_active(batch["features"])

    losses["mel"] = masked_l1(outputs["mel"], batch["mel"], batch.get("frame_lengths"),
                              mean_over_bins=True)
    if "mel_pre" in outputs:
        losses["mel_pre"] = masked_l1(outputs["mel_pre"], batch["mel"], batch.get("frame_lengths"),
                                      mean_over_bins=True)
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

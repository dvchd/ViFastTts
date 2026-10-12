"""Hàm loss cho acoustic model (masked) và bảng trọng số theo config.

Công thức từng loss: xem `docs/MODEL_FLOW.md` §4.4. Mọi loss đều mask theo
độ dài thật để vùng padding không ảnh hưởng gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor | None = None,
              *, mean_over_bins: bool = False) -> torch.Tensor:
    """L1 có mask theo `lengths` (bỏ qua vùng padding).

    `mean_over_bins=True`: chia thêm cho số bin (chiều cuối) — mặc định False giữ
    hành vi cũ (tổng theo bin), dùng cho mel: tránh loss mel lấn át các loss khác
    (mel 100 bin ≈ 100× các loss còn lại).
    """
    if lengths is None:
        loss = F.l1_loss(pred, target)
    else:
        steps = torch.arange(pred.size(1), device=pred.device)[None, :]
        mask = (steps < lengths[:, None]).unsqueeze(-1)
        loss = (pred.sub(target).abs() * mask).sum() / mask.sum().clamp_min(1)
    if mean_over_bins and pred.size(-1) > 0:
        loss = loss / pred.size(-1)
    return loss


def _syllable_mask(lengths: torch.Tensor | None, size: int, device: torch.device) -> torch.Tensor:
    """Mask [B, size] cho cấp âm tiết: 1 = vị trí thật, 0 = padding."""
    if lengths is None:
        return torch.ones(1, size, device=device)
    steps = torch.arange(size, device=device)[None, :]
    return (steps < lengths[:, None]).to(torch.float32)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Trung bình có mask, tự mở rộng chiều mask cho khớp `values`."""
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def acoustic_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Tổng hợp loss acoustic: mel L1 + duration MSE(log1p) + f0/energy L1 + vuv BCE + tone CE.

    Kèm `phonation` L1 khi batch có target (bật `phonation.enabled`).
    Trả về (loss tổng đã nhân trọng số, dict từng thành phần để log).
    """
    syllable_lengths = batch.get("syllable_lengths")
    syllable_mask = _syllable_mask(syllable_lengths, outputs["log_duration"].size(1), outputs["log_duration"].device)

    duration_target = torch.log1p(batch["durations"].float())
    losses = {
        "mel": masked_l1(outputs["mel"], batch["mel"], batch.get("mel_lengths"), mean_over_bins=True),
        "duration": _masked_mean((outputs["log_duration"] - duration_target) ** 2, syllable_mask),
        "f0": _masked_mean((outputs["f0"] - batch["f0"]).abs(), syllable_mask),
        "vuv": _masked_mean(F.binary_cross_entropy_with_logits(
            outputs["vuv_logits"], batch["vuv"].float(), reduction="none"
        ), syllable_mask),
        "energy": _masked_mean((outputs["energy"] - batch["energy"]).abs(), syllable_mask),
        "tone": _masked_mean(
            F.cross_entropy(outputs["tone_logits"].transpose(1, 2), batch["tone_ids"], reduction="none"),
            syllable_mask,
        ),
    }
    if "phonation" in outputs and "phonation" in batch:
        losses["phonation"] = _masked_mean((outputs["phonation"] - batch["phonation"]).abs(), syllable_mask)
    total = torch.zeros((), device=next(iter(losses.values())).device)
    for key, value in losses.items():
        total = total + value * float(weights.get(key, 1.0))
    return total, losses

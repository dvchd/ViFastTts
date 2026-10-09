from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l1(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
    if lengths is None:
        return F.l1_loss(pred, target)
    steps = torch.arange(pred.size(1), device=pred.device)[None, :]
    mask = (steps < lengths[:, None]).unsqueeze(-1)
    return (pred.sub(target).abs() * mask).sum() / mask.sum().clamp_min(1)


def acoustic_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses = {
        "mel": masked_l1(outputs["mel"], batch["mel"], batch.get("mel_lengths")),
        "duration": F.mse_loss(outputs["log_duration"], torch.log1p(batch["durations"].float())),
        "f0": F.l1_loss(outputs["f0"], batch["f0"]),
        "vuv": F.binary_cross_entropy_with_logits(outputs["vuv_logits"], batch["vuv"].float()),
        "energy": F.l1_loss(outputs["energy"], batch["energy"]),
        "tone": F.cross_entropy(outputs["tone_logits"].transpose(1, 2), batch["tone_ids"]),
    }
    total = sum(losses[k] * float(weights.get(k, 1.0)) for k in losses)
    return total, losses

"""Discriminator nhẹ (MPD-lite) và hàm loss cho huấn luyện vocoder.

Bản tham chiếu theo tinh thần HiFi-GAN: nhiều period discriminator với
kernel/stride khác nhau bắt cấu trúc chu kỳ của waveform. Đây là bản *lite*
để smoke test và A/B; chất lượng production cần cấu hình lớn hơn (xem
docs/TRAINING.md giai đoạn 1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PeriodDiscriminatorLite(nn.Module):
    """Một period discriminator: conv (5,1) stride (3,1), 3 tầng + head."""

    def __init__(self, period: int = 2, channels: int = 16) -> None:
        """`period`: chu kỳ reshape waveform; `channels`: số kênh tầng đầu."""
        super().__init__()
        self.period = period
        self.blocks = nn.ModuleList([
            nn.Conv2d(1, channels, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(channels, channels * 2, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(channels * 2, channels * 4, (5, 1), (3, 1), padding=(2, 0)),
        ])
        self.head = nn.Conv2d(channels * 4, 1, (3, 1), padding=(1, 0))

    def _reshape(self, wav: torch.Tensor) -> torch.Tensor:
        """[B, L] → [B, 1, L//period, period]; pad thêm nếu L không chia hết."""
        batch, length = wav.shape
        if length % self.period:
            wav = F.pad(wav, (0, self.period - length % self.period))
        return wav.view(batch, 1, wav.size(1) // self.period, self.period)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """wav [B, L] → (logits [B], danh sách feature map các tầng conv)."""
        x = self._reshape(wav)
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = F.leaky_relu(block(x), 0.1)
            features.append(x)
        logits = self.head(x).flatten(1).mean(dim=1)  # [B]
        return logits, features


class MultiPeriodDiscriminatorLite(nn.Module):
    """Gộp nhiều period discriminator (mặc định 2/3/5) như HiFi-GAN."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5), channels: int = 16) -> None:
        """Mỗi `period` là một sub-discriminator riêng (2/3/5 như HiFi-GAN)."""
        super().__init__()
        self.discriminators = nn.ModuleList(PeriodDiscriminatorLite(period, channels) for period in periods)

    def forward(self, wav: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        """wav [B, L] → list (logits [B], feature maps) cho từng period."""
        return [discriminator(wav) for discriminator in self.discriminators]


def discriminator_loss(
    real_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
) -> torch.Tensor:
    """BCE: real -> 1, fake -> 0 (tổng qua các period)."""
    total = torch.zeros((), device=real_outputs[0][0].device)
    for (real_logits, _), (fake_logits, _) in zip(real_outputs, fake_outputs):
        total = total + F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        total = total + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return total / max(1, len(real_outputs))


def generator_adversarial_loss(fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]]) -> torch.Tensor:
    """BCE: generator muốn discriminator đoán fake là real."""
    total = torch.zeros((), device=fake_outputs[0][0].device)
    for fake_logits, _ in fake_outputs:
        total = total + F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    return total / max(1, len(fake_outputs))


def feature_matching_loss(
    real_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
) -> torch.Tensor:
    """L1 giữa feature map real (detach) và fake — gradient chảy vào generator."""
    total = torch.zeros((), device=real_outputs[0][0].device)
    count = 0
    for (_, real_features), (_, fake_features) in zip(real_outputs, fake_outputs):
        for real_map, fake_map in zip(real_features, fake_features):
            total = total + F.l1_loss(real_map.detach(), fake_map)
            count += 1
    return total / max(1, count)

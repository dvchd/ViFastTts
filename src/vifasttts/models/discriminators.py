"""Discriminator cho huấn luyện vocoder + hàm loss (MPD-lite, MPD đầy đủ, MRD).

- `MultiPeriodDiscriminatorLite`: bản nhẹ (smoke test, A/B nhanh).
- `MultiPeriodMultiResolutionDiscriminator`: bản nâng cấp theo khuyến nghị
  `docs/REVISION_PLAN_V2.md` §10 — **MPD đầy đủ** (periods 2/3/5/7/11, kênh
  `32/128/512/1024`) + **MRD** (multi-resolution STFT, tham chiếu Vocos/BigVGAN).
- `multi_resolution_stft_loss`: multi-resolution STFT loss (spectral convergence
  + log-magnitude L1) cho generator.
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


class PeriodDiscriminator(nn.Module):
    """MPD đầy đủ kiểu HiFi-GAN: 5 tầng conv (kênh `c/4c/16c/32c/32c`) + head."""

    def __init__(self, period: int = 2, channels: int = 32) -> None:
        """`channels` là kênh tầng đầu; các tầng sau nhân 4/16/32 như HiFi-GAN."""
        super().__init__()
        self.period = period
        c = channels
        self.blocks = nn.ModuleList([
            nn.Conv2d(1, c, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(c, c * 4, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(c * 4, c * 16, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(c * 16, c * 32, (5, 1), (3, 1), padding=(2, 0)),
            nn.Conv2d(c * 32, c * 32, (3, 1), padding=(1, 0)),
        ])
        self.head = nn.Conv2d(c * 32, 1, (3, 1), padding=(1, 0))
        self.activation = nn.LeakyReLU(0.1)

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
            x = self.activation(block(x))
            features.append(x)
        logits = self.head(x).flatten(1).mean(dim=1)
        return logits, features


class SpectralDiscriminator(nn.Module):
    """MRD: một discriminator trên phổ STFT (real/imag = 2 kênh), kiểu Vocos/BigVGAN."""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
                 channels: int = 32) -> None:
        """Một độ phân giải; ghép nhiều cái trong `MultiResolutionDiscriminator`."""
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.window: torch.Tensor
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.blocks = nn.ModuleList([
            nn.Conv2d(2, channels, (3, 9), padding=(1, 4)),
            nn.Conv2d(channels, channels, (3, 9), padding=(1, 4)),
            nn.Conv2d(channels, channels, (3, 9), padding=(1, 4)),
            nn.Conv2d(channels, channels, (3, 9), padding=(1, 4)),
        ])
        self.head = nn.Conv2d(channels, 1, (3, 3), padding=(1, 1))
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """wav [B, L] → (logits [B], feature map các tầng conv trên phổ)."""
        spec = torch.stft(
            wav, self.n_fft, self.hop_length, self.win_length,
            window=self.window.to(wav.device), return_complex=True, center=True,
        )
        x = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, bins, T]
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = self.activation(block(x))
            features.append(x)
        logits = self.head(x).flatten(1).mean(dim=1)
        return logits, features


#: Độ phân giải MRD mặc định (khớp hop 256 @ 24 kHz — như Vocos/BigVGAN).
DEFAULT_MRD_RESOLUTIONS: tuple[tuple[int, int, int], ...] = (
    (1024, 256, 1024), (2048, 512, 2048), (512, 128, 512),
)


class MultiResolutionDiscriminator(nn.Module):
    """Gộp nhiều `SpectralDiscriminator` ở các độ phân giải STFT khác nhau."""

    def __init__(self, resolutions: tuple[tuple[int, int, int], ...] = DEFAULT_MRD_RESOLUTIONS,
                 channels: int = 32) -> None:
        """`resolutions` là các bộ (n_fft, hop, win) — mặc định như Vocos/BigVGAN."""
        super().__init__()
        self.discriminators = nn.ModuleList(
            SpectralDiscriminator(n_fft, hop, win, channels) for n_fft, hop, win in resolutions
        )

    def forward(self, wav: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        """wav [B, L] → list (logits, feature maps) cho từng độ phân giải."""
        return [discriminator(wav) for discriminator in self.discriminators]


class MultiPeriodMultiResolutionDiscriminator(nn.Module):
    """MPD đầy đủ (2/3/5/7/11) + MRD — discriminator cho vocoder chất lượng cao."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11), channels: int = 32,
                 resolutions: tuple[tuple[int, int, int], ...] = DEFAULT_MRD_RESOLUTIONS) -> None:
        """`channels` dùng chung cho MPD (kênh đầu) và MRD."""
        super().__init__()
        self.mpd = MultiPeriodDiscriminatorFull(periods, channels)
        self.mrd = MultiResolutionDiscriminator(resolutions, channels)

    def forward(self, wav: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        """wav [B, L] → list (logits, feature maps) của toàn bộ MPD + MRD."""
        return [*self.mpd(wav), *self.mrd(wav)]


class MultiPeriodDiscriminatorFull(nn.Module):
    """Gộp nhiều `PeriodDiscriminator` đầy đủ (mặc định 2/3/5/7/11 như HiFi-GAN)."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11), channels: int = 32) -> None:
        """Mỗi `period` là một sub-discriminator riêng."""
        super().__init__()
        self.discriminators = nn.ModuleList(PeriodDiscriminator(period, channels) for period in periods)

    def forward(self, wav: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        """wav [B, L] → list (logits, feature maps) cho từng period."""
        return [discriminator(wav) for discriminator in self.discriminators]


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    resolutions: tuple[tuple[int, int, int], ...] = DEFAULT_MRD_RESOLUTIONS,
) -> torch.Tensor:
    """Multi-resolution STFT loss: spectral convergence + log-magnitude L1.

    `pred`/`target` [B, L]; trả trung bình qua các độ phân giải. Target nên detach.
    """
    total = torch.zeros((), device=pred.device)
    for n_fft, hop, win in resolutions:
        window = torch.hann_window(win, device=pred.device, dtype=pred.dtype)
        pred_spec = torch.stft(pred, n_fft, hop, win, window=window, return_complex=True, center=True).abs()
        target_spec = torch.stft(target.detach(), n_fft, hop, win, window=window, return_complex=True, center=True).abs()
        convergence = (pred_spec - target_spec).norm(dim=(-2, -1)) / target_spec.norm(dim=(-2, -1)).clamp_min(1e-6)
        log_l1 = F.l1_loss(torch.log(pred_spec.clamp_min(1e-5)), torch.log(target_spec.clamp_min(1e-5)))
        total = total + convergence.mean() + log_l1
    return total / max(1, len(resolutions))

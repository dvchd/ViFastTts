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


def _sub_losses(
    discriminant: torch.Tensor,
    *,
    real: bool,
    mode: str,
) -> torch.Tensor:
    """Loss của một sub-discriminator: BCE (mặc định) hoặc LSGAN relu (kiểu Vocos).

    - `bce`:  real → 1, fake → 0 (BCE-with-logits).
    - `lsgan`: real → clamp(1 − d, 0), fake → clamp(1 + d, 0) — đúng công thức
      mà checkpoint `charactr/vocos-mel-24khz` được huấn luyện (Vocos dùng loss này).
    """
    if mode == "lsgan":
        target = 1.0 - discriminant if real else 1.0 + discriminant
        return torch.clamp(target, min=0.0).mean()
    target = torch.ones_like(discriminant) if real else torch.zeros_like(discriminant)
    return F.binary_cross_entropy_with_logits(discriminant, target)


def _normalize_weights(weights: list[float] | None, count: int) -> list[float]:
    """Trọng số mỗi sub-discriminator (mặc định 1.0); luôn trả list đủ `count` phần tử."""
    if weights is None:
        return [1.0] * count
    if len(weights) != count:
        raise ValueError(f"weights có {len(weights)} phần tử nhưng có {count} sub-discriminator")
    return [float(weight) for weight in weights]


def discriminator_loss(
    real_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    *,
    weights: list[float] | None = None,
    mode: str = "bce",
) -> torch.Tensor:
    """Loss discriminator: real → 1, fake → 0, trung bình có trọng số qua các sub.

    `weights` cho phép hạ đóng góp của MRD (ví dụ ``[1.0]*5 + [0.1]*3`` như Vocos gốc);
    `mode="lsgan"` dùng relu(1∓d) thay BCE.
    """
    grouped = _normalize_weights(weights, len(real_outputs))
    total = torch.zeros((), device=real_outputs[0][0].device)
    for (real_logits, _), (fake_logits, _), weight in zip(real_outputs, fake_outputs, grouped):
        total = total + weight * (
            _sub_losses(real_logits, real=True, mode=mode)
            + _sub_losses(fake_logits, real=False, mode=mode)
        )
    return total / max(sum(grouped), 1e-8)


def generator_adversarial_loss(
    fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    *,
    weights: list[float] | None = None,
    mode: str = "bce",
) -> torch.Tensor:
    """Loss đối kháng của generator (fake → 1), trung bình có trọng số qua các sub."""
    grouped = _normalize_weights(weights, len(fake_outputs))
    total = torch.zeros((), device=fake_outputs[0][0].device)
    for fake_logits, weight in zip((item[0] for item in fake_outputs), grouped):
        total = total + weight * _sub_losses(fake_logits, real=True, mode=mode)
    return total / max(sum(grouped), 1e-8)


def feature_matching_loss(
    real_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    fake_outputs: list[tuple[torch.Tensor, list[torch.Tensor]]],
    *,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """L1 giữa feature map real (detach) và fake — gradient chảy vào generator."""
    grouped = _normalize_weights(weights, len(real_outputs))
    total = torch.zeros((), device=real_outputs[0][0].device)
    weight_sum = 0.0
    for (_, real_features), (_, fake_features), weight in zip(real_outputs, fake_outputs, grouped):
        for real_map, fake_map in zip(real_features, fake_features):
            total = total + weight * F.l1_loss(real_map.detach(), fake_map)
            weight_sum += weight
    return total / max(weight_sum, 1e-8)


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


class MelScaleDiscriminator(nn.Module):
    """Một thang mel: Conv1d giảm dần theo thời gian (mel là kênh vào) — T4.1.

    Vào `mel [B,T,M]` → `(logits [B,1,T'], features theo từng lớp)` (khớp
    `discriminator_loss`/`generator_adversarial_loss`/`feature_matching_loss`).
    """

    def __init__(self, n_mels: int = 100, channels: int = 64, layers: int = 3,
                 spectral_norm: bool = True) -> None:
        """`channels`: kênh tầng đầu (tăng đôi mỗi tầng, tối đa ×8).

        `spectral_norm=True` (mặc định): chuẩn hoá phổ từng conv — chặn Jacobian của D phình to
        (đo thật: thiếu nó thì nhánh adversarial phân kỳ: adv 165, mel 129 chỉ sau ~300 bước).
        """
        super().__init__()

        def conv(in_channels: int, out_channels: int, *, stride: int) -> nn.Conv1d:
            layer = nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=2)
            return torch.nn.utils.parametrizations.spectral_norm(layer) if spectral_norm else layer

        blocks: list[nn.Module] = []
        current_in = int(n_mels)
        current = int(channels)
        for _ in range(max(1, layers)):
            blocks += [conv(current_in, current, stride=2), nn.LeakyReLU(0.1)]
            current_in = current
            current = min(current * 2, channels * 8)
        self.blocks = nn.ModuleList(blocks)
        self.head = conv(current_in, 1, stride=1)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """mel [B,T,M] (không gian chuẩn hoá) → (logits, feature map mỗi lớp)."""
        x = mel.transpose(1, 2)  # [B, M, T]
        features: list[torch.Tensor] = []
        for layer in self.blocks:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                features.append(x)
        logits = self.head(x)
        return logits, features


class MultiScaleMelDiscriminator(nn.Module):
    """Discriminator mel **đa thang** (T4.1/§8): mỗi thang = average-pool trục thời gian rồi conv.

    Trả `list[(logits, features)]` theo thang — dùng thẳng với các helper loss có sẵn.
    """

    def __init__(self, n_mels: int = 100, scales: tuple[int, ...] = (1, 2, 4),
                 channels: int = 64, layers: int = 3, spectral_norm: bool = True) -> None:
        """`scales`: hệ số gộp thời gian mỗi nhánh (1 = không gộp)."""
        super().__init__()
        self.scales = tuple(int(scale) for scale in scales)
        self.discriminators = nn.ModuleList([
            MelScaleDiscriminator(n_mels, channels, layers, spectral_norm)
            for _ in self.scales
        ])

    def forward(self, mel: torch.Tensor) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        """mel [B,T,M] → list theo thang của (logits, features)."""
        outputs: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
        for scale, discriminator in zip(self.scales, self.discriminators):
            current = mel
            if scale > 1:
                pooled = F.avg_pool1d(mel.transpose(1, 2), kernel_size=scale,
                                      stride=scale, ceil_mode=True)
                current = pooled.transpose(1, 2)
            outputs.append(discriminator(current))
        return outputs

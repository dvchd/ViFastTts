"""iSTFTNet: mel → phổ → waveform — hai generator dùng chung giao diện ONNX/synthesize.

- `ISTFTNetLite` (mặc định cũ): 2 tầng conv + upsample ×ratio — tương thích checkpoint cũ.
- `ISTFTNetVocos`: backbone kiểu Vocos (ConvNeXt blocks, hidden 384 × 8 block) — nâng cấp
  chất lượng theo `docs/ARCH_DISCUSSION_M6.md` (D3), chi phí generator vẫn nhỏ so với
  discriminator MPD+MRD.

Dùng `build_generator(vocoder_training, mel_config)` để chọn đúng lớp theo config.
Xem `docs/MODEL_FLOW.md` §5 và `docs/REVISION_PLAN_V2.md` §10.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ConvNeXtBlock(nn.Module):
    """Block ConvNeXt (kiểu Vocos): dwconv k7 → LN → pwconv ×4 → GELU → pwconv → gamma → residual."""

    def __init__(self, hidden: int, expansion: int = 4, kernel_size: int = 7,
                 layer_scale: float = 1e-2) -> None:
        """`hidden` kênh; `layer_scale` là gamma khởi tạo nhỏ (ổn định lúc đầu)."""
        super().__init__()
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden)
        self.norm = nn.LayerNorm(hidden, eps=1e-6)
        self.pwconv1 = nn.Linear(hidden, expansion * hidden)
        self.activation = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * hidden, hidden)
        self.gamma: torch.Tensor = nn.Parameter(layer_scale * torch.ones(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, C, T]."""
        residual = x
        x = self.norm(self.dwconv(x).transpose(1, 2))
        x = self.pwconv2(self.activation(self.pwconv1(x)))
        return residual + (self.gamma * x).transpose(1, 2)


class _ISTFTNetBase(nn.Module):
    """Phần chung: mel [B, T, M] → (real, imag) phổ → iSTFT.

    Lớp con chỉ cần dựng các module `pre`, `net`, `up`, `post`, `out` với tên giữ nguyên
    (để checkpoint cũ vẫn nạp được).
    """

    n_fft: int
    hop_length: int
    upsample_ratio: int
    pre: nn.Module
    net: nn.Module
    up: nn.Module
    post: nn.Module
    out: nn.Module

    def spectrogram_parts(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mel [B, T, M] -> (real, imag) [B, bins, T*ratio] — dùng cho export ONNX.

        Tách khỏi `spectrogram` để graph không chứa `aten::complex` (ONNX không hỗ trợ).
        """
        z = self.out(self.post(self.up(self.net(self.pre(mel.transpose(1, 2))))))
        real, imag = z.chunk(2, dim=1)
        return real, imag

    def spectrogram(self, mel: torch.Tensor) -> torch.Tensor:
        """Mel [B, T, M] -> phổ complex [B, bins, T*ratio] (layout cho istft)."""
        real, imag = self.spectrogram_parts(mel)
        return torch.complex(real, imag)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Mel [B, T, M] → waveform [B, L] (iSTFT center=True, ghép theo batch)."""
        spec = self.spectrogram(mel)
        window = torch.hann_window(self.n_fft, device=mel.device, dtype=mel.dtype)
        waves = [torch.istft(s, self.n_fft, self.hop_length, self.n_fft, window=window) for s in spec]
        return torch.stack(waves)


class ISTFTNetLite(_ISTFTNetBase):
    """iSTFTNet nhẹ tham chiếu, upsample mel frame lên spectrogram frame.

    mel có hop `hop_length`; khối ConvTranspose mở rộng thêm `upsample_ratio` lần
    trước khi dự đoán STFT (giống tinh thần iSTFTNet: vocoder chỉ sinh phổ, iSTFT
    tổng hợp waveform). Giữ nguyên để tương thích checkpoint cũ.
    """

    def __init__(self, n_mels: int = 100, hidden: int = 256, n_fft: int = 1024,
                 hop_length: int = 256, upsample_ratio: int = 4) -> None:
        """`hop_length` phải chia hết cho `upsample_ratio` (hop trong graph = hop/ratio)."""
        super().__init__()
        if hop_length % upsample_ratio != 0:
            raise ValueError("hop_length phải chia hết cho upsample_ratio")
        self.n_fft = n_fft
        self.hop_length = hop_length // upsample_ratio
        self.upsample_ratio = upsample_ratio
        self.pre = nn.Conv1d(n_mels, hidden, 7, padding=3)
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.LeakyReLU(0.1),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.LeakyReLU(0.1),
        )
        self.up = nn.ConvTranspose1d(
            hidden, hidden,
            kernel_size=2 * upsample_ratio,
            stride=upsample_ratio,
            padding=upsample_ratio // 2,
        )
        self.post = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.LeakyReLU(0.1),
        )
        bins = n_fft // 2 + 1
        self.out = nn.Conv1d(hidden, bins * 2, 7, padding=3)


class ISTFTNetVocos(_ISTFTNetBase):
    """Generator kiểu Vocos: `blocks` ConvNeXt + head iSTFT (giữ export ONNX 2 graph)."""

    def __init__(self, n_mels: int = 100, hidden: int = 384, n_fft: int = 1024,
                 hop_length: int = 256, upsample_ratio: int = 4, blocks: int = 8) -> None:
        """`blocks` = số block ConvNeXt; hidden 384 như khuyến nghị (ARCH_DISCUSSION_M6 D3)."""
        super().__init__()
        if hop_length % upsample_ratio != 0:
            raise ValueError("hop_length phải chia hết cho upsample_ratio")
        self.n_fft = n_fft
        self.hop_length = hop_length // upsample_ratio
        self.upsample_ratio = upsample_ratio
        self.pre = nn.Conv1d(n_mels, hidden, 7, padding=3)
        self.net = nn.Sequential(*[ConvNeXtBlock(hidden) for _ in range(blocks)])
        self.up = nn.ConvTranspose1d(
            hidden, hidden,
            kernel_size=2 * upsample_ratio,
            stride=upsample_ratio,
            padding=upsample_ratio // 2,
        )
        self.post = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.LeakyReLU(0.1),
        )
        bins = n_fft // 2 + 1
        self.out = nn.Conv1d(hidden, bins * 2, 7, padding=3)


def build_generator(vocoder_training: dict[str, Any], mel_config: dict[str, Any]) -> _ISTFTNetBase:
    """Dựng generator theo `vocoder_training.generator`: "lite" (mặc định) hoặc "vocos"."""
    kind = str(vocoder_training.get("generator", "lite"))
    common = {
        "n_mels": int(mel_config["n_mels"]),
        "n_fft": int(mel_config["n_fft"]),
        "hop_length": int(mel_config["hop_length"]),
        "upsample_ratio": int(vocoder_training.get("upsample_ratio", 4)),
    }
    if kind == "vocos":
        return ISTFTNetVocos(
            hidden=int(vocoder_training.get("hidden", 384)),
            blocks=int(vocoder_training.get("generator_blocks", 8)),
            **common,
        )
    return ISTFTNetLite(hidden=int(vocoder_training.get("hidden", 256)), **common)

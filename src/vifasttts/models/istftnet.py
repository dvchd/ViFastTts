from __future__ import annotations

import torch
from torch import nn


class ISTFTNetLite(nn.Module):
    """iSTFTNet nhẹ tham chiếu, upsample mel frame lên spectrogram frame.

    mel có hop `hop_length`; khối ConvTranspose mở rộng thêm
    `upsample_ratio` lần trước khi dự đoán STFT (giống tinh thần iSTFTNet:
    vocoder chỉ sinh phổ, iSTFT tổng hợp waveform). Đây vẫn là bản tham
    chiếu để nghiên cứu; chất lượng production cần discriminator và
    spectral loss theo lộ trình trong docs/TRAINING.md.
    """

    def __init__(self, n_mels: int = 100, hidden: int = 256, n_fft: int = 1024,
                 hop_length: int = 256, upsample_ratio: int = 4) -> None:
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

    def spectrogram(self, mel: torch.Tensor) -> torch.Tensor:
        """Mel [B, T, M] -> phổ complex [B, bins, T*ratio] (layout cho istft)."""
        z = self.out(self.post(self.up(self.net(self.pre(mel.transpose(1, 2))))))
        real, imag = z.chunk(2, dim=1)
        return torch.complex(real, imag)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        spec = self.spectrogram(mel)
        window = torch.hann_window(self.n_fft, device=mel.device, dtype=mel.dtype)
        waves = [torch.istft(s, self.n_fft, self.hop_length, self.n_fft, window=window) for s in spec]
        return torch.stack(waves)

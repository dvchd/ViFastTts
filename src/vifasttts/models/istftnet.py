from __future__ import annotations

import torch
from torch import nn


class ISTFTNetLite(nn.Module):
    """Compact iSTFT-style reference vocoder.

    This is intentionally clear and exportable. Production research should tune
    upsampling ratios, discriminators and spectral losses for the 24 kHz dataset.
    """
    def __init__(self, n_mels: int = 100, hidden: int = 256, n_fft: int = 1024, hop_length: int = 256) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pre = nn.Conv1d(n_mels, hidden, 7, padding=3)
        self.net = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.LeakyReLU(0.1),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.LeakyReLU(0.1),
        )
        bins = n_fft // 2 + 1
        self.out = nn.Conv1d(hidden, bins * 2, 7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, T, M]
        z = self.out(self.net(self.pre(mel.transpose(1, 2))))
        real, imag = z.chunk(2, dim=1)
        spec = torch.complex(real, imag)
        window = torch.hann_window(self.n_fft, device=mel.device, dtype=mel.dtype)
        waves = [torch.istft(s, self.n_fft, self.hop_length, self.n_fft, window=window) for s in spec]
        return torch.stack(waves)

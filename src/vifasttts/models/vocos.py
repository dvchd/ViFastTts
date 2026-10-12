"""Vocos thật (`charactr/vocos-mel-24khz`): ConvNeXt backbone + head iSTFT, KHÔNG upsample.

Khác `ISTFTNetVocos` (hybrid trong `istftnet.py`, upsample ×4, head conv — không nạp
được checkpoint pretrained), ở đây:

- backbone giữ nguyên độ phân giải frame mel (không upsample);
- head là `Linear(dim, n_fft + 2)` dự đoán **log-magnitude + phase** ở hop đúng
  bằng hop mel (256), rồi iSTFT (center=True) ngay trong forward;
- nạp được checkpoint MIT `charactr/vocos-mel-24khz` (13,53M tham số) bằng
  `load_vocos_pretrained` — xem `docs/VOCODER_VOCOS.md`.

Generator nhận mel đúng như pipeline hiện hành (mel z-score per-bin khi config có
`mel.normalize_path`): affine `mel·std + mean` được "bake" thành buffer và thực hiện
trong graph, nên phần còn lại (acoustic, cache, ONNX) không đổi giao diện.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from vifasttts.models.istftnet import MelSpaceGenerator

#: Số kênh trung gian mặc định của checkpoint mel-24khz (expansion 3×).
DEFAULT_DIM = 512
DEFAULT_INTERMEDIATE_DIM = 1536
DEFAULT_LAYERS = 8


class ConvNeXtBlock(nn.Module):
    """Block ConvNeXt của Vocos (dwconv k7 → LN → pwconv ×3 → GELU → pwconv → gamma)."""

    def __init__(self, dim: int = DEFAULT_DIM, intermediate_dim: int = DEFAULT_INTERMEDIATE_DIM,
                 layer_scale_init_value: float | None = None) -> None:
        """`layer_scale_init_value=None` → 1/dim như gốc (chỉ ảnh hưởng khi init ngẫu nhiên)."""
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        scale = 1.0 / dim if layer_scale_init_value is None else float(layer_scale_init_value)
        self.gamma = nn.Parameter(scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, C, T]."""
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class VocosBackbone(nn.Module):
    """Backbone Vocos: embed conv k7 → LN → `num_layers` ConvNeXt → final LN."""

    def __init__(self, input_channels: int = 100, dim: int = DEFAULT_DIM,
                 intermediate_dim: int = DEFAULT_INTERMEDIATE_DIM, num_layers: int = DEFAULT_LAYERS) -> None:
        """`input_channels` = n_mels; giữ độ phân giải thời gian (không downsample)."""
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            ConvNeXtBlock(dim, intermediate_dim, layer_scale_init_value=1.0 / num_layers)
            for _ in range(num_layers)
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, T] → [B, T, dim]."""
        x = self.embed(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = cast(ConvNeXtBlock, block)(x)
        normalized: torch.Tensor = self.final_layer_norm(x.transpose(1, 2))
        return normalized


class ISTFTHead(nn.Module):
    """Head Fourier của Vocos: Linear → (log-mag, phase) → real/imag → iSTFT hop mel."""

    def __init__(self, dim: int = DEFAULT_DIM, n_fft: int = 1024, hop_length: int = 256) -> None:
        """`hop_length` là hop mel (không chia cho upsample — Vocos không upsample)."""
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.window: torch.Tensor
        self.register_buffer("window", torch.hann_window(n_fft))
        self.n_fft = n_fft
        self.hop_length = hop_length

    @property
    def n_bins(self) -> int:
        """Số bin một phía của STFT."""
        return self.n_fft // 2 + 1

    def spectrum_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, T, dim] → (real, imag) [B, bins, T] — giống `safe_log` ngược: mag = exp(...)≤1e2."""
        head = self.out(x).transpose(1, 2)
        magnitude, phase = head.chunk(2, dim=1)
        magnitude = torch.exp(magnitude).clamp(max=1e2)
        return magnitude * torch.cos(phase), magnitude * torch.sin(phase)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, dim] → waveform [B, L] (iSTFT center=True, ghép cả batch)."""
        real, imag = self.spectrum_parts(x)
        spec = torch.complex(real, imag)
        return torch.istft(spec, self.n_fft, self.hop_length, self.n_fft,
                           window=self.window.to(spec.device), center=True)


class VocosGenerator(MelSpaceGenerator):
    """Generator Vocos đầy đủ: mel [B, T, M] → waveform [B, L].

    Khi `stats_path` được truyền (thường lấy từ `mel.normalize_path`), model kỳ vọng
    mel **đã chuẩn hoá z** như phần còn lại của pipeline và tự khử chuẩn hoá bằng
    buffer `mel_mean`/`mel_std` (bake được vào ONNX).
    """

    #: Đặt bởi `load_vocos_pretrained` — dùng để phân biệt "đã có trọng số" vs random.
    pretrained_loaded: bool
    pretrained_path: str

    def __init__(self, n_mels: int = 100, dim: int = DEFAULT_DIM,
                 intermediate_dim: int = DEFAULT_INTERMEDIATE_DIM, num_layers: int = DEFAULT_LAYERS,
                 n_fft: int = 1024, hop_length: int = 256, stats_path: str | None = None) -> None:
        """`stats_path`: JSON mean/std per-bin (chuẩn hoá z toàn cục) — None thì nhận mel thô."""
        super().__init__()
        if hop_length != (n_fft // 4):
            raise ValueError(
                f"Vocos mel-24khz cần hop_length = n_fft//4 (được {hop_length} vs n_fft {n_fft}); "
                "đổi mel.hop_length/n_fft là đổi biểu diễn — xem docs/VOCODER_VOCOS.md"
            )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.upsample_ratio = 1  # không upsample (tương thích giao diện _ISTFTNetBase)
        self.mel_normalized = stats_path is not None
        self.mel_mean: torch.Tensor
        self.mel_std: torch.Tensor
        if self.mel_normalized:
            from vifasttts.data.features import _mel_norm_stats

            mean, std = _mel_norm_stats(str(stats_path))
            if mean.numel() != n_mels:
                raise ValueError(
                    f"mel stats có {mean.numel()} bin nhưng n_mels={n_mels} — tính lại stats "
                    "bằng tools/compute_mel_stats.py (xem docs/VOCODER_VOCOS.md §3)"
                )
            self.register_buffer("mel_mean", mean.view(1, 1, -1))
            self.register_buffer("mel_std", std.view(1, 1, -1))
        # Backbone + head đúng cấu hình checkpoint (dim 512 / 1536 / 8 lớp).
        self.backbone = VocosBackbone(n_mels, dim, intermediate_dim, num_layers)
        self.head = ISTFTHead(dim, n_fft, hop_length)

    def denormalize(self, mel: torch.Tensor) -> torch.Tensor:
        """Mel z-score → mel log-magnitude thô (không gian huấn luyện của Vocos).

        Hỗ trợ cả [T, M] lẫn [B, T, M]; buffer được reshape theo số chiều của mel
        (tránh broadcast nhầm [T, M] × [1, 1, M] → [1, T, M]) và chuyển đúng device.
        """
        if not self.mel_normalized:
            return mel
        shape = (1,) * (mel.dim() - 1) + (self.mel_mean.numel(),)
        mean = self.mel_mean.reshape(shape).to(mel.device)
        std = self.mel_std.reshape(shape).to(mel.device)
        return mel * std + mean

    def loss_mel_config(self, mel_config: dict[str, Any]) -> dict[str, Any]:
        """Config mel cho **loss của vocoder**: log-magnitude thô, KHÔNG chuẩn hoá z.

        Nhờ vậy mel L1 khớp đúng không gian mà checkpoint pretrained được huấn luyện
        (khác với loss của acoustic — vẫn dùng z-score).
        """
        raw = {key: value for key, value in mel_config.items() if key != "normalize_path"}
        raw.setdefault("sample_rate", mel_config.get("sample_rate", 24000))
        return raw

    def spectrogram_parts(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mel [B, T, M] → (real, imag) [B, bins, T] cho export ONNX/runtime iSTFT numpy."""
        features = self.backbone(self.denormalize(mel).transpose(1, 2))
        return self.head.spectrum_parts(features)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Mel [B, T, M] → waveform [B, L]."""
        features: torch.Tensor = self.backbone(self.denormalize(mel).transpose(1, 2))
        waveform: torch.Tensor = self.head(features)
        return waveform


def load_vocos_pretrained(model: VocosGenerator, path: str | Path) -> None:
    """Nạp checkpoint HF (`pytorch_model.bin`) vào `VocosGenerator`.

    File checkpoint chứa cả `feature_extractor.*` (filterbank/cửa sổ của torchaudio) —
    bỏ qua vì mel do pipeline trích; buffer `mel_mean/mel_std` (nếu có) giữ nguyên.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy checkpoint Vocos tại {path}. Tải bằng "
            "`python tools/fetch_vocos_pretrained.py` (xem docs/VOCODER_VOCOS.md §1)"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    remapped = {}
    for key, value in state.items():
        if key.startswith("feature_extractor."):
            continue
        if key == "head.istft.window":
            remapped["head.window"] = value
        elif key.startswith(("backbone.", "head.")):
            remapped[key] = value
    try:
        model.load_state_dict(remapped, strict=False)
    except RuntimeError as exc:  # shape mismatch (dim/số lớp... khác checkpoint)
        raise ValueError(
            f"Checkpoint {path} không khớp kiến trúc Vocos hiện tại ({exc}); "
            "kiểm tra dim/intermediate/num_layers/n_mels (mặc định 512/1536/8/100)"
        ) from exc
    loaded = sum(1 for key in model.state_dict() if key in remapped)
    expected = sum(1 for key in model.state_dict() if not key.startswith(("mel_mean", "mel_std")))
    if loaded != expected:
        missing = [key for key in model.state_dict() if key not in remapped and not key.startswith(("mel_mean", "mel_std"))]
        raise ValueError(f"Checkpoint {path} thiếu {len(missing)} tham số, ví dụ: {missing[:5]}")
    model.pretrained_loaded = True
    model.pretrained_path = str(path)

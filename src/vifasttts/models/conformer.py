from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

MAX_POSITIONS = 4096


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding cố định, cắt theo độ dài thực (export ONNX được)."""

    def __init__(self, hidden: int, max_len: int = MAX_POSITIONS) -> None:
        super().__init__()
        half = (hidden + 1) // 2
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(half, dtype=torch.float) * (-math.log(10000.0) / max(hidden, 1)))
        pe = torch.zeros(max_len, hidden)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].size(1)])
        self.pe: torch.Tensor
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class SelfAttention(nn.Module):
    """Self-attention với additive mask, export ONNX với batch/time động.

    nn.MultiheadAttention + key_padding_mask bị exporter "nướng" độ dài
    vào graph; additive mask viết tay tránh được lỗi đó và giữ nguyên
    hành vi attention.
    """

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden phải chia hết cho heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # padding_mask: bool [B, T], True = vị trí hợp lệ (không phải pad).
        batch, frames, _ = x.shape
        qkv = self.qkv(x).reshape(batch, frames, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        bias = None
        if padding_mask is not None:
            neg = torch.zeros(batch, 1, 1, frames, dtype=x.dtype, device=x.device)
            bias = neg.masked_fill((~padding_mask)[:, None, None, :], torch.finfo(x.dtype).min)
        attended = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, frames, self.hidden)
        result: torch.Tensor = self.out(attended)
        return result


class ConformerLiteBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, ffn: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ffn), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ffn, hidden))
        self.attn_norm = nn.LayerNorm(hidden)
        self.attn = SelfAttention(hidden, heads, dropout)
        self.conv_norm = nn.LayerNorm(hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.ff2 = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ffn), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ffn, hidden))
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(self.attn_norm(x), padding_mask)
        h = self.conv_norm(x).transpose(1, 2)
        x = x + self.pointwise(torch.nn.functional.silu(self.depthwise(h))).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        normalized: torch.Tensor = self.out_norm(x)
        return normalized


class ConformerLite(nn.Module):
    def __init__(self, layers: int, hidden: int, heads: int, ffn: int, kernel: int, dropout: float,
                 max_positions: int = MAX_POSITIONS) -> None:
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(hidden, max_positions)
        self.blocks = nn.ModuleList([ConformerLiteBlock(hidden, heads, ffn, kernel, dropout) for _ in range(layers)])

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos(x)
        for block in self.blocks:
            x = block(x, padding_mask)
        return x

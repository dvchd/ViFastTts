from __future__ import annotations

import torch
from torch import nn


class ConformerLiteBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, ffn: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ffn), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ffn, hidden))
        self.attn_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden)
        self.pointwise = nn.Conv1d(hidden, hidden, 1)
        self.ff2 = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ffn), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ffn, hidden))
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        x = x + self.attn(h, h, h, key_padding_mask=padding_mask, need_weights=False)[0]
        h = self.conv_norm(x).transpose(1, 2)
        x = x + self.pointwise(torch.nn.functional.silu(self.depthwise(h))).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class ConformerLite(nn.Module):
    def __init__(self, layers: int, hidden: int, heads: int, ffn: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ConformerLiteBlock(hidden, heads, ffn, kernel, dropout) for _ in range(layers)])

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, padding_mask)
        return x

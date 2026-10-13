"""Khối dựng model: embedding 7 thành phần và ConvPredictor cho các predictor."""

from __future__ import annotations

import torch
from torch import nn


class ComponentEmbedding(nn.Module):
    """7 bảng embedding (mỗi trường một bảng) → concat → Linear về hidden."""

    def __init__(self, sizes: list[int], component_dim: int, hidden_size: int) -> None:
        """`sizes`: vocab size của 7 trường, theo đúng thứ tự ID trong `ComponentVocabs`."""
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(size, component_dim) for size in sizes])
        self.proj = nn.Linear(component_dim * len(sizes), hidden_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features [B, N, 7] (long) → hidden [B, N, H]."""
        parts = [table(features[..., i]) for i, table in enumerate(self.tables)]
        projected: torch.Tensor = self.proj(torch.cat(parts, dim=-1))
        return projected


class ConvPredictor(nn.Module):
    """Predictor dạng Conv1d + ReLU + LayerNorm, đầu ra Linear → giá trị vô hướng/vector."""

    def __init__(self, hidden_size: int, out_size: int, layers: int = 3, dropout: float = 0.1) -> None:
        """`out_size`: 1 (duration/f0/vuv/energy) hoặc 3 (phonation)."""
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks += [
                nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
                nn.ReLU(), nn.LayerNorm(hidden_size), nn.Dropout(dropout),
            ]
        self.layers = nn.ModuleList(blocks)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, N, H] → [B, N, out_size] (giữ nguyên trục thời gian).

        Lưu ý MPS: backward với N = 1 lỗi Metal kernel (đo thật — xem
        `tests/test_mps_kernel_fallbacks.py`); train v2 lọc row N = 1 ở `SlotDataset`
        (`min_syllables=2`), eval/infer chỉ forward nên không ảnh hưởng.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)
        output: torch.Tensor = self.out(x)
        return output

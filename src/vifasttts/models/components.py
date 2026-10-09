from __future__ import annotations

import torch
from torch import nn


class ComponentEmbedding(nn.Module):
    def __init__(self, sizes: list[int], component_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(size, component_dim) for size in sizes])
        self.proj = nn.Linear(component_dim * len(sizes), hidden_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        parts = [table(features[..., i]) for i, table in enumerate(self.tables)]
        return self.proj(torch.cat(parts, dim=-1))


class ConvPredictor(nn.Module):
    def __init__(self, hidden_size: int, out_size: int, layers: int = 3, dropout: float = 0.1) -> None:
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
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = layer(x)
        return self.out(x)

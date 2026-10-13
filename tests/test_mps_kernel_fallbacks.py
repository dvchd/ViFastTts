"""Regression kernel MPS **đã gặp khi train v2** (đo thật trên Mac mini M6, torch 2.14.1).

1. `SelfAttention` + dropout trên MPS lỗi khi q/k/v **không requires_grad** (giai đoạn A
   đóng băng embedding/encoder) → `conformer.SelfAttention` dùng nhánh attention viết tay
   khi `self.training and dropout > 0 and device == mps`; test dưới kiểm tra ca này.

2. `ConvPredictor` backward với **N = 1** (trục âm tiết, kernel 3) → abort tiến trình
   (`MPSNDArrayConvolution… channel mismatch`). Đã thử vá bằng zero-pad tường minh (kể cả
   pad từng lớp) — **vẫn abort**, nên cách xử lý là **tránh ca đó khi train**:
   `SlotDataset(min_syllables=2)` lọc row 1 âm tiết (val/eval chỉ forward nên không ảnh hưởng).
   Vì vậy **không** đặt ca N=1 + backward vào test (abort sẽ giết cả tiến trình pytest);
   hành vi lọc được kiểm ở `tests/test_dataset_v2.py`.
"""

from __future__ import annotations

import pytest
import torch

from vifasttts.models.conformer import SelfAttention

MPS_AVAILABLE = torch.backends.mps.is_available()


@pytest.mark.skipif(not MPS_AVAILABLE, reason="cần MPS")
def test_self_attention_dropout_khong_grad_mps():
    """q/k/v không requires_grad (đóng băng) vẫn chạy được trên MPS khi training."""
    attention = SelfAttention(16, 2, dropout=0.1).to("mps").train()
    x = torch.randn(1, 10, 16, device="mps")  # không requires_grad
    assert attention(x).shape == (1, 10, 16)
    assert attention(x, torch.ones(1, 10, dtype=torch.bool, device="mps")).shape == (1, 10, 16)


@pytest.mark.skipif(not MPS_AVAILABLE, reason="cần MPS")
def test_self_attention_grad_binh_thuong_mps():
    """Đường SDPA gốc (có grad) vẫn nguyên — chỉ đổi ở ca không grad (MPS)."""
    attention = SelfAttention(16, 2, dropout=0.1).to("mps").train()
    x = torch.randn(1, 10, 16, device="mps", requires_grad=True)
    attention(x).sum().backward()
    assert x.grad is not None

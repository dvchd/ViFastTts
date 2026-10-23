"""Test crop D không lấy padding (v18.37)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

from finetune_adv import sample_crops


def test_crop_voi_lengths_khong_lay_padding():
    real = torch.zeros(2, 100, 4)
    fake = torch.ones(2, 100, 4)
    gap = torch.zeros(2, 100)
    lengths = torch.tensor([100, 34])
    crop_real, crop_fake, windows = sample_crops(
        real, fake, gap, random.Random(42), frames=64, lengths=lengths, return_windows=True,
    )
    assert crop_real.shape == crop_fake.shape == (2, 34, 4)
    for start, end in windows:
        assert end <= 100
    # row ngắn: crop kết thúc đúng frame_length, không vào 66 frame padding.
    assert windows[1][1] == 34


def test_crop_khong_lengths_giu_hanh_vi_cu():
    real = torch.zeros(1, 100, 4)
    fake = torch.zeros(1, 100, 4)
    gap = torch.zeros(1, 100)
    crop, _ = sample_crops(real, fake, gap, random.Random(0), frames=64)
    assert crop.shape == (1, 64, 4)

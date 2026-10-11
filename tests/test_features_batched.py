"""Test extract_mel_energy_batched khớp bản 1 mẫu (bắt lỗi broadcast/einsum)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vifasttts.data.features import extract_mel_energy, extract_mel_energy_batched

CONFIG = {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20,
          "f_min": 0, "f_max": 12000, "sample_rate": 24000}


def test_batched_mel_matches_single():
    torch.manual_seed(0)
    wavs = torch.randn(2, 4096)
    batched_mel, batched_energy = extract_mel_energy_batched(wavs, CONFIG)
    assert batched_mel.shape[0] == 2 and batched_mel.shape[2] == 20
    for index in range(2):
        single_mel, single_energy = extract_mel_energy(wavs[index], CONFIG)
        assert torch.allclose(batched_mel[index], single_mel, atol=1e-5)
        assert torch.allclose(batched_energy[index], single_energy, atol=1e-5)

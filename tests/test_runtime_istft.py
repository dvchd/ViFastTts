"""Test iSTFT numpy khớp torch.istft (center=True, hann)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vifasttts.runtime.istft import istft


def test_istft_matches_torch():
    rng = np.random.default_rng(0)
    waveform = rng.standard_normal(4000).astype(np.float32)
    n_fft, hop = 256, 64
    spec = torch.stft(
        torch.from_numpy(waveform), n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=torch.hann_window(n_fft), return_complex=True, center=True,
    )
    expected = torch.istft(spec, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=torch.hann_window(n_fft), center=True).numpy()
    got = istft(spec.real.numpy(), spec.imag.numpy(), n_fft=n_fft, hop_length=hop)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-4)


def test_istft_length_formula():
    frames = 12
    real = np.zeros((129, frames), dtype=np.float32)
    imag = np.zeros((129, frames), dtype=np.float32)
    output = istft(real, imag, n_fft=256, hop_length=64)
    assert output.shape == ((frames - 1) * 64,)

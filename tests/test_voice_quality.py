"""Test đo thanh hầu hoá (`tools/voice_quality.py`) — cổng R8-1."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyworld")

from tools.voice_quality import compare_pair

SAMPLE_RATE, HOP, N_FFT = 24000, 256, 1024


def test_compare_pair_hai_ban_giong_nhau_delta_bang_khong():
    time_axis = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone = sum((0.3 / k) * np.sin(2 * np.pi * 150.0 * k * time_axis + 0.1 * k) for k in range(1, 8))
    wav = tone.astype(np.float32)
    metrics = compare_pair(wav, wav.copy(), SAMPLE_RATE, HOP, N_FFT)
    assert metrics["voiced_frames"] >= 5
    # Hai bản y hệt nhau → mọi delta phonation phải bằng 0.
    assert metrics["hnr_creak_delta"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["hnr_mean_synth"] == pytest.approx(metrics["hnr_mean_orig"], abs=1e-6)
    assert "perturb_orig" in metrics and "perturb_synth" in metrics


def test_compare_pair_im_lang_khong_co_frame_huu_thanh():
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    metrics = compare_pair(silence, silence, SAMPLE_RATE, HOP, N_FFT)
    assert metrics["frames"] > 0
    assert metrics["voiced_frames"] == 0
    assert "hnr_creak_delta" not in metrics

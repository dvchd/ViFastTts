"""Test validate_config: báo lỗi rõ ràng thay vì KeyError giữa chừng."""
from __future__ import annotations

import pytest

from vifasttts.config import load_config, validate_config

MEL = {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20, "f_min": 0, "f_max": 12000}
MODEL = {
    "component_dim": 8, "hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1,
    "attention_heads": 2, "ffn_size": 32, "conv_kernel_size": 3, "dropout": 0.0,
    "duration_layers": 1, "variance_layers": 1,
}


def test_base_config_is_valid():
    validate_config(load_config("configs/base_24k.yaml"), "acoustic")
    validate_config(load_config("configs/base_24k.yaml"), "vocoder")


def test_missing_keys_listed_with_paths():
    with pytest.raises(ValueError) as excinfo:
        validate_config({"sample_rate": 24000, "mel": {"n_fft": 256}, "model": {}}, "acoustic")
    message = str(excinfo.value)
    assert "mel.win_length" in message
    assert "mel.hop_length" in message
    assert "model.hidden_size" in message


def test_vocoder_kind_does_not_require_model():
    validate_config({"sample_rate": 24000, "mel": MEL}, "vocoder")


def test_acoustic_kind_requires_model():
    with pytest.raises(ValueError):
        validate_config({"sample_rate": 24000, "mel": MEL}, "acoustic")


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        validate_config({"sample_rate": 24000, "mel": MEL, "model": MODEL}, "sai-kind")

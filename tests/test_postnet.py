"""Test postnet mel (F2 — 5 lớp conv residual + loss trên mel trước/sau)."""
from __future__ import annotations

import torch

from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.losses import acoustic_losses
from vifasttts.models.vifasttts import ViFastTtsModel

TINY = {
    "mel": {"n_mels": 20},
    "model": {
        "component_dim": 8, "hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1,
        "attention_heads": 2, "ffn_size": 32, "conv_kernel_size": 3, "dropout": 0.0,
        "duration_layers": 1, "variance_layers": 1,
    },
}
WEIGHTS = {"mel": 1.0, "mel_pre": 0.2, "duration": 1.0, "f0": 1.0, "vuv": 0.5,
           "energy": 0.5, "tone": 0.1}


def _vocab_sizes() -> list[int]:
    vocabs = ComponentVocabs.default()
    return [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus),
            len(vocabs.coda), len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]


def _batch(torch_module_dims: dict[str, int], durations: torch.Tensor, mel_frames: int) -> dict:
    syllable_max = 4
    return {
        "mel": torch.randn(2, mel_frames, 20),
        "mel_lengths": torch.tensor([mel_frames, mel_frames - 2]),
        "durations": durations,
        "syllable_lengths": torch.tensor([syllable_max, syllable_max]),
        "f0": torch.rand(2, syllable_max) * 80 + 120,
        "vuv": torch.ones(2, syllable_max),
        "energy": torch.randn(2, syllable_max),
        "tone_ids": torch.zeros(2, syllable_max, dtype=torch.long),
    }


def _features(sizes: list[int]) -> torch.Tensor:
    """ID ngẫu nhiên hợp lệ theo từng vocab thành phần (tránh out-of-range)."""
    return torch.stack([torch.randint(0, size, (2, 4)) for size in sizes], dim=-1)


def test_postnet_biến_đổi_mel_và_loss_có_mel_pre():
    torch.manual_seed(0)
    config = {**TINY, "model": {**TINY["model"], "use_postnet": True}}
    sizes = _vocab_sizes()
    model = ViFastTtsModel(config, sizes)
    features = _features(sizes)
    durations = torch.tensor([[3, 4, 2, 5], [2, 2, 2, 2]])
    outputs = model(features, durations=durations, input_lengths=torch.tensor([4, 4]))
    assert "mel_pre" in outputs
    assert outputs["mel"].shape == outputs["mel_pre"].shape
    assert not torch.allclose(outputs["mel"], outputs["mel_pre"])

    batch = _batch({}, durations, outputs["mel"].size(1))
    loss, parts = acoustic_losses(outputs, batch, WEIGHTS)
    assert "mel_pre" in parts and "mel" in parts
    assert torch.isfinite(loss)
    loss.backward()


def test_postnet_tắt_thì_không_có_mel_pre():
    torch.manual_seed(0)
    config = {**TINY, "model": {**TINY["model"], "use_postnet": False}}
    sizes = _vocab_sizes()
    model = ViFastTtsModel(config, sizes)
    features = _features(sizes)
    durations = torch.tensor([[3, 4, 2, 5], [2, 2, 2, 2]])
    outputs = model(features, durations=durations, input_lengths=torch.tensor([4, 4]))
    assert "mel_pre" not in outputs


def test_postnet_delta_tham_số():
    """Postnet 5 lớp conv 256 kênh: tiny (20 bin) = 1.035.284; thật (100 bin) = 1.240.164."""
    base = ViFastTtsModel({**TINY, "model": {**TINY["model"], "use_postnet": False}}, _vocab_sizes())
    with_post = ViFastTtsModel({**TINY, "model": {**TINY["model"], "use_postnet": True}}, _vocab_sizes())
    count_base = sum(p.numel() for p in base.parameters())
    count_post = sum(p.numel() for p in with_post.parameters())
    assert count_post - count_base == 1_035_284  # n_mels=20 trong TINY


def test_postnet_export_onnx(tmp_path):
    """ONNX: chỉ xuất mel sau postnet (`mel`), không lộ `mel_pre`."""
    import numpy as np
    import pytest

    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")

    from vifasttts.export import export_onnx

    config = {
        "sample_rate": 24000,
        "language": "vi",
        "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20,
                "f_min": 0, "f_max": 12000},
        "model": {**TINY["model"], "use_postnet": True},
    }
    model = ViFastTtsModel(config, _vocab_sizes())
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "config": config, "step": 2}, checkpoint)
    onnx_path = tmp_path / "model.onnx"
    export_onnx(config, str(checkpoint), str(onnx_path))
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    features = np.zeros((1, 4, 7), dtype=np.int64)
    names = [output.name for output in session.get_outputs()]
    got = dict(zip(names, session.run(None, {"features": features})))
    assert "mel" in got and "mel_pre" not in got
    assert got["mel"].shape[0] == 1 and got["mel"].shape[2] == 20

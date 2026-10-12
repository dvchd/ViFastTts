"""Test Vocos thật: dựng/loss-space, nạp checkpoint HF, export ONNX fp32/fp16 vs torch.

Xem `docs/VOCODER_VOCOS.md`. Số đo nền (checkpoint thật 24 kHz, clip thật):
- ONNX fp32 vs torch: |Δ| phổ ≤ ~3e-4;
- ONNX fp16 vs fp32: |Δ| waveform max ~7e-4 (mean ~4e-5) — ngưỡng test thoáng để bắt tràn số.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vifasttts.models.istftnet import build_generator
from vifasttts.models.vocos import VocosGenerator

SMALL = {"n_mels": 20, "dim": 32, "intermediate_dim": 48, "num_layers": 2, "n_fft": 256, "hop_length": 64}


def _stats(tmp_path: Path, n_mels: int = 20) -> Path:
    """Stats z-score giả (mean/std hằng) để bật đường khử chuẩn hoá."""
    path = tmp_path / "mel_stats.json"
    path.write_text(json.dumps({"mean": [0.5] * n_mels, "std": [2.0] * n_mels}), encoding="utf-8")
    return path


def _mel_config(stats: Path | None = None) -> dict:
    """Config mel tối thiểu đúng không gian Vocos (không cần torchaudio)."""
    config = {
        "n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20,
        "f_min": 0, "f_max": 12000, "sample_rate": 24000, "norm": "none",
        "log_magnitude": True, "log_floor": 1e-7, "log_floor_mode": "clamp",
    }
    if stats is not None:
        config["normalize_path"] = str(stats)
    return config


def test_vocos_generator_shapes_and_istft():
    generator = VocosGenerator(**SMALL)
    mel = torch.randn(2, 8, 20)
    real, imag = generator.spectrogram_parts(mel)
    assert real.shape == (2, SMALL["n_fft"] // 2 + 1, 8)
    wave = generator(mel)
    assert wave.shape == (2, (8 - 1) * SMALL["hop_length"])  # iSTFT center: (T-1)*hop
    spec = torch.complex(real, imag)
    expected = torch.istft(spec, SMALL["n_fft"], SMALL["hop_length"], SMALL["n_fft"],
                           window=generator.head.window, center=True)
    assert torch.allclose(wave, expected, atol=1e-5)


def test_vocos_denormalize_and_loss_config(tmp_path: Path):
    generator = VocosGenerator(**SMALL, stats_path=str(_stats(tmp_path)))
    raw = torch.randn(1, 5, 20) * 3
    mean = torch.full((1, 1, 20), 0.5)
    std = torch.full((1, 1, 20), 2.0)
    assert torch.allclose(generator.denormalize((raw - mean) / std), raw, atol=1e-5)
    # Mel 2D [T, M] phải giữ nguyên shape (không bị broadcast thành [1, T, M]) — bug eval R7.
    flat = torch.randn(7, 20)
    assert generator.denormalize(flat).shape == (7, 20)
    # Buffer khác device/dtype của mel (eval: mel CPU, model MPS) → vẫn phải chạy.
    generator.mel_mean.data = generator.mel_mean.data.to(torch.float64)
    generator.mel_std.data = generator.mel_std.data.to(torch.float64)
    assert generator.denormalize(torch.randn(1, 5, 20)).shape == (1, 5, 20)
    # Loss của vocoder sống ở không gian THÔ (bỏ normalize_path), xem docs/VOCODER_VOCOS.md §2.
    loss_config = generator.loss_mel_config(_mel_config(_stats(tmp_path)))
    assert "normalize_path" not in loss_config
    assert loss_config["log_floor_mode"] == "clamp"


def test_build_generator_vocos_pretrained_loads_hf_naming(tmp_path: Path):
    """Checkpoint HF có `head.istft.window` + `feature_extractor.*` — loader phải map/bỏ qua."""
    reference = VocosGenerator(**SMALL)
    state = {
        ("head.istft.window" if key == "head.window" else key): value
        for key, value in reference.state_dict().items()
    }
    state["feature_extractor.mel_spec.mel_scale.fb"] = torch.rand(129, 20)
    path = tmp_path / "pytorch_model.bin"
    torch.save(state, path)

    config = {"generator": "vocos_pretrained", "hidden": 32, "intermediate_dim": 48,
              "generator_blocks": 2, "pretrained_path": str(path)}
    generator = build_generator(config, _mel_config())
    assert isinstance(generator, VocosGenerator)
    assert getattr(generator, "pretrained_loaded", False) is True
    assert torch.allclose(generator.backbone.embed.weight, reference.backbone.embed.weight)
    assert torch.allclose(generator.head.window, reference.head.window)
    # Thiếu file → lỗi rõ ràng kèm hướng dẫn tải.
    with pytest.raises(FileNotFoundError, match="fetch_vocos_pretrained"):
        build_generator({**config, "pretrained_path": str(tmp_path / "missing.bin")}, _mel_config())


def test_vocoder_onnx_fp32_fp16_matches_torch(tmp_path: Path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    from vifasttts.export import export_vocoder_onnx
    from vifasttts.runtime.istft import istft as numpy_istft

    stats = _stats(tmp_path)
    generator = VocosGenerator(**SMALL, stats_path=str(stats))
    checkpoint = tmp_path / "last.pt"
    torch.save({"vocoder": generator.state_dict(), "step": 0, "clean": True}, checkpoint)
    config = {
        "sample_rate": 24000,
        "mel": _mel_config(stats),
        "vocoder_training": {"generator": "vocos_pretrained", "hidden": 32,
                             "intermediate_dim": 48, "generator_blocks": 2},
    }
    onnx32 = tmp_path / "vocoder32.onnx"
    export_vocoder_onnx(config, str(checkpoint), str(onnx32))

    torch.manual_seed(0)
    mel = torch.randn(1, 12, 20)
    with torch.no_grad():
        real_t, imag_t = generator.spectrogram_parts(mel)
    session = ort.InferenceSession(str(onnx32), providers=["CPUExecutionProvider"])
    real_o, imag_o = session.run(None, {"mel": mel.numpy()})
    assert np.abs(real_o - real_t.numpy()).max() < 2e-3
    assert np.abs(imag_o - imag_t.numpy()).max() < 2e-3
    metadata = dict(session.get_modelmeta().custom_metadata_map or {})
    assert metadata["hop_length"] == "64"          # Vocos: hop iSTFT = hop mel
    assert metadata["mel_hop_length"] == "64"
    assert metadata["upsample_ratio"] == "1"
    assert metadata["generator"] == "vocos_pretrained"

    onnx16 = tmp_path / "vocoder16.onnx"
    try:
        export_vocoder_onnx(config, str(checkpoint), str(onnx16), precision="fp16")
    except RuntimeError as exc:  # pragma: no cover - thiếu converter fp16
        pytest.skip(f"Không chuyển được fp16 trong môi trường này: {exc}")
    session16 = ort.InferenceSession(str(onnx16), providers=["CPUExecutionProvider"])
    real16, imag16 = session16.run(None, {"mel": mel.numpy()})
    assert np.isfinite(real16).all() and np.isfinite(imag16).all()  # kẹp exp ≤ 1e2 giữ trong graph
    wave32 = numpy_istft(real_o[0].T, imag_o[0].T, n_fft=256, hop_length=64)
    wave16 = numpy_istft(real16[0].T, imag16[0].T, n_fft=256, hop_length=64)
    assert np.abs(wave16 - wave32).max() < 5e-3  # đo thực trên checkpoint 24k: ~7e-4

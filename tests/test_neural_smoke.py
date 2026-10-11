"""Smoke test toàn chuỗi neural: features -> dataset -> train -> synth -> export.

Cần extra 'train' (torch) và tùy chọn 'onnx'/'onnxruntime'; toàn bộ test
được bỏ qua nếu thiếu torch.
"""
from __future__ import annotations

import csv
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vifasttts.config import load_config
from vifasttts.data.dataset import TtsDataset, allocate_durations, collate_batch
from vifasttts.data.features import mel_filterbank
from vifasttts.data.manifest import build_manifest
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.infer import synthesize
from vifasttts.losses import acoustic_losses
from vifasttts.models.istftnet import ISTFTNetLite
from vifasttts.models.vifasttts import LengthRegulator, ViFastTtsModel
from vifasttts.train import set_seed, train_reference

TINY_CONFIG = {
    "project": "ViFastTts",
    "seed": 0,
    "language": "vi",
    "sample_rate": 24000,
    "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20, "f_min": 0, "f_max": 12000},
    "model": {
        "component_dim": 8, "hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1,
        "attention_heads": 2, "ffn_size": 32, "conv_kernel_size": 3, "dropout": 0.0,
        "duration_layers": 1, "variance_layers": 1,
    },
    "training": {
        "batch_size": 2, "learning_rate": 1e-3, "weight_decay": 0.0, "max_steps": 2,
        "warmup_steps": 1, "grad_clip": 1.0, "log_every": 1, "validate_every": 2,
        "save_every": 2, "num_workers": 0, "mixed_precision": False,
    },
    "loss_weights": {"mel": 1.0, "duration": 1.0, "f0": 1.0, "vuv": 0.5, "energy": 0.5, "tone": 0.25},
}


def _write_wav(path: Path, seconds: float = 0.4, freq: float = 220.0) -> None:
    import numpy as np

    sample_rate = 24000
    time_axis = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.4 * np.sin(2 * np.pi * freq * time_axis) * (1.0 + 0.5 * np.sin(2 * np.pi * 3.0 * time_axis))
    pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def _vocab_sizes() -> list[int]:
    vocabs = ComponentVocabs.default()
    return [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
            len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]


def test_filterbank_shape():
    bank = mel_filterbank(24000, 256, 20, 0, 12000)
    assert bank.shape == (20, 256 // 2 + 1)
    assert (bank >= 0).all()
    assert bank.sum() > 0


def test_allocate_durations():
    assert sum(allocate_durations(100, 4)) == 100
    assert min(allocate_durations(100, 4)) >= 1
    assert allocate_durations(3, 5) == [1, 1, 1, 0, 0]
    assert allocate_durations(0, 2) == [0, 0]


def test_length_regulator_masks_and_skips_zero():
    regulator = LengthRegulator()
    x = torch.tensor([[[1.0], [2.0], [3.0]]])  # 1 batch, 3 syllables
    durations = torch.tensor([[2, 0, 1]])
    expanded, lengths, mask = regulator(x, durations)
    assert lengths.tolist() == [3]
    assert expanded[0, :, 0].tolist() == [1.0, 1.0, 3.0]
    assert mask.tolist() == [[True, True, True, ]]

    x2 = torch.tensor([[[1.0], [2.0], [3.0]]])
    expanded2, lengths2, mask2 = regulator(x2, torch.tensor([[2, 0, 0]]))
    assert lengths2.tolist() == [2]
    assert mask2.tolist() == [[True, True]]
    # Frame ngoài tổng duration phải được zero.
    assert expanded2.size(1) == 2


def test_model_forward_backward_and_masks():
    set_seed(0)
    sizes = _vocab_sizes()
    model = ViFastTtsModel(TINY_CONFIG, sizes)
    batch_size, syllables = 2, 5
    features = torch.stack([torch.randint(size, (batch_size, syllables)) for size in sizes], dim=-1)
    outputs = model(features, input_lengths=torch.tensor([5, 3]))
    assert outputs["mel"].size(0) == batch_size
    assert outputs["mel"].size(2) == TINY_CONFIG["mel"]["n_mels"]
    assert outputs["durations"].shape == (batch_size, syllables)

    batch = {
        "mel": torch.randn(batch_size, outputs["mel"].size(1), TINY_CONFIG["mel"]["n_mels"]),
        "mel_lengths": outputs["lengths"],
        "durations": outputs["durations"],
        "f0": torch.rand(batch_size, syllables),
        "vuv": torch.randint(0, 2, (batch_size, syllables)),
        "energy": torch.rand(batch_size, syllables),
        "tone_ids": features[..., 4],
        "syllable_lengths": torch.tensor([5, 3]),
    }
    loss, parts = acoustic_losses(outputs, batch, TINY_CONFIG["loss_weights"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in parts.values())


def test_istftnet_shapes():
    vocoder = ISTFTNetLite(n_mels=20, hidden=16, n_fft=256, hop_length=64, upsample_ratio=4)
    mel = torch.randn(1, 12, 20)
    with torch.no_grad():
        spec = vocoder.spectrogram(mel)
        wav = vocoder(mel)
    assert spec.shape == (1, 256 // 2 + 1, 48)
    assert wav.shape == (1, 47 * 16)
    assert torch.isfinite(wav).all()


def test_end_to_end_train_synth_export(tmp_path: Path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "s1.wav", freq=220)
    _write_wav(wav_dir / "s2.wav", freq=300)
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["s1", "Chào bạn hôm nay"])
        writer.writerow(["s2", "Thử hệ thống mới"])
    manifest = tmp_path / "manifest.jsonl"
    build_manifest(str(metadata), str(wav_dir), str(manifest))

    dataset = TtsDataset(str(manifest), TINY_CONFIG)
    item = dataset[0]
    assert item["features"].dim() == 2 and item["mel"].size(1) == 20
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch["features"].size(0) == 2 and batch["syllable_lengths"].tolist()[0] == 4

    out_dir = tmp_path / "run"
    train_reference(TINY_CONFIG, str(manifest), str(manifest), str(out_dir), max_steps=2)
    checkpoint = out_dir / "last.pt"
    assert checkpoint.exists()
    assert (out_dir / "resolved_config.json").exists()

    wav_out = tmp_path / "out.wav"
    samples, sample_rate = synthesize(TINY_CONFIG, str(checkpoint), "Chào bạn", str(wav_out))
    assert wav_out.exists() and samples > 0 and sample_rate == 24000
    with wave.open(str(wav_out), "rb") as fh:
        assert fh.getnframes() == samples

    pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    from vifasttts.export import export_onnx

    onnx_path = tmp_path / "model.onnx"
    export_onnx(TINY_CONFIG, str(checkpoint), str(onnx_path))
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    vocabs = ComponentVocabs.default()
    from vifasttts.frontend.pipeline import parse_text

    features = torch.tensor([[vocabs.encode(x) for x in parse_text("Chào bạn")]], dtype=torch.long)
    results = session.run(None, {"features": features.numpy()})
    names = [output.name for output in session.get_outputs()]
    got = dict(zip(names, results))
    assert got["mel"].shape[0] == 1 and got["mel"].shape[2] == TINY_CONFIG["mel"]["n_mels"]
    assert got["lengths"].tolist()[0] == got["durations"].sum()

    # INT8 động: file chạy được và giữ nguyên hình dạng đầu ra.
    int8_path = tmp_path / "model.int8.onnx"
    export_onnx(TINY_CONFIG, str(checkpoint), str(int8_path), precision="int8")
    assert int8_path.exists()
    session_int8 = onnxruntime.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    got_int8 = dict(zip(
        [output.name for output in session_int8.get_outputs()],
        session_int8.run(None, {"features": features.numpy()}),
    ))
    assert got_int8["mel"].shape == got["mel"].shape

    # FP16: chỉ chạy khi onnxruntime.transformers hỗ trợ chuyển đổi.
    fp16_path = tmp_path / "model.fp16.onnx"
    try:
        export_onnx(TINY_CONFIG, str(checkpoint), str(fp16_path), precision="fp16")
    except RuntimeError as exc:  # pragma: no cover - phụ thuộc build onnxruntime
        pytest.skip(f"fp16 không khả dụng: {exc}")
    assert fp16_path.exists() and not (tmp_path / "model.fp16.fp32.onnx").exists()
    session_fp16 = onnxruntime.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
    got_fp16 = dict(zip(
        [output.name for output in session_fp16.get_outputs()],
        session_fp16.run(None, {"features": features.numpy()}),
    ))
    assert got_fp16["mel"].shape == got["mel"].shape


def test_config_base_loads():
    config = load_config("configs/base_24k.yaml")
    assert config["mel"]["n_mels"] == 100
    assert config["model"]["hidden_size"] == 256


def test_model_phonation_branch():
    config = {**TINY_CONFIG, "model": {**TINY_CONFIG["model"], "use_phonation": True}}
    sizes = _vocab_sizes()
    model = ViFastTtsModel(config, sizes)
    batch_size, syllables = 2, 4
    features = torch.stack([torch.randint(size, (batch_size, syllables)) for size in sizes], dim=-1)
    outputs = model(features, input_lengths=torch.tensor([4, 3]))
    assert outputs["phonation"].shape == (batch_size, syllables, 3)

    batch = {
        "mel": torch.randn(batch_size, outputs["mel"].size(1), TINY_CONFIG["mel"]["n_mels"]),
        "mel_lengths": outputs["lengths"],
        "durations": outputs["durations"],
        "f0": torch.rand(batch_size, syllables),
        "vuv": torch.randint(0, 2, (batch_size, syllables)),
        "energy": torch.rand(batch_size, syllables),
        "phonation": torch.rand(batch_size, syllables, 3),
        "tone_ids": features[..., 4],
        "syllable_lengths": torch.tensor([4, 3]),
    }
    loss, parts = acoustic_losses(outputs, batch, {**TINY_CONFIG["loss_weights"], "phonation": 0.5})
    assert "phonation" in parts
    loss.backward()
    assert torch.isfinite(loss)

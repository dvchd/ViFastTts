"""Test runtime ONNX end-to-end: export acoustic + vocoder → synthesize-onnx."""
from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from typer.testing import CliRunner

from vifasttts.cli import app
from vifasttts.data.manifest import build_manifest
from vifasttts.export import export_onnx, export_vocoder_onnx
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.models.vifasttts import ViFastTtsModel
from vifasttts.runtime.onnx_engine import OnnxAcousticModel, read_metadata, synthesize_onnx
from vifasttts.train import train_reference, train_vocoder_reference

runner = CliRunner()

TINY_CONFIG = {
    "seed": 0,
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
    "vocoder_training": {"batch_size": 2, "hidden": 16, "log_every": 1, "save_every": 1, "discriminator_channels": 8},
}


def _write_wav(path: Path, freq: float = 220.0, seconds: float = 0.3) -> None:
    sample_rate = 24000
    time_axis = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * freq * time_axis)
    pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def _manifest(tmp_path: Path) -> Path:
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir(exist_ok=True)
    _write_wav(wav_dir / "s1.wav", 220)
    _write_wav(wav_dir / "s2.wav", 300)
    metadata = tmp_path / "meta.csv"
    with metadata.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["s1", "Chào bạn"])
        writer.writerow(["s2", "Thử nghiệm"])
    manifest = tmp_path / "manifest.jsonl"
    build_manifest(str(metadata), str(wav_dir), str(manifest))
    return manifest


def test_onnx_runtime_end_to_end(tmp_path: Path):
    manifest = _manifest(tmp_path)

    # 1) Train + export acoustic
    acoustic_run = tmp_path / "acoustic"
    train_reference(TINY_CONFIG, str(manifest), str(manifest), str(acoustic_run), max_steps=2)
    acoustic_onnx = tmp_path / "model.onnx"
    export_onnx(TINY_CONFIG, str(acoustic_run / "last.pt"), str(acoustic_onnx))
    acoustic_meta = read_metadata(acoustic_onnx)
    assert acoustic_meta.get("vifasttts.model") == "acoustic"
    assert acoustic_meta.get("sample_rate") == "24000"

    # 2) Train + export vocoder
    vocoder_run = tmp_path / "vocoder"
    train_vocoder_reference(TINY_CONFIG, str(manifest), str(vocoder_run), max_steps=2)
    vocoder_onnx = tmp_path / "vocoder.onnx"
    export_vocoder_onnx(TINY_CONFIG, str(vocoder_run / "last.pt"), str(vocoder_onnx))
    vocoder_meta = read_metadata(vocoder_onnx)
    assert vocoder_meta["n_fft"] == "256"
    assert vocoder_meta["hop_length"] == "64"
    assert vocoder_meta["sample_rate"] == "24000"

    # 3) Mel ONNX phải khớp PyTorch (cùng checkpoint, eval mode)
    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
             len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    model = ViFastTtsModel(TINY_CONFIG, sizes)
    state = torch.load(acoustic_run / "last.pt", map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    items = parse_text("Chào bạn")
    features = torch.tensor([[vocabs.encode(item) for item in items]], dtype=torch.long)
    with torch.no_grad():
        mel_torch = model(features)["mel"].numpy()
    onnx_outputs = OnnxAcousticModel(acoustic_onnx).run(features.numpy())
    assert onnx_outputs["mel"].shape == mel_torch.shape
    assert float(np.max(np.abs(onnx_outputs["mel"] - mel_torch))) < 1e-2
    assert onnx_outputs["lengths"].tolist()[0] == onnx_outputs["durations"].sum()

    # 4) Tổng hợp bằng runtime ONNX (không cần PyTorch cho bước này)
    output = tmp_path / "out.wav"
    samples, sample_rate = synthesize_onnx(
        "Chào bạn hôm nay. Đây là câu thứ hai!",
        acoustic_onnx, output, vocoder_path=vocoder_onnx, peak_normalize=True,
    )
    assert output.exists() and samples > 0 and sample_rate == 24000
    with wave.open(str(output), "rb") as fh:
        assert fh.getnframes() == samples

    # 5) CLI synthesize-onnx chạy được
    cli_output = tmp_path / "cli.wav"
    result = runner.invoke(app, [
        "synthesize-onnx",
        "--acoustic", str(acoustic_onnx),
        "--vocoder", str(vocoder_onnx),
        "--text", "Xin chào",
        "--output", str(cli_output),
    ])
    assert result.exit_code == 0, result.stdout
    assert cli_output.exists()

    # 6) Streaming ONNX: 2 câu + khoảng nghỉ (max_chars nhỏ để buộc tách câu)
    from vifasttts.runtime.onnx_engine import iter_synthesize_onnx

    streamed = list(iter_synthesize_onnx(
        "Chào bạn. Câu hai!", acoustic_onnx, vocoder_path=vocoder_onnx, max_chars=12,
    ))
    assert len(streamed) >= 3
    assert sum(int(audio.size) for audio, _ in streamed) > 0
    assert all(rate == 24000 for _, rate in streamed)

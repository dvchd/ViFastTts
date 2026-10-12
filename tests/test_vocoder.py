"""Test vocoder: discriminator MPD-lite, dataset, huấn luyện tham chiếu, CLI."""
from __future__ import annotations

import csv
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from typer.testing import CliRunner

from vifasttts.cli import app
from vifasttts.data.dataset import VocoderDataset, collate_vocoder_batch
from vifasttts.data.manifest import build_manifest
from vifasttts.infer import synthesize
from vifasttts.models.discriminators import (
    MultiPeriodDiscriminatorLite,
    MultiPeriodMultiResolutionDiscriminator,
    discriminator_loss,
    feature_matching_loss,
    generator_adversarial_loss,
    multi_resolution_stft_loss,
)
from vifasttts.models.istftnet import ISTFTNetLite
from vifasttts.train import train_reference, train_vocoder_reference

runner = CliRunner()

VOC_CONFIG = {
    "seed": 0,
    "sample_rate": 24000,
    "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20, "f_min": 0, "f_max": 12000},
    "model": {"hidden_size": 16},
    "training": {"batch_size": 1, "num_workers": 0},
    "vocoder_training": {
        "batch_size": 1, "hidden": 16, "max_steps": 2, "log_every": 1, "save_every": 1,
        "discriminator_channels": 8, "crop_frames": 8,
    },
}


def _write_wav(path: Path, freq: float = 220.0, seconds: float = 0.3) -> None:
    import numpy as np

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
    _write_wav(wav_dir / "s1.wav", freq=220)
    _write_wav(wav_dir / "s2.wav", freq=300)
    metadata = tmp_path / "meta.csv"
    with metadata.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["s1", "Chào bạn"])
        writer.writerow(["s2", "Thử nghiệm"])
    manifest = tmp_path / "manifest.jsonl"
    build_manifest(str(metadata), str(wav_dir), str(manifest))
    return manifest


def test_mpd_lite_shapes_and_losses():
    discriminator = MultiPeriodDiscriminatorLite(periods=(2, 3, 5), channels=8)
    real = torch.randn(2, 2311)  # số lẻ để kiểm padding theo period
    fake = torch.randn(2, 2311, requires_grad=True)
    real_outputs = discriminator(real)
    fake_outputs = discriminator(fake)
    assert len(real_outputs) == 3
    logits, features = real_outputs[0]
    assert logits.shape == (2,)
    assert len(features) == 3

    d_loss = discriminator_loss(real_outputs, fake_outputs)
    assert torch.isfinite(d_loss)
    d_loss.backward(retain_graph=True)

    g_loss = generator_adversarial_loss(fake_outputs) + feature_matching_loss(real_outputs, fake_outputs)
    assert torch.isfinite(g_loss)
    g_loss.backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()


def test_mpd_full_mrd_shapes_and_losses():
    """MPD đầy đủ + MRD: logits/feature hợp lệ và loss backprop được."""
    discriminator = MultiPeriodMultiResolutionDiscriminator(
        periods=(2, 3), channels=8, resolutions=((64, 16, 64), (128, 32, 128)),
    )
    real = torch.randn(2, 512)
    fake = torch.randn(2, 512, requires_grad=True)
    real_outputs = discriminator(real)
    fake_outputs = discriminator(fake)
    assert len(real_outputs) == 4  # 2 period + 2 spectral
    for logits, features in real_outputs:
        assert logits.shape == (2,)
        assert len(features) in (4, 5)

    d_loss = discriminator_loss(real_outputs, fake_outputs)
    assert torch.isfinite(d_loss)
    d_loss.backward(retain_graph=True)

    g_loss = generator_adversarial_loss(fake_outputs) + feature_matching_loss(real_outputs, fake_outputs)
    assert torch.isfinite(g_loss)
    g_loss.backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()


def test_multi_resolution_stft_loss():
    pred = torch.randn(2, 1024, requires_grad=True)
    target = torch.randn(2, 1024)
    loss = multi_resolution_stft_loss(pred, target, resolutions=((64, 16, 64), (128, 32, 128)))
    assert torch.isfinite(loss) and float(loss) > 0.0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_build_generator_lite_vs_vocos():
    """Factory generator: mặc định lite (tương thích checkpoint cũ), vocos khi cấu hình."""
    from vifasttts.models.istftnet import ISTFTNetLite, ISTFTNetVocos, build_generator

    mel_config = {"n_mels": 20, "n_fft": 256, "hop_length": 64}
    lite = build_generator({}, mel_config)
    assert isinstance(lite, ISTFTNetLite)
    vocos = build_generator({"generator": "vocos", "hidden": 32, "generator_blocks": 2}, mel_config)
    assert isinstance(vocos, ISTFTNetVocos)
    mel = torch.randn(2, 16, 20)
    real, imag = vocos.spectrogram_parts(mel)
    assert real.shape == imag.shape and real.dim() == 3
    wave = vocos(mel)
    assert wave.dim() == 2 and wave.size(0) == 2


def test_vocoder_training_mpd_mrd_smoke(tmp_path: Path):
    """Train vocoder 1 bước với MPD đầy đủ + MRD + MR-STFT loss (config-driven)."""
    manifest = _manifest(tmp_path)
    config = {
        **VOC_CONFIG,
        "vocoder_training": {
            **VOC_CONFIG["vocoder_training"],
            "generator": "vocos",
            "generator_blocks": 2,
            "hidden": 16,
            "discriminator": "mpd_mrd",
            "discriminator_channels": 8,
            "mrd_resolutions": [[64, 16, 64], [128, 32, 128]],
            "stft_weight": 1.0,
        },
    }
    out = tmp_path / "vocoder_mrd"
    train_vocoder_reference(config, str(manifest), str(out), max_steps=1)
    assert (out / "last.pt").exists()
    state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert "discriminator" in state and state["step"] == 1


def test_vocoder_dataset_and_collate(tmp_path: Path):
    manifest = _manifest(tmp_path)
    dataset = VocoderDataset(str(manifest), VOC_CONFIG)
    item = dataset[0]
    frames = item["mel"].size(0)
    assert frames > 0 and item["mel"].size(1) == 20
    assert item["wav"].numel() == frames * 64  # khớp hop_length

    batch = collate_vocoder_batch([dataset[0], dataset[1]])
    assert batch["mel"].size(0) == 2
    assert batch["wav"].size(1) == batch["mel"].size(1) * 64
    assert batch["mel_lengths"].tolist() == [dataset[0]["mel"].size(0), dataset[1]["mel"].size(0)]


def test_vocoder_training_smoke(tmp_path: Path):
    manifest = _manifest(tmp_path)
    out = tmp_path / "vocoder_run"
    train_vocoder_reference(VOC_CONFIG, str(manifest), str(out), max_steps=2)

    assert (out / "last.pt").exists()
    assert (out / "history_vocoder.csv").exists()
    assert (out / "resolved_config.json").exists()
    state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert "vocoder" in state and "discriminator" in state and state["step"] == 2
    from vifasttts.data.features import canonical_mel_config, mel_config_hash

    assert state["mel_hash"] == mel_config_hash(canonical_mel_config(VOC_CONFIG))

    # Checkpoint khớp kiến trúc ISTFTNetLite mà `synthesize` dựng lên.
    vocoder = ISTFTNetLite(n_mels=20, hidden=16, n_fft=256, hop_length=64)
    vocoder.load_state_dict(state["vocoder"])


def test_vocoder_training_resume(tmp_path: Path):
    manifest = _manifest(tmp_path)
    out = tmp_path / "vocoder_run"
    train_vocoder_reference(VOC_CONFIG, str(manifest), str(out), max_steps=2)
    state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 2 and "opt_g" in state and "opt_d" in state

    train_vocoder_reference(VOC_CONFIG, str(manifest), str(out), max_steps=4,
                            resume=str(out / "last.pt"))
    state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 4


def test_train_vocoder_cli(tmp_path: Path):
    manifest = _manifest(tmp_path)
    config_path = tmp_path / "voc.yaml"
    config_path.write_text(
        "sample_rate: 24000\n"
        "mel:\n  n_fft: 256\n  win_length: 256\n  hop_length: 64\n  n_mels: 20\n  f_min: 0\n  f_max: 12000\n"
        "model:\n  hidden_size: 16\n"
        "training:\n  num_workers: 0\n"
        "vocoder_training:\n  batch_size: 1\n  hidden: 16\n  log_every: 1\n  save_every: 1\n  discriminator_channels: 8\n",
        encoding="utf-8",
    )
    output = tmp_path / "cli_vocoder"
    result = runner.invoke(app, [
        "train-vocoder", "--config", str(config_path), "--manifest", str(manifest),
        "--output", str(output), "--max-steps", "1",
    ])
    assert result.exit_code == 0, result.stdout
    assert (output / "last.pt").exists()


ACOUSTIC_TINY = {
    **VOC_CONFIG,
    "model": {
        "component_dim": 8, "hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1,
        "attention_heads": 2, "ffn_size": 32, "conv_kernel_size": 3, "dropout": 0.0,
        "duration_layers": 1, "variance_layers": 1,
    },
    "training": {
        "batch_size": 1, "learning_rate": 1e-3, "weight_decay": 0.0, "max_steps": 2,
        "warmup_steps": 1, "grad_clip": 1.0, "log_every": 1, "validate_every": 2,
        "save_every": 2, "num_workers": 0, "mixed_precision": False,
    },
    "loss_weights": {"mel": 1.0, "duration": 1.0, "f0": 1.0, "vuv": 0.5, "energy": 0.5, "tone": 0.25},
}


def test_synthesize_uses_trained_vocoder(tmp_path: Path):
    """Inference dùng đúng hidden/upsample_ratio từ config như lúc train vocoder."""
    manifest = _manifest(tmp_path)
    acoustic_out = tmp_path / "acoustic"
    train_reference(ACOUSTIC_TINY, str(manifest), str(manifest), str(acoustic_out), max_steps=2)
    vocoder_out = tmp_path / "vocoder"
    train_vocoder_reference(ACOUSTIC_TINY, str(manifest), str(vocoder_out), max_steps=2)

    wav_out = tmp_path / "out.wav"
    samples, sample_rate = synthesize(
        ACOUSTIC_TINY, str(acoustic_out / "last.pt"), "Chào bạn", str(wav_out),
        vocoder_checkpoint=str(vocoder_out / "last.pt"),
    )
    assert wav_out.exists() and samples > 0 and sample_rate == 24000

    # Không truyền vocoder → cảnh báo RuntimeWarning thay vì im lặng.
    with pytest.warns(RuntimeWarning):
        synthesize(ACOUSTIC_TINY, str(acoustic_out / "last.pt"), "Chào", str(tmp_path / "warn.wav"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS không khả dụng")
def test_vocoder_mps_smoke(tmp_path: Path):
    """Mac Apple Silicon: 1 bước train vocoder trên MPS (torch.istft phải chạy được)."""
    manifest = _manifest(tmp_path)
    config = {
        **VOC_CONFIG,
        "vocoder_training": {**VOC_CONFIG["vocoder_training"], "device": "mps"},
    }
    out = tmp_path / "vocoder_mps"
    train_vocoder_reference(config, str(manifest), str(out), max_steps=1)
    assert (out / "last.pt").exists()

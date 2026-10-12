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


def test_train_resume_continues_from_step(tmp_path: Path):
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

    out_dir = tmp_path / "run"
    train_reference(TINY_CONFIG, str(manifest), str(manifest), str(out_dir), max_steps=3)
    state = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 3
    assert "optimizer" in state  # đủ để resume đúng trạng thái

    train_reference(TINY_CONFIG, str(manifest), str(manifest), str(out_dir),
                    max_steps=5, resume=str(out_dir / "last.pt"))
    state = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 5
    assert "best_val" in state  # lưu lại để best.pt không bị ghi đè oan sau resume
    # history.csv ghi tiếp, không xoá lịch sử val của lần chạy trước.
    lines = (out_dir / "history.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("step,")
    assert sum(1 for line in lines if line.startswith("step,")) == 1
    assert len(lines) >= 3


def test_learning_rate_schedules():
    from vifasttts.train import _learning_rate

    base = 2e-4
    # Warmup tuyến tính.
    assert _learning_rate(0, 100, 1000, base) < base
    assert _learning_rate(100, 100, 1000, base) == base
    # Hằng số (mặc định, hành vi cũ).
    assert _learning_rate(500, 100, 1000, base) == base
    # Cosine: giảm đơn điệu sau warmup, chạm min_ratio ở bước cuối.
    early = _learning_rate(200, 100, 1000, base, "cosine")
    mid = _learning_rate(500, 100, 1000, base, "cosine")
    final = _learning_rate(1000, 100, 1000, base, "cosine")
    assert base > early > mid > final
    assert abs(final - base * 0.05) < 1e-12
    # Chạy theo khối: mốc cuối khối (20k) không được kéo LR xuống — phải dùng
    # decay_steps toàn cục (60k), nếu không LR sẽ sụt sai khi mỗi khối bắt đầu.
    chunk_lr = _learning_rate(15001, 6000, 20000, base, "cosine", 0.05, decay_steps=60000)
    assert chunk_lr > base * 0.85


def test_run_validation_max_batches(tmp_path: Path):
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

    from vifasttts.models.vifasttts import ViFastTtsModel
    from vifasttts.train import _build_loaders, _run_validation

    _, val_loader = _build_loaders(TINY_CONFIG, str(manifest), str(manifest), TINY_CONFIG["training"], None)
    model = ViFastTtsModel(TINY_CONFIG, _vocab_sizes())
    loss = _run_validation(model, val_loader, TINY_CONFIG, torch.device("cpu"), max_batches=1)
    assert loss > 0


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

    # Streaming: các đoạn ghép lại đúng bằng waveform đầy đủ.
    from vifasttts.infer import iter_synthesize, synthesize_batch

    streamed = list(iter_synthesize(TINY_CONFIG, str(checkpoint), "Chào bạn"))
    assert streamed and sum(int(audio.size) for audio, _ in streamed) == samples
    assert all(rate == 24000 for _, rate in streamed)

    # Batch: mỗi câu một file, nạp model một lần.
    batch_dir = tmp_path / "batch"
    paths = synthesize_batch(
        TINY_CONFIG, str(checkpoint),
        [("s1", "Chào bạn"), ("s2", "Tạm biệt")], batch_dir,
    )
    assert len(paths) == 2 and all(path.exists() for path in paths)

    # CLI batch: dòng có tab là id, dòng trần tự sinh id.
    import yaml
    from typer.testing import CliRunner

    from vifasttts.cli import app

    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(TINY_CONFIG, allow_unicode=True), encoding="utf-8")
    batch_input = tmp_path / "batch.txt"
    batch_input.write_text("id1\tChào bạn\nChào buổi sáng\n", encoding="utf-8")
    cli = CliRunner().invoke(app, [
        "synthesize-batch", "--config", str(config_path), "--checkpoint", str(checkpoint),
        "--input", str(batch_input), "--output-dir", str(tmp_path / "batch_cli"),
    ])
    assert cli.exit_code == 0, cli.stdout
    assert (tmp_path / "batch_cli" / "id1.wav").exists()
    assert (tmp_path / "batch_cli" / "line_0002.wav").exists()

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


def test_variant_configs_match_estimated_params():
    """Config biến thể phải khớp số tham số ghi trong TRAINING_ESTIMATES.md."""
    from vifasttts.config import validate_config

    sizes = _vocab_sizes()
    small = load_config("configs/small_24k.yaml")
    large = load_config("configs/large_24k.yaml")
    validate_config(small, "acoustic")
    validate_config(large, "acoustic")
    assert sum(p.numel() for p in ViFastTtsModel(small, sizes).parameters()) == 6_139_054
    assert sum(p.numel() for p in ViFastTtsModel(large, sizes).parameters()) == 32_156_654


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS không khả dụng")
def test_mps_training_and_synthesis_smoke(tmp_path: Path):
    """Mac Apple Silicon: train 1 bước + tổng hợp bằng MPS phải chạy được."""
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "s1.wav", freq=220)
    metadata = tmp_path / "meta.csv"
    with metadata.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["s1", "Chào bạn"])
    manifest = tmp_path / "manifest.jsonl"
    build_manifest(str(metadata), str(wav_dir), str(manifest))

    config = {**TINY_CONFIG, "training": {**TINY_CONFIG["training"], "device": "mps"}}
    out = tmp_path / "mps_run"
    train_reference(config, str(manifest), str(manifest), str(out), max_steps=1)
    assert (out / "last.pt").exists()

    wav_out = tmp_path / "mps.wav"
    with pytest.warns(RuntimeWarning):  # chưa có vocoder riêng
        samples, rate = synthesize(config, str(out / "last.pt"), "Chào", str(wav_out), device="mps")
    assert wav_out.exists() and samples > 0 and rate == 24000


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

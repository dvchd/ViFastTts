"""Test HTTP server TTS (stdlib) và engine giữ model đã nạp."""
from __future__ import annotations

import csv
import io
import json
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vifasttts.data.manifest import build_manifest
from vifasttts.server import TtsEngine, create_server, wav_bytes
from vifasttts.train import train_reference

CONFIG = {
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


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    tmp = tmp_path_factory.mktemp("server")
    wav_dir = tmp / "wavs"
    wav_dir.mkdir()
    _write_wav(wav_dir / "s1.wav", 220)
    _write_wav(wav_dir / "s2.wav", 300)
    metadata = tmp / "meta.csv"
    with metadata.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["s1", "Chào bạn"])
        writer.writerow(["s2", "Thử nghiệm"])
    manifest = tmp / "manifest.jsonl"
    build_manifest(str(metadata), str(wav_dir), str(manifest))
    out = tmp / "run"
    train_reference(CONFIG, str(manifest), str(manifest), str(out), max_steps=2)
    return str(out / "last.pt")


def test_wav_bytes_is_valid_wav():
    data = wav_bytes(np.zeros(2400, dtype=np.float32), 24000)
    assert data[:4] == b"RIFF"
    with wave.open(io.BytesIO(data), "rb") as fh:
        assert fh.getframerate() == 24000
        assert fh.getnframes() == 2400
        assert fh.getnchannels() == 1


def test_engine_validates_payload(checkpoint: str):
    with pytest.warns(RuntimeWarning):  # không truyền vocoder → random weights
        engine = TtsEngine(CONFIG, checkpoint)
    with pytest.raises(ValueError):
        engine.synthesize_bytes({"text": "   "})
    with pytest.raises(ValueError):
        engine.synthesize_bytes({"text": "ba", "speed": 99.0})
    with pytest.raises(ValueError):
        engine.synthesize_bytes({"text": "ba", "overrides": ["không phải dict"]})


def test_engine_synthesizes_wav_bytes(checkpoint: str):
    with pytest.warns(RuntimeWarning):
        engine = TtsEngine(CONFIG, checkpoint)
    audio = engine.synthesize_bytes({"text": "Chào bạn hôm nay", "peak_normalize": True})
    assert audio[:4] == b"RIFF"
    with wave.open(io.BytesIO(audio), "rb") as fh:
        assert fh.getnframes() > 0
        assert fh.getframerate() == 24000


def test_http_endpoints(checkpoint: str):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        engine = TtsEngine(CONFIG, checkpoint)
    server = create_server(engine, host="127.0.0.1", port=0)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        health = json.loads(urllib.request.urlopen(f"{base}/health", timeout=5).read())
        assert health["status"] == "ok"

        info = json.loads(urllib.request.urlopen(f"{base}/info", timeout=5).read())
        assert info["sample_rate"] == 24000

        request = urllib.request.Request(
            f"{base}/synthesize",
            data=json.dumps({"text": "Chào bạn"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.headers["Content-Type"] == "audio/wav"
            audio = response.read()
        assert audio[:4] == b"RIFF"

        bad = urllib.request.Request(
            f"{base}/synthesize",
            data=json.dumps({"text": ""}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(bad, timeout=5)
        assert excinfo.value.code == 400

        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"{base}/khong-ton-tai", timeout=5)
        assert excinfo.value.code == 404
    finally:
        server.shutdown()
        server.server_close()

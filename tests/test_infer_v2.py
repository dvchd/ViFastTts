"""Test `Synthesizer` với checkpoint v2 + CER free-run v2 (CPU, không cần audio thật).

Bao phủ (docs/ACOUSTIC_V2.md §5, §10–§12):
- `Synthesizer` nhận checkpoint v2 (`arch="v2_slot5"` + `f0_stats` giả) và tổng hợp
  được với vocoder lite nhỏ (không raise, WAV float32, đúng sample_rate).
- `speed=0.5` cho waveform dài hơn (hoặc bằng) `speed=1.0` ở cả v1 lẫn v2.
- Checkpoint thiếu `arch` vẫn chạy đường v1 cũ.
- `evaluate_free_v2` gọi trực tiếp với dataset có `text`: không raise; `cer` là số
  khi có faster-whisper, `None` (+ `cer_note`, không crash) khi thiếu.

Toàn bộ chạy CPU (`device="cpu"`), không dùng MPS, không đọc/ghi `runs/`.
"""

from __future__ import annotations

import json
import math
import sys
import wave
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import eval_acoustic as ev

from vifasttts.data.cache import feature_cache_key
from vifasttts.data.dataset_v2 import SlotDataset
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.infer import Synthesizer
from vifasttts.models.factory import ARCH_V1, ARCH_V2, build_model
from vifasttts.models.istftnet import build_generator
from vifasttts.models.vifasttts import ViFastTtsModel

MEL_DIM = 20
CONFIG = {
    "seed": 0,
    "sample_rate": 24000,
    "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": MEL_DIM,
            "f_min": 0, "f_max": 12000},
    "model": {
        "component_dim": 8, "hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1,
        "attention_heads": 2, "ffn_size": 32, "conv_kernel_size": 3, "dropout": 0.0,
        "duration_layers": 1, "variance_layers": 1, "use_phonation": False,
        "use_postnet": False, "role_dim": 4, "segment_mlp_layers": 2,
    },
    "vocoder_training": {"hidden": 8},  # lite nhỏ cho test (không train vocoder thật)
    "f0": {"method": "pyworld"},
    "phonation": {"enabled": False, "channels": []},
}
F0_STATS = {
    "f0_log_mean": 4.8, "f0_log_std": 0.2,
    "shape_mean": [0.0, 0.0, 0.0, 0.0], "shape_std": [1.0, 1.0, 1.0, 1.0],
    "energy_mean": 0.0, "energy_std": 1.0,
}


def _sizes() -> list[int]:
    vocabs = ComponentVocabs.default()
    return [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
            len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]


def _save_v2_checkpoint(config: dict, path: Path) -> None:
    torch.manual_seed(0)
    model = build_model(config, _sizes(), arch=ARCH_V2)
    model.load_f0_stats(F0_STATS)
    torch.save({"model": model.state_dict(), "arch": ARCH_V2, "f0_stats": dict(F0_STATS)},
               str(path))


def _save_v1_checkpoint(config: dict, path: Path) -> None:
    """Checkpoint v1 (thiếu `arch` → tương thích ngược), trọng số ngẫu nhiên."""
    torch.manual_seed(0)
    model = ViFastTtsModel(config, _sizes()).eval()
    torch.save({"model": model.state_dict()}, str(path))


def _save_vocoder_checkpoint(config: dict, path: Path) -> None:
    """Vocoder lite nhỏ ở định dạng `train-vocoder` (`{"vocoder": ...}`)."""
    torch.manual_seed(1)
    generator = build_generator(config.get("vocoder_training", {}), config["mel"]).eval()
    torch.save({"vocoder": generator.state_dict()}, str(path))


def _wav_frames(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as fh:
        return fh.getnframes(), fh.getframerate()


def test_synthesizer_v2(tmp_path):
    ckpt, vckpt = tmp_path / "v2.pt", tmp_path / "voc.pt"
    _save_v2_checkpoint(CONFIG, ckpt)
    _save_vocoder_checkpoint(CONFIG, vckpt)
    engine = Synthesizer(CONFIG, str(ckpt), vocoder_checkpoint=str(vckpt), device="cpu")
    assert engine.arch == ARCH_V2

    out = tmp_path / "v2.wav"
    samples, rate = engine.synthesize("cà phê", out)
    assert rate == 24000 and samples > 0 and out.exists()
    frames, wav_rate = _wav_frames(out)
    assert (frames, wav_rate) == (samples, 24000)


def test_synthesizer_v2_speed(tmp_path):
    ckpt, vckpt = tmp_path / "v2.pt", tmp_path / "voc.pt"
    _save_v2_checkpoint(CONFIG, ckpt)
    _save_vocoder_checkpoint(CONFIG, vckpt)
    engine = Synthesizer(CONFIG, str(ckpt), vocoder_checkpoint=str(vckpt), device="cpu")

    normal, _ = engine.synthesize("cà phê", tmp_path / "s1.wav", speed=1.0)
    slow, _ = engine.synthesize("cà phê", tmp_path / "s05.wav", speed=0.5)
    assert slow >= normal > 0  # speed < 1 → dài hơn (speed trong graph, §5)


def test_synthesizer_v1_khong_doi(tmp_path):
    ckpt, vckpt = tmp_path / "v1.pt", tmp_path / "voc.pt"
    _save_v1_checkpoint(CONFIG, ckpt)
    _save_vocoder_checkpoint(CONFIG, vckpt)
    engine = Synthesizer(CONFIG, str(ckpt), vocoder_checkpoint=str(vckpt), device="cpu")
    assert engine.arch == ARCH_V1

    out = tmp_path / "v1.wav"
    samples, rate = engine.synthesize("cà phê", out)
    assert rate == 24000 and samples > 0
    assert _wav_frames(out)[0] == samples
    slow, _ = engine.synthesize("cà phê", tmp_path / "v1_slow.wav", speed=0.5)
    assert slow >= samples  # đường 2-forward cũ vẫn scale durations


def _syllable(raw: str, abstract: dict) -> dict:
    return {
        "raw": raw, "normalized": raw,
        "orthography": {"onset": None, "medial": None, "nucleus": "a", "coda": None},
        "abstract": abstract,
        "boundaries": {"word_end": True, "phrase_end": False, "sentence_end": False},
        "status": "VALID", "confidence": 1.0, "punctuation": None,
    }


def _free_cmp_fixture(tmp_path: Path):
    """Dataset v2 1 row có `text` + cache giả (copy fixture `test_eval_acoustic_v2`)."""
    torch.manual_seed(0)
    row = {
        "id": "clip__s00", "audio_path": "/khong/ton/tai.wav", "text": "cà phê",
        "frame_range": [10, 23],
        "durations": [5, 6], "gaps": [2], "seg_durations": [[2, 0, 2, 1], [2, 0, 2, 2]],
        "lead_frames": 0,
        "syllables": [
            _syllable("các", {"onset": "K", "medial": "NONE", "nucleus": "A",
                              "coda": "C", "tone": "SAC"}),
            _syllable("ma", {"onset": "M", "medial": "NONE", "nucleus": "A",
                             "coda": "N", "tone": "NGANG"}),
        ],
    }
    manifest = tmp_path / "slot4.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    mel_stats = tmp_path / "mel_stats.json"
    mel_stats.write_text(json.dumps({"mean": [0.0] * MEL_DIM, "std": [1.0] * MEL_DIM}),
                         encoding="utf-8")
    config = {**CONFIG, "mel": {**CONFIG["mel"], "normalize_path": str(mel_stats)}}
    start, end = row["frame_range"]
    torch.save({
        "mel": torch.randn(end, MEL_DIM),
        "energy": torch.full((end,), 2.5),
        "f0": torch.full((end,), 150.0),
        "vuv": torch.ones(end),
        "hnr": torch.zeros(end),
        "glottalization": torch.full((end,), 0.1),
        "jitter": torch.zeros(end),
    }, cache_dir / f"{feature_cache_key(row, config)}.pt")
    assert end - start == sum(row["durations"]) + sum(row["gaps"])  # 13 frame khớp cắt cache
    dataset = SlotDataset(manifest, config, F0_STATS, cache_dir=cache_dir)
    model = build_model(config, _sizes(), arch=ARCH_V2)
    model.load_f0_stats(F0_STATS)
    return config, dataset, model


def test_evaluate_free_v2_cer(tmp_path):
    config, dataset, model = _free_cmp_fixture(tmp_path)
    report = ev.evaluate_free_v2(model, dataset, config, device="cpu", speed=1.0)
    assert report["mode"] == "free" and report["arch"] == ARCH_V2
    assert report["clips"] == 1
    json.dumps(report)  # JSON-serializable
    try:
        import faster_whisper  # noqa: F401

        has_whisper = True
    except ImportError:
        has_whisper = False
    if has_whisper:
        assert isinstance(report["cer"], dict)
        assert math.isfinite(report["cer"]["mean"])
        assert math.isfinite(report["per_clip"][0]["cer"])
    else:
        assert report["cer"] is None and report["cer_note"]

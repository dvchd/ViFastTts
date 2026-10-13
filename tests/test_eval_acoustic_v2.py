"""Test nhánh v2 của `tools/eval_acoustic.py` (cache giả + checkpoint ngẫu nhiên, CPU)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import eval_acoustic as ev

from vifasttts.data.cache import feature_cache_key
from vifasttts.data.dataset_v2 import SlotDataset
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.models.factory import arch_of_checkpoint, build_model

MEL_DIM = 8
CONFIG = {
    "sample_rate": 24000,
    "mel": {"hop_length": 256, "n_fft": 256, "n_mels": MEL_DIM},
    "f0": {"method": "pyworld"},
    "phonation": {"enabled": False, "channels": []},
    "model": {
        "component_dim": 8, "hidden_size": 16, "encoder_layers": 1, "decoder_layers": 1,
        "attention_heads": 2, "ffn_size": 32, "conv_kernel_size": 3, "dropout": 0.0,
        "duration_layers": 1, "variance_layers": 1, "use_phonation": False,
        "use_postnet": False,
    },
}
F0_STATS = {
    "f0_log_mean": 4.8, "f0_log_std": 0.2,
    "shape_mean": [0.0, 0.0, 0.0, 0.0], "shape_std": [1.0, 1.0, 1.0, 1.0],
    "energy_mean": 0.0, "energy_std": 1.0,
}


def _syllable(raw: str, abstract: dict) -> dict:
    return {
        "raw": raw, "normalized": raw,
        "orthography": {"onset": None, "medial": None, "nucleus": "a", "coda": None},
        "abstract": abstract,
        "boundaries": {"word_end": True, "phrase_end": False, "sentence_end": False},
        "status": "VALID", "confidence": 1.0, "punctuation": None,
    }


def _row(row_id: str, start: int) -> dict:
    return {
        "id": row_id, "audio_path": f"/khong/ton/tai_{row_id}.wav",
        "frame_range": [start, start + 13],
        "durations": [5, 6], "gaps": [2], "seg_durations": [[2, 0, 2, 1], [2, 0, 2, 2]],
        "lead_frames": 0,
        "syllables": [
            _syllable("các", {"onset": "K", "medial": "NONE", "nucleus": "A",
                              "coda": "C", "tone": "SAC"}),
            _syllable("ma", {"onset": "M", "medial": "NONE", "nucleus": "A",
                             "coda": "N", "tone": "NGANG"}),
        ],
    }


ROW1 = _row("clip__s00", 10)
ROW2 = _row("clip__s01", 30)


def _write_cache(cache_dir: Path, row: dict) -> None:
    """Cache giả độ dài clip cha (f0 dương để F0 eval có mẫu hữu thanh)."""
    _write_cache_with(CONFIG, cache_dir, row)


def _write_cache_with(config: dict, cache_dir: Path, row: dict) -> None:
    """Ghi cache với đúng config dùng cho dataset (key gồm cả `normalize_path`)."""
    start = row["frame_range"][0]
    length = start + (row["frame_range"][1] - start)
    torch.save({
        "mel": torch.randn(length, MEL_DIM),
        "energy": torch.full((length,), 2.5),
        "f0": torch.full((length,), 150.0),
        "vuv": torch.ones(length),
        "hnr": torch.zeros(length),
        "glottalization": torch.full((length,), 0.1),
        "jitter": torch.zeros(length),
    }, cache_dir / f"{feature_cache_key(row, config)}.pt")


@pytest.fixture()
def setup(tmp_path):
    torch.manual_seed(0)
    manifest = tmp_path / "slot4.jsonl"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in (ROW1, ROW2)),
                        encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    mel_stats = tmp_path / "mel_stats.json"
    mel_stats.write_text(json.dumps({"mean": [0.0] * MEL_DIM, "std": [1.0] * MEL_DIM}),
                         encoding="utf-8")
    config = {**CONFIG, "mel": {**CONFIG["mel"], "normalize_path": str(mel_stats)}}
    for row in (ROW1, ROW2):
        _write_cache_with(config, cache_dir, row)
    dataset = SlotDataset(manifest, config, F0_STATS, cache_dir=cache_dir)
    sizes = [len(getattr(ComponentVocabs.default(), name)) for name in
             ("onset", "medial", "nucleus", "coda", "tone", "boundary", "punctuation")]
    model = build_model(config, sizes, arch="v2_slot5")
    model.load_f0_stats(F0_STATS)
    checkpoint = tmp_path / "v2.pt"
    torch.save({"model": model.state_dict(), "arch": "v2_slot5", "f0_stats": F0_STATS,
                "seg_method": "test", "seg_method_version": 1, "seg_weights_hash": "x"},
               checkpoint)
    return config, dataset, checkpoint


def _assert_finite(report: dict, keys: list[str]) -> None:
    for key in keys:
        value = report[key]
        stats = value if isinstance(value, dict) else {"mean": value}
        for name, number in stats.items():
            if name == "n" or number is None:
                continue
            assert math.isfinite(float(number)), (key, name, number)


def test_checkpoint_arch_v2(setup):
    _, _, checkpoint = setup
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert arch_of_checkpoint(state) == "v2_slot5"
    assert arch_of_checkpoint({}) == "v1_syl"  # tương thích ngược


def test_load_eval_model_v2(setup):
    config, _, checkpoint = setup
    model, arch = ev.load_eval_model(str(checkpoint), config, f0_stats=F0_STATS)
    assert arch == "v2_slot5"
    assert float(model.f0_log_mean) == pytest.approx(4.8)


def test_evaluate_tf_seg_v2(setup):
    config, dataset, checkpoint = setup
    model, _ = ev.load_eval_model(str(checkpoint), config, f0_stats=F0_STATS)
    report = ev.evaluate_tf_v2(model, dataset, config, use_seg=True, device="cpu")
    assert report["mode"] == "tf_seg" and report["arch"] == "v2_slot5"
    assert report["clips"] == 2
    _assert_finite(report, ["mel_l1_z", "mel_l1_raw", "mel_mae_db", "duration_mae_frames",
                            "slot_sum_mae_frames"])
    assert math.isfinite(report["vuv_accuracy"])
    assert math.isfinite(report["energy_mae"])
    assert report["vuv_note"] and report["energy_note"] and report["f0_note"]
    assert report["slot_sum_mae_frames"] == pytest.approx(0.0, abs=1e-3)  # seg thật → Σ khớp
    assert any(stats.get("n") for stats in report["f0_rmse_cents_by_tone"].values())
    assert len(report["per_clip"]) == 2


def test_evaluate_tf_syl_v2_slot_b2(setup):
    config, dataset, checkpoint = setup
    model, _ = ev.load_eval_model(str(checkpoint), config, f0_stats=F0_STATS)
    report = ev.evaluate_tf_v2(model, dataset, config, use_seg=False, device="cpu")
    assert report["mode"] == "tf_syl"
    assert math.isfinite(report["slot_sum_mae_frames"])  # B2 tự chia, Σ phải khớp
    assert report["slot_sum_mae_frames"] == pytest.approx(0.0, abs=1.0)
    _assert_finite(report, ["mel_l1_z", "mel_mae_db"])


def test_evaluate_free_v2_speed(setup):
    config, dataset, checkpoint = setup
    model, _ = ev.load_eval_model(str(checkpoint), config, f0_stats=F0_STATS)
    report = ev.evaluate_free_v2(model, dataset, config, device="cpu", speed=1.0)
    assert report["mode"] == "free" and report["arch"] == "v2_slot5"
    assert report["cer"] is None and report["cer_note"]
    assert math.isfinite(report["rtf"]) and math.isfinite(report["ram_peak_gib"])
    assert all(math.isfinite(entry["length_ratio"]) for entry in report["per_clip"])
    assert "slot_durations" not in report  # không rò tensor vào JSON
    json.dumps(report)  # JSON-serializable


def test_resolve_f0_stats_priority(setup, tmp_path):
    config, _, checkpoint = setup
    from_cli = tmp_path / "cli_stats.json"
    from_cli.write_text(json.dumps({**F0_STATS, "f0_log_mean": 5.0}), encoding="utf-8")
    resolved = ev.resolve_f0_stats(str(checkpoint), config, str(from_cli))
    assert resolved["f0_log_mean"] == 5.0  # CLI thắng checkpoint
    resolved2 = ev.resolve_f0_stats(str(checkpoint), config, None)
    assert resolved2["f0_log_mean"] == pytest.approx(4.8)  # checkpoint

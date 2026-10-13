"""Test `data/dataset_v2.py` — SlotDataset + collate_slot_batch (dùng cache giả, <5 s)."""

from __future__ import annotations

import json

import pytest
import torch

from vifasttts.data.cache import feature_cache_key
from vifasttts.data.dataset import encode_syllables
from vifasttts.data.dataset_v2 import SlotDataset, collate_slot_batch
from vifasttts.frontend.vocab import ComponentVocabs

CONFIG = {
    "sample_rate": 24000,
    "mel": {"hop_length": 256, "n_fft": 1024},
    "f0": {"method": "pyworld"},
    "phonation": {"enabled": True, "channels": ["hnr", "glottalization", "jitter"]},
}
F0_STATS = {
    "f0_log_mean": 4.8, "f0_log_std": 0.2,
    "shape_mean": [0, 0, 0, 0], "shape_std": [1, 1, 1, 1],
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


def _row(row_id: str, frame_start: int, durations: list[int], gaps: list[int],
         seg: list[list[int]]) -> dict:
    return {
        "id": row_id, "audio_path": f"/khong/ton/tai_{row_id}.wav",
        "frame_range": [frame_start, frame_start + sum(durations) + sum(gaps)],
        "durations": durations, "gaps": gaps, "seg_durations": seg, "lead_frames": 0,
        "syllables": [
            _syllable("các", {"onset": "K", "medial": "NONE", "nucleus": "A",
                              "coda": "C", "tone": "SAC"}),
            _syllable("ma", {"onset": "M", "medial": "NONE", "nucleus": "A",
                             "coda": "N", "tone": "NGANG"}),
        ],
    }


ROW1 = _row("clip__s00", 10, [5, 6], [2], [[2, 0, 2, 1], [2, 0, 2, 2]])  # 13 frame
ROW2 = _row("clip__s01", 30, [4, 5], [3], [[1, 0, 2, 1], [2, 0, 1, 2]])  # 12 frame


def _write_cache(cache_dir, row: dict, length: int | None = None) -> None:
    """Ghi cache giả **độ dài clip cha** (`start + frames`) như cache thật."""
    start = row["frame_range"][0]
    frames = row["frame_range"][1] - start
    length = length if length is not None else start + frames
    torch.save({
        "mel": torch.randn(length, 20),
        "energy": torch.full((length,), 2.5),
        "f0": torch.full((length,), 150.0),
        "vuv": torch.ones(length),
        "hnr": torch.zeros(length),
        "glottalization": torch.full((length,), 0.1),
        "jitter": torch.zeros(length),
    }, cache_dir / f"{feature_cache_key(row, CONFIG)}.pt")


@pytest.fixture()
def dataset(tmp_path):
    manifest = tmp_path / "slot4.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in (ROW1, ROW2)),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_cache(cache_dir, ROW1)
    _write_cache(cache_dir, ROW2)
    return SlotDataset(manifest, CONFIG, F0_STATS, cache_dir=cache_dir)


def test_getitem_key_shape_dtype(dataset):
    item = dataset[0]
    for key in ("id", "features", "durations", "gaps", "lead", "mel", "energy", "f0",
                "vuv", "seg_durations", "seg_ratio", "seg_ratio_mask", "f0_shape",
                "f0_weight", "seg_energy", "seg_vuv", "pause_gate", "pause_log_gap",
                "pause_mask"):
        assert key in item, key
    assert item["id"] == "clip__s00"
    assert item["features"].shape == (2, 7) and item["features"].dtype == torch.long
    assert item["mel"].shape == (13, 20) and item["mel"].dtype == torch.float32
    assert item["energy"].shape == (13,) and item["f0"].shape == (13,)
    assert item["vuv"].shape == (13,)
    assert item["durations"].dtype == torch.long and item["gaps"].dtype == torch.long
    assert item["lead"].shape == () and item["lead"].dtype == torch.long
    assert item["seg_durations"].shape == (2, 4) and item["seg_durations"].dtype == torch.long
    assert item["seg_ratio"].shape == (2, 4) and item["f0_shape"].shape == (2, 4, 4)
    assert item["seg_ratio_mask"].dtype == torch.bool and item["pause_mask"].dtype == torch.bool
    assert len(dataset) == 2
    assert all(isinstance(v, torch.Tensor) for k, v in item.items() if k != "id")


def test_getitem_gia_tri_khop(dataset):
    item = dataset[0]
    expect_features = torch.tensor(
        encode_syllables(ROW1["syllables"], ComponentVocabs.default()), dtype=torch.long)
    assert torch.equal(item["features"], expect_features)
    assert item["durations"].tolist() == [5, 6]
    assert item["gaps"].tolist() == [2, 0]  # gaps N-1 → thêm 0 cuối
    assert int(item["lead"]) == 0
    assert item["seg_durations"].tolist() == [[2, 0, 2, 1], [2, 0, 2, 2]]
    assert torch.allclose(item["seg_ratio"][0], torch.tensor([2 / 5, 0.0, 2 / 5, 1 / 5]))
    assert torch.allclose(item["seg_ratio"][1], torch.tensor([2 / 6, 0.0, 2 / 6, 2 / 6]))


def test_getitem_thieu_seg_durations_raise(tmp_path):
    rows = [{**ROW1, "id": "thieu-seg"}]
    del rows[0]["seg_durations"]
    manifest = tmp_path / "thieu.jsonl"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                        encoding="utf-8")
    dataset = SlotDataset(manifest, CONFIG, F0_STATS, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="seg_durations"):
        dataset[0]


def test_getitem_cache_ngan_hon_frame_range_raise(tmp_path):
    """Không fallback im lặng khi cache ngắn hơn frame_range (dữ liệu hỏng phải lộ)."""
    manifest = tmp_path / "ngan.jsonl"
    manifest.write_text(json.dumps(ROW1, ensure_ascii=False), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_cache(cache_dir, ROW1, length=5)  # < start(10) + frames(13)
    dataset = SlotDataset(manifest, CONFIG, F0_STATS, cache_dir=cache_dir)
    with pytest.raises(ValueError, match="ngắn hơn frame_range"):
        dataset[0]


def test_getitem_full_clip_khong_frame_range(tmp_path):
    """Row clip đầy đủ (không `frame_range`, vd holdout) → dùng trọn tensor cache."""
    row = {**ROW1, "id": "clip-day-du"}
    del row["frame_range"]
    manifest = tmp_path / "full.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_cache(cache_dir, ROW1, length=15)  # key tính theo audio_path (giống ROW1)
    dataset = SlotDataset(manifest, CONFIG, F0_STATS, cache_dir=cache_dir)
    item = dataset[0]
    assert item["mel"].shape == (15, 20)
    assert item["f0"].shape == (15,) and item["vuv"].shape == (15,)


def test_min_syllables_loc_row_n_1(tmp_path):
    """`min_syllables=2` (train v2): lọc row 1 âm tiết — MPS abort conv backward với N=1."""
    one_syllable = {**ROW1, "id": "clip__s99", "audio_path": "/khong/ton/tai_n1.wav",
                    "durations": [5], "gaps": [], "seg_durations": [[2, 0, 2, 1]],
                    "frame_range": [0, 5], "syllables": ROW1["syllables"][:1]}
    manifest = tmp_path / "loc.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                                  for row in (ROW1, ROW2, one_syllable)), encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for row in (ROW1, ROW2, one_syllable):
        _write_cache(cache_dir, row)
    full = SlotDataset(manifest, CONFIG, F0_STATS, cache_dir=cache_dir)
    filtered = SlotDataset(manifest, CONFIG, F0_STATS, cache_dir=cache_dir, min_syllables=2)
    assert len(full) == 3 and len(filtered) == 2
    assert all(len(item["durations"]) >= 2 for item in (filtered[0], filtered[1]))


def test_collate_pad_frame_stack_syllable(dataset):
    batch = collate_slot_batch([dataset[0], dataset[1]])
    assert batch["id"] == ["clip__s00", "clip__s01"]
    assert batch["mel"].shape == (2, 13, 20)  # Tmax = 13
    assert batch["frame_lengths"].tolist() == [13, 12]
    assert torch.equal(batch["mel"][1, 12:], torch.zeros(1, 20))  # pad 0
    assert batch["f0"].shape == (2, 13) and batch["vuv"].shape == (2, 13)
    assert batch["energy"].shape == (2, 13)
    assert batch["durations"].shape == (2, 2) and batch["durations"][0].tolist() == [5, 6]
    assert batch["durations"][1].tolist() == [4, 5]
    assert batch["gaps"].shape == (2, 2) and batch["lead"].shape == (2,)
    assert batch["features"].shape == (2, 2, 7)
    assert batch["seg_durations"].shape == (2, 2, 4)
    assert batch["seg_ratio"].shape == (2, 2, 4)
    assert batch["f0_shape"].shape == (2, 2, 4, 4)
    assert batch["pause_gate"].shape == (2, 2)


def test_collate_khac_n_raise(dataset):
    item = dataset[0]
    narrow = {**item, "id": "hep",
              "features": item["features"][:1], "durations": item["durations"][:1],
              "gaps": item["gaps"][:1], "seg_durations": item["seg_durations"][:1],
              "seg_ratio": item["seg_ratio"][:1],
              "seg_ratio_mask": item["seg_ratio_mask"][:1],
              "f0_shape": item["f0_shape"][:1], "f0_weight": item["f0_weight"][:1],
              "seg_energy": item["seg_energy"][:1], "seg_vuv": item["seg_vuv"][:1],
              "pause_gate": item["pause_gate"][:1],
              "pause_log_gap": item["pause_log_gap"][:1],
              "pause_mask": item["pause_mask"][:1]}
    with pytest.raises(ValueError, match="cùng N"):
        collate_slot_batch([item, narrow])

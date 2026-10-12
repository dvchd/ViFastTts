"""Test `tools/cut_segments.py` + slice `frame_range` của `TtsDataset`."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from tools.cut_segments import cut_clip, cut_manifest


def test_cut_clip_uu_tien_pause_trong_cua_so():
    # min=10, max=20: [12, 3, 12, 4, 12] — pause sau âm tiết 2 (index 2) ở mốc 27 frame? Không:
    # cum = [0,12,15,27,31,43]. Từ 0: cửa sổ trong [10,20] → các end ∈ {1(12)}; 2(15); 3(27>20 loại).
    # pause sau âm tiết 1 (index 1) → end=2 (15 frame) là pause xa nhất.
    cuts = cut_clip([12, 3, 12, 4, 12], {1}, min_frames=10, max_frames=20)
    assert cuts == [(0, 2), (2, 4), (4, 5)]
    # Không pause: cắt xa nhất trong cửa sổ → end=2 (15) từ 0; rồi 27-15=12 → end=4 (31-15=16)
    cuts2 = cut_clip([12, 3, 12, 4, 12], set(), min_frames=10, max_frames=20)
    assert cuts2 == [(0, 2), (2, 4), (4, 5)]


def test_cut_clip_gop_duoi_ngan():
    # cum = [0,12,24,29]; từ 0 chọn end=2 (24) [trong 10..20? 24>20 → end=1 12]. Let's check:
    # từ 0: cửa sổ ≤20 → end=1 (12). Từ 1: 12→ 17 (end=3?) cum[3]-cum[1]=17 ∈ [10,20] → end=3 (29-12=17).
    # → [(0,1),(1,3)] — đuôi không ngắn. Ca khác:
    cuts = cut_clip([6, 6, 2], set(), min_frames=10, max_frames=20)
    # từ 0: end=1 (6<10, chưa); end=2 (12 ≥10 ✓, ≤20) → end=2. đuôi (2 frames) < 10 → gộp vào trước nếu ≤20:
    # 2→ tổng 14 ≤ 20 ✓ → [(0,3)]
    assert cuts == [(0, 3)]


def test_cut_manifest_xuat_segment_co_frame_range(tmp_path: Path):
    row = {
        "id": "v1_0001", "audio_path": "/tmp/khong_can_audio.wav", "text": "a b c",
        "syllables": [{"raw": x} for x in ("a", "b", "c")],
        "durations": [12, 3, 12],
        "pauses": [{"after": 1, "seconds": 0.2}],
    }
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    stats = cut_manifest(manifest, tmp_path / "m_seg.jsonl", min_frames=10, max_frames=20)
    assert stats["segments"] == 2 and stats["clips_split"] == 1
    segments = [json.loads(line) for line in (tmp_path / "m_seg.jsonl").read_text(encoding="utf-8").splitlines()]
    assert segments[0]["frame_range"] == [0, 15] and segments[0]["durations"] == [12, 3]
    assert segments[1]["frame_range"] == [15, 27] and segments[1]["durations"] == [12]
    assert segments[0]["parent"] == "v1_0001" and segments[0]["id"] == "v1_0001__s00"
    _ = segments[1].get("pauses", True)  # pauses không bắt buộc giữ trong row segment


def test_tts_dataset_slice_theo_frame_range(tmp_path: Path):
    import pytest

    pytest.importorskip("torch")

    def _wav(path: Path, seconds: float = 1.0) -> None:
        time_axis = np.arange(int(seconds * 24000)) / 24000
        tone = 0.3 * np.sin(2 * np.pi * 150.0 * time_axis)
        pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(24000)
            fh.writeframes(pcm.tobytes())

    from vifasttts.data.dataset import TtsDataset

    wav_path = tmp_path / "s1.wav"
    _wav(wav_path)
    config = {
        "sample_rate": 24000,
        "mel": {"n_fft": 256, "win_length": 256, "hop_length": 64, "n_mels": 20, "f_min": 0, "f_max": 12000},
        "f0": {"method": "autocorr"},
    }
    syllables = [
        {"raw": "chào", "normalized": "chào", "orthography": {"onset": "ch", "medial": None, "nucleus": "a", "coda": "o", "tone_location": "a"},
         "abstract": {"onset": "CH", "medial": "NONE", "nucleus": "A", "coda": "W", "tone": "HUYEN"}, "status": "VALID",
         "boundaries": {"word_end": True, "phrase_end": False, "sentence_end": False}, "punctuation": None},
        {"raw": "bạn", "normalized": "bạn", "orthography": {"onset": "b", "medial": None, "nucleus": "a", "coda": "n", "tone_location": "a"},
         "abstract": {"onset": "B", "medial": "NONE", "nucleus": "A", "coda": "N", "tone": "NANG"}, "status": "VALID",
         "boundaries": {"word_end": True, "phrase_end": False, "sentence_end": True}, "punctuation": "."},
    ]
    row = {
        "id": "s1", "audio_path": str(wav_path), "text": "chào bạn",
        "syllables": syllables, "frame_range": [10, 60], "durations": [20, 30],
    }
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset = TtsDataset(manifest, config)
    item = dataset[0]
    full_frames = 1 + 24000 // 64
    assert item["mel"].size(0) == 50
    assert item["durations"].tolist() == [20, 30]
    assert item["features"].size(0) == 2
    assert item["mel"].size(0) < full_frames

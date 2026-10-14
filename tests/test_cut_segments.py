"""Test `tools/cut_segments.py` + slice `frame_range` của `TtsDataset`."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from tools.cut_segments import cut_clip, cut_manifest


def _spans(durations: list[int], gaps: list[int] | None = None, lead: int = 0) -> list[tuple[int, int]]:
    """Dựng spans tuyệt đối từ durations + gaps (test helper)."""
    gaps = gaps or [0] * len(durations)
    spans: list[tuple[int, int]] = []
    cursor = lead
    for i, duration in enumerate(durations):
        spans.append((cursor, cursor + duration))
        cursor += duration + gaps[i]
    return spans


def test_cut_clip_uu_tien_pause_trong_cua_so():
    # spans từ [12, 3, 12, 4, 12] → [(0,12),(12,15),(15,27),(27,31),(31,43)].
    # Từ 0: cửa sổ trong [10,20] → các end ∈ {1(12), 2(15)}; pause sau âm tiết 1 → end=2 (15 frame).
    cuts = cut_clip(_spans([12, 3, 12, 4, 12]), {1}, min_frames=10, max_frames=20)
    assert cuts == [(0, 2), (2, 4), (4, 5)]
    # Không pause: cắt xa nhất trong cửa sổ → end=2 (15) từ 0; rồi 27-15=12 → end=4 (31-15=16)
    cuts2 = cut_clip(_spans([12, 3, 12, 4, 12]), set(), min_frames=10, max_frames=20)
    assert cuts2 == [(0, 2), (2, 4), (4, 5)]


def test_cut_clip_gop_duoi_ngan():
    # spans [6,6,2] → [(0,6),(6,12),(12,14)]; từ 0: end=2 (12 ≥10) → window gồm end=3 (14, end==count)
    # → chọn end=3 (xa nhất) → [(0,3)] (đuôi 2 frame gộp vào segment trước).
    cuts = cut_clip(_spans([6, 6, 2]), set(), min_frames=10, max_frames=20)
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
    stats = cut_manifest(manifest, tmp_path / "m_seg.jsonl", min_frames=10, max_frames=20, floor_frames=0)
    assert stats["segments"] == 2 and stats["clips_split"] == 1
    segments = [json.loads(line) for line in (tmp_path / "m_seg.jsonl").read_text(encoding="utf-8").splitlines()]
    assert segments[0]["frame_range"] == [0, 15] and segments[0]["durations"] == [12, 3]
    assert segments[1]["frame_range"] == [15, 27] and segments[1]["durations"] == [12]
    assert segments[0]["parent"] == "v1_0001" and segments[0]["id"] == "v1_0001__s00"
    _ = segments[1].get("pauses", True)  # pauses không bắt buộc giữ trong row segment


def test_cut_manifest_tach_gaps_va_lead(tmp_path: Path):
    # durations [12,3,12] + gaps [4,10,5] + lead 7 → spans [(7,19),(23,26),(36,48)] (T=53).
    # A2 (§43.1): segment **gồm pause đuôi** ⇒ segment (0,2) frame_range [7,36] (bỏ lead 7 nhưng
    # GIỮ gap 10 sau âm tiết 1 = pause ở đúng chỗ cắt) và gaps [4, 10]; segment (2,3) [36, 53]
    # (giữ luôn 5 frame lặng đuôi clip). Bất biến: T == Σdur + Σgap cho MỌI row.
    row = {
        "id": "v1_0002", "audio_path": "/tmp/khong_can_audio.wav", "text": "a b c",
        "syllables": [{"raw": x} for x in ("a", "b", "c")],
        "durations": [12, 3, 12], "gaps": [4, 10, 5], "lead_frames": 7,
        "pauses": [{"after": 1, "seconds": 0.2}],
    }
    manifest = tmp_path / "m2.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    cut_manifest(manifest, tmp_path / "m2_seg.jsonl", min_frames=10, max_frames=20, floor_frames=0)
    segments = [json.loads(line) for line in (tmp_path / "m2_seg.jsonl").read_text(encoding="utf-8").splitlines()]
    assert segments[0]["frame_range"] == [7, 36]
    assert segments[0]["durations"] == [12, 3] and segments[0]["gaps"] == [4, 10]
    assert segments[0]["lead_frames"] == 0  # toạ độ row tính từ âm tiết đầu ⇒ lead tương đối = 0
    assert segments[1]["frame_range"] == [36, 53]
    assert segments[1]["durations"] == [12] and segments[1]["gaps"] == [5]
    # Bất biến (A2): sum(durations) + sum(gaps) == span từng segment; len(gaps) == số âm tiết
    for segment in segments:
        span = segment["frame_range"][1] - segment["frame_range"][0]
        assert sum(segment["durations"]) + sum(segment["gaps"]) == span
        assert len(segment["gaps"]) == len(segment["durations"])
    # Hai segment liền kề khít nhau (không chồng, không hở): 36 = 36
    assert segments[0]["frame_range"][1] == segments[1]["frame_range"][0]
    # Sàn `floor_frames` áp trên **độ dài hữu thanh**: segment [36,53] dài 17 frame nhưng Σdur = 12
    # < 13 ⇒ bị bỏ (gap đuôi dài KHÔNG "cứu" mảnh quá ít tiếng)
    cut_manifest(manifest, tmp_path / "m3_seg.jsonl", min_frames=10, max_frames=20, floor_frames=13)
    kept = [json.loads(line) for line in (tmp_path / "m3_seg.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(kept) == 1 and kept[0]["durations"] == [12, 3]  # đuôi vụn vẫn bị bỏ như trước


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
    assert item["spans"].tolist() == [20, 30]
    assert item["features"].size(0) == 2
    assert item["mel"].size(0) < full_frames


def test_tts_dataset_doc_gaps_va_lead(tmp_path: Path):
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
    # Segment: gaps chỉ còn khoảng lặng nội bộ (len = n-1); span = 60 = 20+10+30.
    row_segment = {
        "id": "s1", "audio_path": str(wav_path), "text": "chào bạn",
        "syllables": syllables, "frame_range": [10, 70], "durations": [20, 30], "gaps": [10],
    }
    manifest = tmp_path / "m_seg.jsonl"
    manifest.write_text(json.dumps(row_segment, ensure_ascii=False) + "\n", encoding="utf-8")
    item = TtsDataset(manifest, config)[0]
    assert item["durations"].tolist() == [20, 30]
    assert item["spans"].tolist() == [30, 30]
    assert item["mel"].size(0) == 60

    # Clip đầy đủ: lead gộp vào span đầu khi mở rộng (duration vẫn là phần hữu thanh).
    full_frames = 1 + 24000 // 64
    lead = full_frames - (20 + 30 + 10 + 5)
    row_clip = {
        "id": "s2", "audio_path": str(wav_path), "text": "chào bạn",
        "syllables": syllables, "durations": [20, 30], "gaps": [10, 5], "lead_frames": lead,
    }
    manifest2 = tmp_path / "m_clip.jsonl"
    manifest2.write_text(json.dumps(row_clip, ensure_ascii=False) + "\n", encoding="utf-8")
    item2 = TtsDataset(manifest2, config)[0]
    assert item2["durations"].tolist() == [20, 30]
    assert item2["spans"].tolist() == [lead + 30, 35]
    assert item2["mel"].size(0) == full_frames


def test_tail_cap_chi_ap_doan_cuoi_clip(tmp_path):
    """A1: trần lặng đuôi — đoạn cuối clip bị cắt gap cuối + frame_range[1]; chỗ cắt không đổi."""
    from tools.cut_segments import cut_manifest

    row = {"id": "clip", "audio_path": "a.wav", "sample_rate": 24000,
           "syllables": [{"raw": f"s{i}"} for i in range(4)],
           "durations": [20, 20, 20, 20], "gaps": [300, 300, 300, 300],
           "pauses": [{"after": 1}]}
    manifest = tmp_path / "in.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    plain = tmp_path / "plain_seg.jsonl"
    capped = tmp_path / "capped_seg.jsonl"
    cut_manifest(manifest, plain, min_frames=10, max_frames=200, floor_frames=5)
    cut_manifest(manifest, capped, min_frames=10, max_frames=200, floor_frames=5, tail_cap_frames=67)
    plain_rows = [json.loads(l) for l in plain.read_text().splitlines()]
    capped_rows = [json.loads(l) for l in capped.read_text().splitlines()]
    assert len(plain_rows) == len(capped_rows) >= 2
    for a, b in zip(plain_rows, capped_rows):
        assert a["id"] == b["id"] and a["syllables"] == b["syllables"] and a["durations"] == b["durations"]
        is_last_of_clip = a["id"].endswith(f"__s{len(plain_rows) - 1:02d}")
        if is_last_of_clip:
            assert b["gaps"][-1] == 67 and a["gaps"][-1] == 300
            assert a["frame_range"][1] - b["frame_range"][1] == 233  # cắt cả target
        else:
            assert a["gaps"] == b["gaps"] and a["frame_range"] == b["frame_range"]
        assert b["frame_range"][1] - b["frame_range"][0] == sum(b["durations"]) + sum(b["gaps"])

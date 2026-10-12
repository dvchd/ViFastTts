"""Test helper của tools/diarize_filter.py (không cần transformers/model)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # cho phép chạy cả `pytest` trần
    sys.path.insert(0, str(ROOT))

from tools.diarize_filter import (
    audio_path_of,
    build_anchor_clips,
    choose_video_anchors,
    load_durations,
)


def test_audio_path_of_prefers_existing_relative(tmp_path: Path) -> None:
    """Đường dẫn tuyệt đối giữ nguyên; tương đối không tồn tại → ghép --audio-root."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"")
    assert audio_path_of({"audio_path": str(wav)}, tmp_path) == wav
    assert audio_path_of({"audio_path": "missing/x.wav"}, tmp_path) == tmp_path / "missing/x.wav"


def test_build_anchor_clips_picks_closest_to_12s() -> None:
    """Chọn 2 clip gần 12s nhất trong video làm anchor."""
    rows = [
        {"id": "V_0001"}, {"id": "V_0002"}, {"id": "V_0003"}, {"id": "W_0001"},
    ]
    durations = {"V_0001": 3.0, "V_0002": 11.5, "V_0003": 19.0, "W_0001": 12.0}
    assert build_anchor_clips(rows, "V", [], durations) == ["V_0002", "V_0003"]
    assert build_anchor_clips(rows, None, ["W_0001", "V_0001"], durations) == ["W_0001", "V_0001"]


def test_choose_video_anchors_filters_and_orders() -> None:
    """Chỉ nhận clip sạch (other_s thấp, owner_s đủ, unlabeled thấp), ưu tiên ~12s."""
    pass_a = {
        "A_1": {"video": "A", "other_s": 0.0, "owner_s": 5.0, "unlabeled_frac": 0.0, "duration": 18.0},
        "A_2": {"video": "A", "other_s": 0.0, "owner_s": 4.0, "unlabeled_frac": 0.1, "duration": 11.0},
        "A_3": {"video": "A", "other_s": 0.8, "owner_s": 4.0, "unlabeled_frac": 0.0, "duration": 12.0},
        "A_4": {"video": "A", "other_s": 0.0, "owner_s": 1.0, "unlabeled_frac": 0.0, "duration": 12.0},
        "B_1": {"video": "B", "other_s": 0.0, "owner_s": 6.0, "unlabeled_frac": 0.2, "duration": 12.0},
    }
    chosen = choose_video_anchors(pass_a)
    assert chosen["A"] == ["A_2", "A_1"]
    assert chosen["B"] == ["B_1"]


def test_load_durations(tmp_path: Path) -> None:
    """Đọc thời lượng clip từ clips.jsonl; không có file thì trả {}."""
    clips = tmp_path / "clips.jsonl"
    clips.write_text(
        json.dumps({"id": "V_0001", "duration": 12.5}) + "\n"
        + json.dumps({"id": "V_0002", "duration": 3.0}) + "\n",
        encoding="utf-8",
    )
    assert load_durations(tmp_path / "manifest.jsonl", str(clips)) == {"V_0001": 12.5, "V_0002": 3.0}
    empty = tmp_path / "khong-co-clips"
    empty.mkdir()
    assert load_durations(empty / "manifest.jsonl", None) == {}

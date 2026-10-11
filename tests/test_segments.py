"""Test Segment Expander và phân bổ duration theo role."""
from __future__ import annotations

from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.segments import (
    distribute_frames,
    expand_syllable,
    expand_text,
    segment_summary,
)
from vifasttts.frontend.syllable import parse_syllable


def test_expand_quoc_roles_and_units():
    segments = expand_syllable(parse_syllable("quốc"))
    assert [segment.role for segment in segments] == ["onset", "medial", "nucleus", "coda", "tone"]
    assert [segment.unit for segment in segments] == ["K", "W", "Ô", "C", "SAC"]
    assert segments[0].phoneme == "k"
    assert segments[1].phoneme == "w"
    assert segments[2].phoneme == "o"
    assert segments[-1].weight == 0.0


def test_expand_diphthong_splits_into_two_targets():
    segments = expand_syllable(parse_syllable("yêu"))
    nucleus = [segment for segment in segments if segment.role == "nucleus"]
    assert [segment.phoneme for segment in nucleus] == ["i", "ə"]
    assert [segment.part for segment in nucleus] == [0, 1]
    assert [segment.unit for segment in nucleus] == ["IÊ", "IÊ"]
    coda = [segment for segment in segments if segment.role == "coda"]
    assert coda[0].phoneme == "w"


def test_expand_oo_long_weight_and_dialects():
    north = expand_syllable(parse_syllable("xoong"))
    nucleus = next(segment for segment in north if segment.role == "nucleus")
    assert nucleus.unit == "OO"
    assert nucleus.phoneme == "ɔː"
    assert nucleus.weight > 1.0

    south = expand_syllable(parse_syllable("xoong"), "south")
    assert next(segment for segment in south if segment.role == "nucleus").phoneme == "ɔ"


def test_expand_narrow_hanoi_fronting():
    narrow = expand_syllable(parse_syllable("cứu"), "north", style="narrow")
    nucleus = next(segment for segment in narrow if segment.role == "nucleus")
    assert nucleus.phoneme == "i"

    broad = expand_syllable(parse_syllable("cứu"), "north")
    assert next(segment for segment in broad if segment.role == "nucleus").phoneme == "ɯ"

    # "ươu" narrow Bắc: một đích [iəw] thay vì tách ɯ+ə.
    huou = expand_syllable(parse_syllable("hươu"), "north", style="narrow")
    nucleus_units = [segment for segment in huou if segment.role == "nucleus"]
    assert len(nucleus_units) == 1 and nucleus_units[0].phoneme == "iə"


def test_expand_fallback_is_empty():
    items = parse_text("Hello")
    assert expand_syllable(items[0]) == []
    assert expand_text(items) == [[]]


def test_expand_text_counts():
    items = parse_text("Chào bạn hôm nay")
    expanded = expand_text(items)
    assert len(expanded) == 4
    assert all(segments[-1].role == "tone" for segments in expanded)


def test_distribute_frames_invariants():
    segments = expand_syllable(parse_syllable("khuỷu"))
    tone_index = next(i for i, segment in enumerate(segments) if segment.role == "tone")
    for frames in (0, 1, 2, 5, 7, 40, 101):
        distribution = distribute_frames(frames, segments)
        assert sum(distribution) == frames
        assert distribution[tone_index] == 0
    distribution = distribute_frames(40, segments)
    positives = [i for i, segment in enumerate(segments) if segment.weight > 0]
    assert all(distribution[i] >= 1 for i in positives)
    small = distribute_frames(3, segments)
    assert sum(small) == 3 and small.count(1) == 3


def test_distribute_frames_proportional():
    segments = expand_syllable(parse_syllable("ba"))
    distribution = distribute_frames(100, segments)
    roles = {segment.role: distribution[i] for i, segment in enumerate(segments)}
    assert roles["nucleus"] > roles["onset"] > 0


def test_segment_summary():
    summary = segment_summary(expand_syllable(parse_syllable("quốc")))
    assert summary["count"] == 5
    assert "k w o k" in str(summary["phonemes"])

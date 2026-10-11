"""Test benchmark biểu diễn frontend."""
from __future__ import annotations

from vifasttts.frontend.benchmark import format_stats, representation_stats


def test_stats_basic():
    stats = representation_stats(["Chào bạn hôm nay", "Thử nghiệm"])
    assert stats["sentences"] == 2
    assert stats["tokens"] == 6
    assert stats["fallback"] == 0
    representations = stats["representations"]
    assert representations["slot"]["units_total"] == 5 * 6
    assert representations["rime"]["units_total"] == 2 * 6
    assert representations["phoneme"]["units_per_syllable"] > 2
    assert representations["char"]["units_per_syllable"] >= 1


def test_fallback_counted():
    stats = representation_stats(["Hello", "ba"])
    assert stats["tokens"] == 2
    assert stats["fallback"] == 1


def test_format_stats_smoke():
    text = format_stats(representation_stats(["ba"]))
    assert "slot" in text and "âm tiết" in text

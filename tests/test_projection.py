"""Test phép chiếu vần (rime projection) từ AbstractSyllable."""
from __future__ import annotations

from vifasttts.frontend.syllable import parse_syllable


def test_rime_projection_components():
    a = parse_syllable("quốc").abstract
    assert a.rime == ("W", "Ô", "C")
    assert a.rime_key == "W+Ô+C"
    # Gộp vần là tuple nên vẫn khôi phục nguyên vẹn 5 thành phần.
    assert (a.onset, *a.rime) == (a.onset, a.medial, a.nucleus, a.coda)


def test_rime_projection_hashable_and_complete():
    rimes = set()
    for word in ("hoà", "thuỷ", "khuỷu", "yêu", "xoong", "đường", "khuya", "quốc"):
        a = parse_syllable(word).abstract
        assert len(a.rime) == 3
        assert all(isinstance(part, str) and part for part in a.rime)
        rimes.add(a.rime)
    # Các âm tiết khác vần phải có rime khác nhau.
    assert len(rimes) == 8


def test_rime_key_stable():
    assert parse_syllable("xoong").abstract.rime_key == "NONE+OO+NG"
    assert parse_syllable("xong").abstract.rime_key == "NONE+O+NG"
    assert parse_syllable("thuỷ").abstract.rime_key == "W+I+NONE"

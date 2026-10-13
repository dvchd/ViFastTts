"""Test `tools/r2_weak_labels.py`: phân tích cấu trúc phone → role (R2)."""

from __future__ import annotations

from tools.r2_weak_labels import analyze, is_vowel


def _phones(*labels: str) -> list[tuple[float, float, str]]:
    return [(float(i), float(i + 1), label) for i, label in enumerate(labels)]


def test_is_vowel():
    assert is_vowel("aː˨˨")
    assert not is_vowel("k")
    assert not is_vowel("ʔ")
    assert not is_vowel("w")


def test_analyze_onset_nucleus_coda():
    result = analyze(_phones("h", "aː˨˨", "n"), "han")
    assert result["parse_ok"] and result["hybrid_ok"] and result["ipa_full_ok"]
    assert result["onset_labels"] == ["h"]
    assert result["nucleus_labels"] == ["aː˨˨"]
    assert result["coda_labels"] == ["n"]


def test_analyze_medial_va_glottal():
    result = analyze(_phones("h", "w", "aː˨˨"), "hoa")
    assert result["medial_labels"] == ["w"] and result["ipa_full_ok"]

    result = analyze(_phones("ʔ", "aː˨˨"), "a")
    assert result["onset_labels"] == []
    assert result["nucleus_labels"] == ["ʔ", "aː˨˨"]
    assert result["ipa_full_ok"]


def test_analyze_coda_tac_thieu_phone():
    result = analyze(_phones("k", "aː˦˥"), "các")
    assert not result["ipa_full_ok"]
    assert result["fail"] == "coda_thieu"
    assert result["hybrid_ok"]  # vẫn có biên onset|rhyme


def test_analyze_khong_nguyen_am():
    result = analyze(_phones("t", "w"), "tu")
    assert result["fail"] == "khong_nguyen_am"
    assert "boundary" not in result

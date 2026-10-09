"""Extended tone placement and phonotactics (from goxviet/VKey research).

Modern (hoà) vs classic (hòa), triphthongs, qu/gi clusters,
stop-coda tone restriction, invalid combos.
"""

import pytest
from vifasttts.frontend.normalizer import canonicalize_vietnamese_tokens
from vifasttts.frontend.syllable import parse_syllable, safe_parse_syllable


@pytest.mark.parametrize("source,canonical", [
    # oa/oe modern: tone on second vowel
    ("hòa", "hoà"), ("hoà", "hoà"),
    ("hóa", "hoá"), ("hoá", "hoá"),
    ("khỏe", "khoẻ"), ("khoẻ", "khoẻ"),
    # uy modern: tone on y
    ("thủy", "thuỷ"), ("thuỷ", "thuỷ"),
    ("tùy", "tuỳ"), ("tuỳ", "tuỳ"),
    ("quý", "quý"),
    # neighboring valid, unchanged
    ("thuế", "thuế"), ("người", "người"), ("tiếng", "tiếng"),
    ("được", "được"), ("nguyễn", "nguyễn"),
])
def test_modern_canonical(source, canonical):
    assert canonicalize_vietnamese_tokens(source) == canonical
    assert parse_syllable(source).abstract == parse_syllable(canonical).abstract


@pytest.mark.parametrize("word,expected_onset", [
    ("trăm", "TR"), ("chăm", "CH"), ("sương", "S"), ("xương", "X"),
    ("da", "D"), ("gia", "GI"), ("ra", "R"),
    ("nghe", "NG"), ("nghệ", "NG"),
    ("qua", "K"), ("quả", "K"),
])
def test_onset_distinct(word, expected_onset):
    assert parse_syllable(word).abstract.onset == expected_onset


@pytest.mark.parametrize("word", [
    "ách", "ếch", "các", "mật", "cấp",
    "quyết", "kiếp", "thác",
])
def test_stop_coda_valid_with_sac_nang(word):
    assert parse_syllable(word) is not None


@pytest.mark.parametrize("word", [
    "bap", "bat", "bac", "bach",  # stop coda + ngang -> invalid
    "ach",  # no onset, stop + ngang
])
def test_stop_coda_rejects_invalid_tone(word):
    assert safe_parse_syllable(word) is None
    with pytest.raises(ValueError):
        parse_syllable(word)


@pytest.mark.parametrize("word", [
    "", "123", "xyz", "f", "qu", "ea",
    "sea", "you", "yoke",
    # Note: "ou"/"yo" parse permissively as o+u / y+o (productive combos,
    # lexicon check before training per rules.py) — not asserted here.
])
def test_foreign_invalid_fallback(word):
    assert safe_parse_syllable(word) is None


def test_triphthong_uye():
    a = parse_syllable("nguyễn").abstract
    assert a.nucleus == "IÊ"
    assert a.onset == "NG"


def test_qu_medial():
    a = parse_syllable("quốc").abstract
    assert a.onset == "K"
    assert a.medial == "W"


def test_gi_cluster():
    a = parse_syllable("gia").abstract
    assert a.onset == "GI"

import pytest
from vifasttts.frontend.normalizer import canonicalize_vietnamese_tokens
from vifasttts.frontend.syllable import parse_syllable


@pytest.mark.parametrize("source,canonical", [
    ("hòa", "hoà"), ("hoà", "hoà"),
    ("hóa", "hoá"), ("hoá", "hoá"),
    ("hỏa", "hoả"), ("hõa", "hoã"), ("họa", "hoạ"),
    ("thủy", "thuỷ"), ("thuỷ", "thuỷ"),
    ("tùy", "tuỳ"), ("tuỳ", "tuỳ"),
    ("quý", "quý"), ("thuế", "thuế"),
    ("người", "người"), ("tiếng", "tiếng"),
])
def test_canonical_tone_position(source, canonical):
    assert canonicalize_vietnamese_tokens(source) == canonical
    assert parse_syllable(source).abstract == parse_syllable(canonical).abstract


def test_sentence_case_preserved():
    assert canonicalize_vietnamese_tokens("Hòa bình") == "Hoà bình"

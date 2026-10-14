import pytest

from vifasttts.frontend.english_fallback import (
    ACRONYM_READINGS,
    CASE_INSENSITIVE_ENGLISH,
    EXACT_PRODUCT_NAMES,
    lookup_case_insensitive_english,
    lookup_exact_acronym,
    lookup_product_name,
)
from vifasttts.frontend.syllable import safe_parse_syllable


@pytest.mark.parametrize('token,reading', sorted(ACRONYM_READINGS.items()))
def test_acronyms_exact(token, reading):
    assert lookup_exact_acronym(token) == reading
    assert lookup_exact_acronym(token.lower()) is None

@pytest.mark.parametrize('token,reading', sorted(CASE_INSENSITIVE_ENGLISH.items()))
def test_english_words_case_insensitive(token, reading):
    assert lookup_case_insensitive_english(token) == reading
    assert lookup_case_insensitive_english(token.upper()) == reading

@pytest.mark.parametrize('token,reading', sorted(EXACT_PRODUCT_NAMES.items()))
def test_products_exact(token, reading):
    assert lookup_product_name(token) == reading

@pytest.mark.parametrize('reading', sorted({
    *ACRONYM_READINGS.values(),
    *CASE_INSENSITIVE_ENGLISH.values(),
    *EXACT_PRODUCT_NAMES.values(),
}))
def test_readings_are_valid_vietnamese_syllables(reading):
    """Mọi reading phải parse được thành âm tiết Việt, nếu không pipeline
    sẽ rơi lại vào FALLBACK_REQUIRED ngay sau khi chuẩn hoá."""
    for word in reading.split():
        assert safe_parse_syllable(word) is not None, f"{word!r} trong {reading!r}"



def test_oov_danh_van_moi_token_chu():
    """P0.2 (14/10): "B"/"wav"/"COVID-19" phải được đánh vần, KHÔNG rơi về fallback "a"."""
    from vifasttts.frontend.oov import resolve_oov_token

    assert resolve_oov_token("B") == "bê"
    assert resolve_oov_token("wav") == "vê kép a vê"
    assert resolve_oov_token("COVID-19") == "xê o vê i dê 19"
    assert resolve_oov_token("TP.HCM") == "tê pê hát xê em"

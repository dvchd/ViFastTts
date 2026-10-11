"""Tính khả nghịch chính tả: chữ canonical ↔ (bộ âm vị + chính tả dư).

Định nghĩa:
- Hàm: mỗi nguồn → đúng một đích.
- Đơn ánh: hai nguồn khác nhau không cho cùng đích.
- Toàn ánh: mọi đích đều có ít nhất một nguồn.
- Song ánh = đơn ánh + toàn ánh → tồn tại ánh xạ ngược (round-trip được).

Với tiếng Việt:
- chữ canonical → bộ âm vị: TOÀN ÁNH (mọi tuple hiện thực hoá đều có chữ, xem
  `test_parser_matrix.py`) nhưng KHÔNG ĐƠN ÁNH vì đồng âm/chính tả tự do
  (qui/quy, mĩ/mỹ, í/ý cùng tuple).
- bộ âm vị thuần → chữ: không phải hàm (một tuple ứng nhiều chữ).
- chữ canonical ↔ (bộ âm vị + chính tả dư) = `ParsedSyllable`: SONG ÁNH —
  compose trả lại đúng chữ gốc, parse trả lại đúng tuple.
- `tai`/`tay` KHÔNG còn là đồng âm: "ai" đọc a dài [aː], "ay" đọc ă ngắn
  (Help:IPA/Vietnamese), nên hai tuple khác nhau — xem test dưới.
"""
from __future__ import annotations

import pytest

from vifasttts.frontend.normalizer import canonicalize_vietnamese_tokens
from vifasttts.frontend.orthography import compose_canonical
from vifasttts.frontend.syllable import parse_syllable


def _compose(parsed) -> str:
    return compose_canonical(
        parsed.orthography.onset, parsed.orthography.medial,
        parsed.orthography.nucleus, parsed.orthography.coda, parsed.abstract.tone,
    )


CORPUS = ["khuya", "khuỷu", "ngoằn", "oai", "oanh", "uông", "uôm", "ươm", "ướp", "yêu", "yên", "uyên", "quán", "quạnh", "quắt", "quang", "thuở", "huơ", "khuấy", "khuynh", "xoong", "boong", "moóc", "xoài", "khoai", "ngoài", "khoắng", "hoẵng", "toạch", "ướt", "khuất", "quốc", "quyến", "nguyệt", "khuyết", "chuyển", "chuyện", "duyệt", "tuyết", "thuyền", "quyền", "nguệch", "oẹ", "loè", "khỏe", "ạch", "ếch", "ích", "hộp", "hợp", "nghiêng", "nghiệng", "nguyễn", "giếng", "giết", "giêng", "giữ", "già", "gì", "gìn", "đường", "được", "tưởng", "ngưỡng", "cưỡng", "quỳnh", "huỳnh", "oắt", "oăn", "oóc", "éc", "ếc", "âu", "ấu", "ầu", "ếu", "ều", "lau", "láu", "lâu", "lều"]


@pytest.mark.parametrize("word", CORPUS)
def test_canonical_roundtrip(word):
    """chữ canonical → parse → compose trả lại đúng chữ đó (và ổn định)."""
    parsed = parse_syllable(word)
    back = _compose(parsed)
    assert back == canonicalize_vietnamese_tokens(word)
    assert _compose(parse_syllable(back)) == back


@pytest.mark.parametrize("variant,canonical", [
    ("hòa", "hoà"),
    ("thủy", "thuỷ"),
    ("tùy", "tuỳ"),
    ("xòong", "xoòng"),
])
def test_old_style_variants_normalize_without_losing_phonology(variant, canonical):
    """Lối viết dấu cũ được chuẩn hoá về canonical, abstract không đổi."""
    parsed = parse_syllable(variant)
    assert _compose(parsed) == canonical
    assert parsed.abstract == parse_syllable(canonical).abstract


@pytest.mark.parametrize("first,second", [
    ("qui", "quy"),
    ("mĩ", "mỹ"),
    ("í", "ý"),
])
def test_free_spelling_variants_share_tuple_but_spelling_is_preserved(first, second):
    """Biến thể chính tả tự do (i/y) dùng chung tuple, nhưng chữ vẫn khả nghịch
    nhờ tầng chính tả dư."""
    parsed_first, parsed_second = parse_syllable(first), parse_syllable(second)
    assert parsed_first.abstract == parsed_second.abstract
    assert _compose(parsed_first) == first
    assert _compose(parsed_second) == second


@pytest.mark.parametrize("long_vowel, short_vowel", [
    ("tai", "tay"),
    ("mai", "may"),
    ("rao", "rau"),
    ("quai", "quay"),
])
def test_ai_ay_ao_au_are_distinct(long_vowel, short_vowel):
    """a dài [aː] (ai/ao) vs ă ngắn [a] (ay/au) — hai tuple khác nhau."""
    long_parsed, short_parsed = parse_syllable(long_vowel), parse_syllable(short_vowel)
    assert long_parsed.abstract != short_parsed.abstract
    assert long_parsed.abstract.nucleus == "A"
    assert short_parsed.abstract.nucleus == "Ă"
    assert _compose(long_parsed) == long_vowel
    assert _compose(short_parsed) == short_vowel


@pytest.mark.parametrize("first,second", [("xoong", "xong"), ("moóc", "móc")])
def test_oo_split_strengthens_injectivity(first, second):
    """Tách OO giúp `xoong` ≠ `xong` ngay ở tầng âm vị, chữ vẫn khả nghịch."""
    parsed_first, parsed_second = parse_syllable(first), parse_syllable(second)
    assert parsed_first.abstract != parsed_second.abstract
    assert parsed_first.abstract.nucleus == "OO"
    assert parsed_second.abstract.nucleus == "O"
    assert _compose(parsed_first) == first
    assert _compose(parsed_second) == second


@pytest.mark.parametrize("first,second", [
    ("tùy", "tuỳ"),
    ("hòa", "hoà"),
    ("thủy", "thuỷ"),
    ("hủy", "huỷ"),
])
def test_tone_position_variants_give_identical_model_input(first, second):
    """Vị trí dấu thanh không tồn tại ở tầng âm vị: hai cách ghi cho cùng 7 ID."""
    from vifasttts.frontend.vocab import ComponentVocabs

    vocabs = ComponentVocabs.default()
    parsed_first, parsed_second = parse_syllable(first), parse_syllable(second)
    assert parsed_first.abstract == parsed_second.abstract
    assert vocabs.encode(parsed_first) == vocabs.encode(parsed_second)


def test_canonical_tone_spelling_option():
    """Quy ước chữ cài đặt được; cả hai chế độ cho cùng tuple."""
    from vifasttts.frontend.normalizer import NormalizerOptions, normalize_text

    modern = NormalizerOptions(canonical_tone_spelling=True)
    preserve = NormalizerOptions(canonical_tone_spelling=False)
    assert normalize_text("tùy hòa thủy", options=modern) == "tuỳ hoà thuỷ"
    assert normalize_text("tùy hòa thủy", options=preserve) == "tùy hòa thủy"

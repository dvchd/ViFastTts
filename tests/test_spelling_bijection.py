"""Tính khả nghịch chính tả: chữ canonical ↔ (bộ âm vị + chính tả dư).

Định nghĩa:
- Hàm: mỗi nguồn → đúng một đích.
- Đơn ánh: hai nguồn khác nhau không cho cùng đích.
- Toàn ánh: mọi đích đều có ít nhất một nguồn.
- Song ánh = đơn ánh + toàn ánh → tồn tại ánh xạ ngược (round-trip được).

Với tiếng Việt:
- chữ canonical → bộ âm vị: TOÀN ÁNH (mọi tuple hiện thực hoá đều có chữ, xem
  `test_parser_matrix.py`) nhưng KHÔNG ĐƠN ÁNH vì đồng âm/chính tả tự do
  (tai/tay, qui/quy, í/ý cùng tuple).
- bộ âm vị thuần → chữ: không phải hàm (một tuple ứng nhiều chữ).
- chữ canonical ↔ (bộ âm vị + chính tả dư) = `ParsedSyllable`: SONG ÁNH —
  compose trả lại đúng chữ gốc, parse trả lại đúng tuple.
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
    ("tai", "tay"),
    ("qui", "quy"),
    ("í", "ý"),
])
def test_homophones_share_tuple_but_spelling_is_preserved(first, second):
    """Tuple không đơn ánh (đồng âm/chính tả tự do), nhưng chữ vẫn khả nghịch
    nhờ tầng chính tả dư — đúng ca `tai`/`tay`."""
    parsed_first, parsed_second = parse_syllable(first), parse_syllable(second)
    assert parsed_first.abstract == parsed_second.abstract
    assert _compose(parsed_first) == first
    assert _compose(parsed_second) == second


@pytest.mark.parametrize("first,second", [("xoong", "xong"), ("moóc", "móc")])
def test_oo_split_strengthens_injectivity(first, second):
    """Tách OO giúp `xoong` ≠ `xong` ngay ở tầng âm vị, chữ vẫn khả nghịch."""
    parsed_first, parsed_second = parse_syllable(first), parse_syllable(second)
    assert parsed_first.abstract != parsed_second.abstract
    assert parsed_first.abstract.nucleus == "OO"
    assert parsed_second.abstract.nucleus == "O"
    assert _compose(parsed_first) == first
    assert _compose(parsed_second) == second

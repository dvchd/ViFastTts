"""Edge case hồi quy cho parser và tokenizer (từ audit ngày 11/10/2026).

- "q" trần không phải âm đầu hợp lệ (chỉ có trong "qu").
- Chữ ngoài Latin (CJK, Cyrillic, Thái...) được giữ làm fallback, không bị nuốt.
- Unicode: NFC, dấu thanh kép, chữ hoa.
- Corpus từ thật/khó phải parse được (đã trừ các ca cố ý sai như stop coda + ngang).
"""
from __future__ import annotations

import pytest

from vifasttts.frontend.orthography import compose_canonical
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.syllable import parse_syllable, safe_parse_syllable
from vifasttts.types import ParseStatus


@pytest.mark.parametrize("word", ["qa", "qi", "qo", "qy", "qê", "qum", "qu"])
def test_bare_q_is_invalid(word):
    assert safe_parse_syllable(word) is None


@pytest.mark.parametrize("word", ["quả", "quy", "quốc", "quánh", "quạnh", "quyên", "quơ", "quân", "quay", "quê"])
def test_qu_forms_still_parse(word):
    assert safe_parse_syllable(word) is not None


def test_quoc_u_thuoc_am_doi_khong_phai_am_dem():
    """ThS Đinh Thị Hằng (Viện Ngôn ngữ học): /w/ không đứng trước âm tròn môi (u, ô, o) —
    "quốc" = /k/ + uô + /k/, chữ u thuộc nguyên âm đôi; đồng âm với "cuốc"."""
    a = parse_syllable("quốc").abstract
    assert (a.onset, a.medial, a.nucleus, a.coda) == ("K", "NONE", "UÔ", "C")
    assert a == parse_syllable("cuốc").abstract
    # "quất": â không tròn môi → u vẫn là âm đệm W (khác "quốc").
    b = parse_syllable("quất").abstract
    assert (b.medial, b.nucleus) == ("W", "Â")


@pytest.mark.parametrize("word", ["quo", "quom", "quò", "quu", "quoo", "quuô"])
def test_qu_with_invalid_nucleus_is_fallback(word):
    """Sau qu, âm đệm W chỉ kết hợp với a ă â e ê i/y ơ (quơ...); u/ô/o là âm tròn môi, không làm âm đệm."""
    assert safe_parse_syllable(word) is None


@pytest.mark.parametrize("text,expected", [
    ("中文", ["中文"]),
    ("Привет", ["Привет"]),
    ("ไทย", ["ไทย"]),
    ("Qwzx 中文 xyz", ["Qwzx", "中文", "xyz"]),
])
def test_foreign_scripts_preserved(text, expected):
    items = parse_text(text)
    assert [item.raw for item in items] == expected
    assert all(item.status == ParseStatus.FALLBACK_REQUIRED for item in items)


def test_nfd_input_is_composed():
    # "ế" ở dạng NFD (e + dấu mũ + dấu sắc) phải được NFC rồi parse.
    assert parse_syllable("e\u0302\u0301").abstract.nucleus == "Ê"


def test_double_tone_mark_rejected():
    with pytest.raises(ValueError):
        parse_syllable("a\u0300\u0301")


def test_uppercase_and_diacritics():
    assert parse_syllable("QUỐC").abstract.nucleus == "UÔ"
    assert parse_syllable("GÌ").abstract.onset == "GI"
    assert parse_syllable("ĐƯỜNG").abstract.nucleus == "ƯƠ"


@pytest.mark.parametrize("word", ["ooc", "moc", "hoch", "mach", "cêt"])
def test_stop_coda_requires_sac_or_nang(word):
    """Coda tắc P/T/C/CH chỉ đi với sắc/nặng: "ooc" ngang là fallback."""
    assert safe_parse_syllable(word) is None


@pytest.mark.parametrize("word", ["moóc", "mọc", "hoạch", "mách", "tết"])
def test_stop_coda_with_valid_tone_parses(word):
    assert safe_parse_syllable(word) is not None


def test_hyphenated_loan_splits_into_syllables():
    items = parse_text("a-xít và pa-lăng")
    assert [item.raw for item in items] == ["a", "xít", "và", "pa", "lăng"]
    assert all(item.status == ParseStatus.VALID for item in items)


def test_single_tone_mark_doubled_is_canonicalized():
    # "hòà" (cùng một dấu huyền dán hai lần) vẫn parse; compose về dạng chuẩn.
    parsed = parse_syllable("hòà")
    o = parsed.orthography
    composed = compose_canonical(o.onset, o.medial, o.nucleus, o.coda, parsed.abstract.tone)
    assert composed == "hoà"


def test_nfd_text_parses_via_parse_text():
    import unicodedata
    items = parse_text(unicodedata.normalize("NFD", "Hoà bình"))
    assert [item.status for item in items] == [ParseStatus.VALID, ParseStatus.VALID]



PARSABLE_CORPUS = sorted({"khuya", "khuỷu", "khoeo", "ngoằn", "ngoèo", "ngoéo", "ngoạm", "ngoạc", "oai", "oái", "oanh", "oách", "uông", "uôm", "uốc", "uống", "ươm", "ướp", "ương", "yêu", "yếu", "yểng", "yên", "uyên", "uyển", "quán", "quạnh", "quắt", "quặn", "quang", "quạu", "quấy", "quây", "quờ", "quở", "quơ", "thuở", "huơ", "huých", "khuấy", "khuây", "khuếch", "khuynh", "huynh", "xoong", "boong", "moóc", "xoài", "khoai", "ngoài", "ngoáy", "oáy", "khoắng", "hoẵng", "toạch", "ngoạch", "oạch", "ướt", "khuất", "khuân", "quốc", "quyến", "quyệt", "nguyệt", "khuyết", "khuyên", "chuyển", "chuyện", "nguyện", "duyệt", "tuyết", "thuyền", "quyền", "quyển", "nguệch", "oẹ", "oe", "loè", "xoè", "ngoè", "nhòe", "khỏe", "hòa", "ạch", "ếch", "ích", "ốc", "ộc", "ợp", "ớp", "ứp", "ụp", "ắc", "ấc", "ật", "ất", "ết", "ệt", "ịt", "ọt", "ột", "ụt", "ựt", "hộp", "hợp", "rộp", "nghiêng", "nghiệng", "nghiến", "nghiệt", "nguyễn", "giếng", "giết", "giêng", "giữ", "già", "gì", "gìn", "đường", "đượng", "được", "tưởng", "tượng", "hưởng", "thưởng", "ngưỡng", "cưỡng", "quẽ", "quánh", "uých", "oắt", "oăn", "oăng", "oóc", "quỳnh", "huỳnh", "khuỳnh", "oong", "eng", "éc", "ếc", "âu", "ấu", "ầu", "ếu", "ều", "lau", "láu", "lâu", "lều"})


@pytest.mark.parametrize("word", PARSABLE_CORPUS)
def test_real_word_corpus_parses(word):
    assert safe_parse_syllable(word) is not None, word

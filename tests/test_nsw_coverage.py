"""NSW coverage from web research (VietNormalizer, VieNeu-TTS, soe-vinorm, TTSProof).

Covers semiotic classes: cardinal, decimal, percent, currency, phone,
thousand separators, leading-zero codes, acronyms, versions/codes,
URLs/emails/hashtags, whitespace/punctuation robustness, idempotency.
Known gaps (dates, times, fractions, ranges) are documented as fallback
no-crash tests + xfail for ideal verbalization.
"""

import pytest
from vifasttts.frontend.normalizer import integer_to_words, normalize_text
from vifasttts.frontend.pipeline import parse_text
from vifasttts.types import ParseStatus


# --- Thousand separators (Vietnamese "." and space) ---
@pytest.mark.parametrize("source,expected", [
    ("1.500.000", "một triệu năm trăm nghìn"),
    ("70 000", "bảy mươi nghìn"),
    ("50.000", "năm mươi nghìn"),
])
def test_thousand_separators(source, expected):
    assert normalize_text(source) == expected


def test_decimal_dot_preserved():
    # 3.14 has only 2 digits after dot -> decimal, not thousand
    assert normalize_text("3.14") == "ba phẩy một bốn"
    assert normalize_text("3,14") == "ba phẩy một bốn"


# --- Currency ---
@pytest.mark.parametrize("source,expected", [
    ("50.000đ", "năm mươi nghìn đồng"),
    ("50.000d", "năm mươi nghìn đồng"),
    ("1000VND", "một nghìn đồng"),
    ("1000 đồng", "một nghìn đồng"),
    ("$5", "năm đô la"),
    ("5$", "năm đô la"),
])
def test_currency(source, expected):
    assert normalize_text(source) == expected


def test_currency_3d_not_mangled():
    # "3d" (phim 3D) must not become currency
    assert normalize_text("phim 3d") != "phim ba đồng"


# --- Phone numbers digit-by-digit ---
def test_phone_continuous():
    assert normalize_text("0977123456") == "không chín bảy bảy một hai ba bốn năm sáu"


def test_phone_plus84():
    out = normalize_text("+84977123456")
    assert out.startswith("không chín")


# --- Leading-zero codes ---
@pytest.mark.parametrize("source,expected", [
    ("007", "không không bảy"),
    ("01", "không một"),
])
def test_leading_zeros(source, expected):
    assert normalize_text(source) == expected


# --- Chem false positive guard ---
def test_version_code_not_chemistry():
    assert normalize_text("RTX3080") == "RTX3080"
    items = parse_text("RTX3080")
    assert len(items) == 1
    assert items[0].status == ParseStatus.FALLBACK_REQUIRED


# --- Acronyms ---
def test_nasa_acronym():
    assert normalize_text("NASA") == "na xa"


def test_version_strings_conservative():
    # v2, 3.5.1, MP3-style codes stay unchanged (no crash, fallback)
    assert normalize_text("v2") == "v2"
    items = parse_text("v2")
    assert items[0].status == ParseStatus.FALLBACK_REQUIRED


# --- URLs / emails / hashtags: no crash, fallback ---
@pytest.mark.parametrize("source", [
    "hello@example.com",
    "https://example.com",
    "#hashtag",
    "contact@test.vn",
])
def test_url_email_no_crash(source):
    items = parse_text(source)
    assert len(items) >= 1  # never crashes


# --- Whitespace / punctuation robustness ---
def test_whitespace_flood():
    items = parse_text("Chào   \t  bạn!!!")
    assert len(items) == 2


def test_zero_width_no_crash():
    items = parse_text("a\u200bb")
    assert len(items) >= 1


def test_emoji_no_crash():
    items = parse_text("Chào 😀 bạn")
    assert len(items) >= 2


def test_punctuation_only():
    assert parse_text("&") == []
    assert parse_text("@") == []


# --- Idempotency (OmniVoice-style: normalize twice == once) ---
@pytest.mark.parametrize("source", [
    "Tôi có 105 sách",
    "50%",
    "VI",
    "H2O 5kg",
    "1.500.000đ",
    "0977123456",
])
def test_idempotent(source):
    once = normalize_text(source)
    assert normalize_text(once) == once


# --- Ordinals via numbers ---
def test_ordinal_thu():
    assert normalize_text("thứ 2") == "thứ hai"


# --- Known gaps: documented, fallback no-crash now, ideal later ---
@pytest.mark.parametrize("source", [
    "25/12/2023",
    "14:30",
    "2/3",
    "3-5",
    "1873-1907",
])
def test_known_gap_no_crash(source):
    items = parse_text(source)
    assert len(items) >= 1


@pytest.mark.xfail(reason="Dates not yet verbalized (need DD/MM/YYYY support)")
def test_date_ideal():
    assert normalize_text("25/12/2023") == "ngày hai mươi lăm tháng mười hai năm hai nghìn không trăm hai mươi ba"


@pytest.mark.xfail(reason="Times not yet verbalized (need HH:MM support)")
def test_time_ideal():
    assert normalize_text("14:30") == "mười bốn giờ ba mươi phút"


@pytest.mark.xfail(reason="Fractions not yet verbalized (need 2/3 support)")
def test_fraction_ideal():
    assert normalize_text("2/3") == "hai phần ba"


@pytest.mark.xfail(reason="Ranges not yet verbalized (need 3-5 -> den support)")
def test_range_ideal():
    assert normalize_text("3-5") == "ba đến năm"

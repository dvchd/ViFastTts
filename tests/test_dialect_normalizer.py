import pytest
from vifasttts.frontend.normalizer import NormalizerOptions, integer_to_words, normalize_text
from vifasttts.frontend.dialect import Dialect


@pytest.mark.parametrize("dialect,expected", [
    ("north", "một nghìn"),
    ("central", "một ngàn"),
    ("south", "một ngàn"),
])
def test_thousand_by_region(dialect, expected):
    assert integer_to_words(1000, dialect) == expected


@pytest.mark.parametrize("dialect,expected", [
    ("north", "một trăm linh năm"),
    ("central", "một trăm lẻ năm"),
    ("south", "một trăm lẻ năm"),
])
def test_zero_tens_by_region(dialect, expected):
    assert integer_to_words(105, dialect) == expected


def test_four_after_tens_profile():
    assert integer_to_words(24, "north") == "hai mươi tư"
    assert integer_to_words(24, "central") == "hai mươi tư"
    assert integer_to_words(24, "south") == "hai mươi bốn"


@pytest.mark.parametrize("number,expected", [
    (0, "không"), (1, "một"), (10, "mười"), (11, "mười một"),
    (14, "mười bốn"), (15, "mười lăm"), (20, "hai mươi"),
    (21, "hai mươi mốt"), (25, "hai mươi lăm"),
    (99, "chín mươi chín"), (100, "một trăm"),
    (101, "một trăm linh một"), (115, "một trăm mười lăm"),
    (1001, "một nghìn không trăm linh một"),
    (1_000_000, "một triệu"), (1_000_000_000, "một tỷ"),
])
def test_number_inventory(number, expected):
    assert integer_to_words(number, "north") == expected


def test_decimal_percent_and_english():
    assert normalize_text("3,5%", "north") == "ba phẩy năm phần trăm"
    assert normalize_text("WiFi USB AI", "north") == "quai phai diu ét bi ây ai"


def test_override_english():
    opt = NormalizerOptions(dialect=Dialect.NORTH, english_overrides={"app": "ứng dụng"})
    assert normalize_text("app", options=opt) == "ứng dụng"

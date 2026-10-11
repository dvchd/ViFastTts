import pytest

from vifasttts.frontend.normalizer import integer_to_words, normalize_text


@pytest.mark.parametrize("number,expected", [
    (1_000_500, "một triệu năm trăm"),
    (1_005_000, "một triệu không trăm linh năm nghìn"),
    (1_050_000, "một triệu không trăm năm mươi nghìn"),
    (1_010_101, "một triệu không trăm mười nghìn một trăm linh một"),
    (1_000_000_005, "một tỷ không trăm linh năm"),
    (1_005_000_000, "một tỷ không trăm linh năm triệu"),
    (1_000_000_000_000, "một nghìn tỷ"),
    (2_500_000_000_000, "hai nghìn năm trăm tỷ"),
    (999_000_999_000, "chín trăm chín mươi chín tỷ chín trăm chín mươi chín nghìn"),
])
def test_zero_group_fillers(number, expected):
    assert integer_to_words(number, "north") == expected


def test_zero_group_dialect():
    assert integer_to_words(1_005_000, "south") == "một triệu không trăm lẻ năm ngàn"


def test_huge_number_no_crash():
    # Trước đây IndexError với số >= 1e12; giờ không được crash.
    out = integer_to_words(10**21)
    assert isinstance(out, str) and out


def test_large_amount_in_text():
    assert normalize_text("2.500.000.000.000 đồng") == "hai nghìn năm trăm tỷ đồng"


@pytest.mark.parametrize("source", [
    "1.005.000",
    "2.500.000.000.000 đồng",
    "1.010.101",
])
def test_idempotent_numbers(source):
    once = normalize_text(source)
    assert normalize_text(once) == once

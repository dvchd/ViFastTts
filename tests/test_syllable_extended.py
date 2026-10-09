import pytest
from vifasttts.frontend.syllable import parse_syllable


def signature(word):
    a = parse_syllable(word).abstract
    return a.onset, a.medial, a.nucleus, a.coda, a.tone.value


@pytest.mark.parametrize("word,expected", [
    ("ăn", ("NONE", "NONE", "Ă", "N", "NGANG")),
    ("anh", ("NONE", "NONE", "A", "NH", "NGANG")),
    ("ách", ("NONE", "NONE", "A", "CH", "SAC")),
    ("ông", ("NONE", "NONE", "Ô", "NG", "NGANG")),
    ("ích", ("NONE", "NONE", "I", "CH", "SAC")),
    ("mẹ", ("M", "NONE", "E", "NONE", "NANG")),
    ("mê", ("M", "NONE", "Ê", "NONE", "NGANG")),
    ("mơ", ("M", "NONE", "Ơ", "NONE", "NGANG")),
    ("mư", ("M", "NONE", "Ư", "NONE", "NGANG")),
    ("mía", ("M", "NONE", "IÊ", "NONE", "SAC")),
    ("mua", ("M", "NONE", "UÔ", "NONE", "NGANG")),
    ("mưa", ("M", "NONE", "ƯƠ", "NONE", "NGANG")),
    ("oai", ("NONE", "W", "A", "J", "NGANG")),
    ("oán", ("NONE", "W", "A", "N", "SAC")),
    ("oách", ("NONE", "W", "A", "CH", "SAC")),
    ("khoai", ("KH", "W", "A", "J", "NGANG")),
    ("xoay", ("X", "W", "A", "J", "NGANG")),
    ("khuấy", ("KH", "W", "Â", "J", "SAC")),
    ("thuế", ("TH", "W", "Ê", "NONE", "SAC")),
    ("huyền", ("H", "W", "IÊ", "N", "HUYEN")),
    ("tuyên", ("T", "W", "IÊ", "N", "NGANG")),
    ("quyết", ("K", "W", "IÊ", "T", "SAC")),
    ("chuối", ("CH", "NONE", "UÔ", "J", "SAC")),
    ("mười", ("M", "NONE", "ƯƠ", "J", "HUYEN")),
    ("thiếu", ("TH", "NONE", "IÊ", "W", "SAC")),
    ("yêu", ("NONE", "NONE", "IÊ", "W", "NGANG")),
    ("vui", ("V", "NONE", "U", "J", "NGANG")),
    ("hữu", ("H", "NONE", "Ư", "W", "NGA")),
    ("mèo", ("M", "NONE", "E", "W", "HUYEN")),
    ("tôi", ("T", "NONE", "Ô", "J", "NGANG")),
    ("mời", ("M", "NONE", "Ơ", "J", "HUYEN")),
])
def test_extended_inventory(word, expected):
    assert signature(word) == expected


@pytest.mark.parametrize("word", ["bap", "bat", "bac", "bach"])
def test_stop_coda_rejects_non_entering_tone(word):
    with pytest.raises(ValueError):
        parse_syllable(word)


def test_orthographic_onsets_remain_distinct():
    assert signature("trăm")[0] == "TR"
    assert signature("chăm")[0] == "CH"
    assert signature("sương")[0] == "S"
    assert signature("xương")[0] == "X"
    assert signature("da")[0] == "D"
    assert signature("gia")[0] == "GI"
    assert signature("ra")[0] == "R"

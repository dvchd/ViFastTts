import pytest
from vifasttts.frontend.syllable import parse_syllable


@pytest.mark.parametrize("word,expected", [
    ("học", ("H", "NONE", "O", "C", "NANG")),
    ("hộc", ("H", "NONE", "Ô", "C", "NANG")),
    ("hột", ("H", "NONE", "Ô", "T", "NANG")),
    ("khanh", ("KH", "NONE", "A", "NH", "NGANG")),
    ("khang", ("KH", "NONE", "A", "NG", "NGANG")),
    ("trăm", ("TR", "NONE", "Ă", "M", "NGANG")),
    ("chăm", ("CH", "NONE", "Ă", "M", "NGANG")),
    ("hoa", ("H", "W", "A", "NONE", "NGANG")),
    ("hoàn", ("H", "W", "A", "N", "HUYEN")),
    ("ngoài", ("NG", "W", "A", "J", "HUYEN")),
    ("mai", ("M", "NONE", "A", "J", "NGANG")),
    ("sau", ("S", "NONE", "A", "W", "NGANG")),
    ("tiến", ("T", "NONE", "IÊ", "N", "SAC")),
    ("muốn", ("M", "NONE", "UÔ", "N", "SAC")),
    ("tưởng", ("T", "NONE", "ƯƠ", "NG", "HOI")),
    ("quả", ("K", "W", "A", "NONE", "HOI")),
    ("quyển", ("K", "W", "IÊ", "N", "HOI")),
    ("giá", ("GI", "NONE", "A", "NONE", "SAC")),
    ("giếng", ("GI", "NONE", "Ê", "NG", "SAC")),
])
def test_parser(word, expected):
    a = parse_syllable(word).abstract
    assert (a.onset, a.medial, a.nucleus, a.coda, a.tone.value) == expected

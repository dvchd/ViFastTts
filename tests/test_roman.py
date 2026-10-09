import pytest
from vifasttts.frontend.roman import roman_to_int
from vifasttts.frontend.normalizer import NormalizerOptions, normalize_text

@pytest.mark.parametrize("roman,value", [
    ("II",2),("III",3),("IV",4),("VI",6),("IX",9),("XI",11),
    ("XIV",14),("XIX",19),("XX",20),("XL",40),("XLIV",44),
    ("L",None),("XC",90),("XCIX",99),("C",None),("CD",400),
    ("D",None),("CM",900),("M",None),("MMXXVI",2026),("MMMCMXCIX",3999),
])
def test_roman_parser(roman,value):
    assert roman_to_int(roman) == value

@pytest.mark.parametrize("invalid", ["vi","Vi","IIII","VV","VX","IC","IL","XD","XM","MCMC","ABC","AI","VIVI",""])
def test_invalid_or_noncanonical_roman(invalid):
    assert roman_to_int(invalid) is None

@pytest.mark.parametrize("source,expected", [
    ("VI", "sáu"),
    ("vi", "vi"),
    ("Vi", "Vi"),
    ("Chương VI", "Chương sáu"),
    ("thế kỷ XXI", "thế kỷ hai mươi mốt"),
    ("AI và VI", "ây ai và sáu"),
    ("IV, VI, IX", "bốn, sáu, chín"),
])
def test_roman_normalization(source, expected):
    assert normalize_text(source) == expected

def test_roman_can_be_disabled():
    options = NormalizerOptions(roman_numerals=False)
    assert normalize_text("VI", options=options) == "VI"

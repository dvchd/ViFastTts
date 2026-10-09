import pytest
from vifasttts.frontend.normalizer import normalize_text

@pytest.mark.parametrize("source,expected", [
    ("AI đang phát triển.", "ây ai đang phát triển."),
    ("ai đang phát triển?", "ai đang phát triển?"),
    ("Ai đang ở ngoài?", "Ai đang ở ngoài?"),
    ("AI có thể biết ai đang nói.", "ây ai có thể biết ai đang nói."),
    ("Ai dùng AI?", "Ai dùng ây ai?"),
    ("API dùng AI.", "ây pi ai dùng ây ai."),
    ("api", "api"),
    ("CPU GPU USB", "xi pi diu gi pi diu diu ét bi"),
    ("Cpu Gpu Usb", "Cpu Gpu Usb"),
])
def test_acronym_case_sensitivity(source, expected):
    assert normalize_text(source) == expected

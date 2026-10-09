from vifasttts.frontend.tone import extract_tone
from vifasttts.types import Tone


def test_preserve_vowel_quality():
    assert extract_tone("học")[:2] == ("hoc", Tone.NANG)
    assert extract_tone("hộc")[:2] == ("hôc", Tone.NANG)
    assert extract_tone("hợ")[:2] == ("hơ", Tone.NANG)

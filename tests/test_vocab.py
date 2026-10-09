from vifasttts.frontend.syllable import parse_syllable
from vifasttts.frontend.vocab import ComponentVocabs


def test_encode():
    encoded = ComponentVocabs.default().encode(parse_syllable("ngoài"))
    assert len(encoded) == 7
    assert all(isinstance(x, int) for x in encoded)

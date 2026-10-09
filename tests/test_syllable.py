from vifasttts.frontend.syllable import parse_syllable
def test_basic(): assert parse_syllable("hộc").abstract.nucleus=="Ô"

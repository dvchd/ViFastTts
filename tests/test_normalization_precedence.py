from vifasttts.frontend.dialect import Dialect
from vifasttts.frontend.normalizer import NormalizerOptions, normalize_text

def test_exact_override_beats_everything():
    o=NormalizerOptions(exact_overrides={"VI":"vê i", "AI":"trí tuệ nhân tạo"})
    assert normalize_text("VI AI",options=o)=="vê i trí tuệ nhân tạo"

def test_vietnamese_beats_english_fallback():
    assert normalize_text("video vi ai") == "vi đi ô vi ai"

def test_roman_uses_dialect_number_reading():
    # MMIV = 2004, showing regional thousand reading in Roman expansion.
    assert normalize_text("MMIV", "north") == "hai nghìn không trăm linh bốn"
    assert normalize_text("MMIV", "south") == "hai ngàn không trăm lẻ bốn"

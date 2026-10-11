
from vifasttts.frontend.oov import LETTER_READINGS, resolve_oov_token, spell_token
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.syllable import safe_parse_syllable
from vifasttts.types import ParseStatus


def test_letter_readings_are_valid_syllables():
    for letter, reading in LETTER_READINGS.items():
        for word in reading.split():
            assert safe_parse_syllable(word) is not None, f"{letter!r} -> {reading!r}"


def test_spell_token():
    assert spell_token("VNG") == "vê en giê"
    assert spell_token("abc") == "a bê xê"


def test_resolve_oov_priorities():
    assert resolve_oov_token("NASA") == "na xa"          # acronym
    assert resolve_oov_token("MacBook") == "mác búc"     # product
    assert resolve_oov_token("tiktok") == "tíc tốc"      # english fallback
    assert resolve_oov_token("VNG") == "vê en giê"       # all-caps spelling
    assert resolve_oov_token("RTX3080") is None          # has digits -> keep fallback
    assert resolve_oov_token("blorptang") is None        # unknown lowercase -> keep fallback


def test_pipeline_resolve_oov_opt_in():
    items = parse_text("VNG tăng trưởng")
    assert items[0].status == ParseStatus.FALLBACK_REQUIRED

    items = parse_text("VNG tăng trưởng", resolve_oov=True)
    assert [item.normalized for item in items[:3]] == ["vê", "en", "giê"]
    assert all(item.raw == "VNG" for item in items[:3])
    assert all(item.status == ParseStatus.VALID for item in items[:3])
    assert all(item.confidence == 0.5 for item in items[:3])
    assert items[3].normalized == "tăng"

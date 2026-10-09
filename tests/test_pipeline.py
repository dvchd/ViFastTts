from vifasttts.frontend.pipeline import parse_text
from vifasttts.types import ParseStatus


def test_valid_vietnamese():
    items = parse_text("Hôm nay trời đẹp.")
    assert len(items) == 4
    assert all(x.status == ParseStatus.VALID for x in items)


def test_punctuation_boundaries():
    items = parse_text("Chào, bạn!")
    assert len(items) == 2
    assert items[0].boundaries.phrase_end and not items[0].boundaries.sentence_end
    assert items[1].boundaries.sentence_end
    assert "," in (items[0].punctuation or "")
    assert "!" in (items[1].punctuation or "")


def test_english_fallback_no_crash():
    items = parse_text("Hello world")
    assert len(items) == 2
    assert all(x.status == ParseStatus.FALLBACK_REQUIRED for x in items)
    assert all(x.confidence == 0.0 for x in items)


def test_mixed_valid_and_fallback():
    items = parse_text("Chào Hello")
    assert len(items) == 2
    assert items[0].status == ParseStatus.VALID
    assert items[1].status == ParseStatus.FALLBACK_REQUIRED


def test_invalid_syllable_no_crash():
    # "sach" without tone is invalid (CH coda requires SAC/NANG)
    items = parse_text("sach")
    assert len(items) == 1
    assert items[0].status == ParseStatus.FALLBACK_REQUIRED


def test_empty_and_spaces():
    assert parse_text("") == []
    assert parse_text("   ") == []


def test_leading_punctuation():
    items = parse_text('"Chào"')
    assert len(items) == 1
    assert items[0].status == ParseStatus.VALID


def test_dialect_option():
    north = parse_text("1000", dialect="north")
    south = parse_text("1000", dialect="south")
    assert len(north) == 2
    assert len(south) == 2


def test_fallback_encodable():
    from vifasttts.frontend.vocab import ComponentVocabs

    vocabs = ComponentVocabs.default()
    items = parse_text("Hello")
    assert len(items) == 1
    encoded = vocabs.encode(items[0])
    assert len(encoded) == 7

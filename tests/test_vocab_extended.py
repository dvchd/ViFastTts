from vifasttts.frontend.syllable import parse_syllable
from vifasttts.frontend.vocab import ComponentVocabs


def test_encode_boundaries_and_punctuation():
    vocabs = ComponentVocabs.default()
    items = __import__("vifasttts.frontend.pipeline", fromlist=["parse_text"]).parse_text("Chào, bạn!")
    assert len(items) == 2
    e0 = vocabs.encode(items[0])
    e1 = vocabs.encode(items[1])
    # boundary: PHRASE=2, SENTENCE=3
    assert e0[5] == vocabs.boundary["PHRASE"]
    assert e1[5] == vocabs.boundary["SENTENCE"]
    # punctuation: COMMA vs EXCLAMATION
    assert e0[6] == vocabs.punctuation["COMMA"]
    assert e1[6] == vocabs.punctuation["EXCLAMATION"]


def test_encode_fallback_valid_ids():
    from vifasttts.frontend.pipeline import parse_text

    vocabs = ComponentVocabs.default()
    items = parse_text("Hello")
    assert len(items) == 1
    encoded = vocabs.encode(items[0])
    assert len(encoded) == 7
    assert all(isinstance(x, int) for x in encoded)


def test_vocab_tables_cover_rules():
    from vifasttts.frontend.rules import CODA_ABSTRACT, NUCLEUS_ABSTRACT, ONSET_ABSTRACT

    vocabs = ComponentVocabs.default()
    for v in ONSET_ABSTRACT.values():
        assert v in vocabs.onset, f"onset {v} missing"
    for v in NUCLEUS_ABSTRACT.values():
        assert v in vocabs.nucleus, f"nucleus {v} missing"
    for v in CODA_ABSTRACT.values():
        assert v in vocabs.coda, f"coda {v} missing"

from vifasttts.frontend.unicode import normalize_unicode


def test_spaces_collapsed():
    assert normalize_unicode("  Hello   world  ") == "Hello world"


def test_ellipsis_and_quotes():
    assert normalize_unicode("a…b") == "a...b"
    assert normalize_unicode("“Hello”") == '"Hello"'
    assert normalize_unicode("‘test’") == "'test'"
    assert normalize_unicode("’") == "'"


def test_nfc_normalized():
    # e + combining acute should become precomposed
    assert normalize_unicode("e\u0301") == "é"

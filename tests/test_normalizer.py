from vifasttts.frontend.normalizer import integer_to_words, normalize_text


def test_numbers():
    assert integer_to_words(105) == "một trăm linh năm"
    assert integer_to_words(21) == "hai mươi mốt"
    assert integer_to_words(15) == "mười lăm"
    assert normalize_text("Tôi có 105 sách") == "Tôi có một trăm linh năm sách"

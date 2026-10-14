from vifasttts.frontend.normalizer import integer_to_words, normalize_text


def test_numbers():
    assert integer_to_words(105) == "một trăm linh năm"
    assert integer_to_words(21) == "hai mươi mốt"
    assert integer_to_words(15) == "mười lăm"
    assert normalize_text("Tôi có 105 sách") == "Tôi có một trăm linh năm sách"


def test_symbols_math_and_common():
    """V11/P0.2: ký hiệu giữa chữ/số phải được đọc, không bị nuốt."""
    assert normalize_text("2+2=4") == "hai cộng hai bằng bốn"
    assert normalize_text("3x4") == "ba nhân bốn"
    assert normalize_text("10÷2") == "mười chia hai"
    assert normalize_text("a = b") == "a bằng b"
    assert normalize_text("A & B") == "A và B"
    assert normalize_text("5% > 3%") == "năm phần trăm lớn hơn ba phần trăm"
    assert normalize_text("n < 10") == "n nhỏ hơn mười"
    assert normalize_text("x >= 5") == "x lớn hơn hoặc bằng năm"
    assert normalize_text("email@abc.com") == "i meo a còng a bê xê chấm com"


def test_symbols_do_not_break_decoration_or_phone():
    """Trang trí (>>, C++, ==) và số điện thoại +84 giữ nguyên hành vi cũ."""
    assert normalize_text("C++") == "C++"
    assert normalize_text("<<<>>>") == "<<<>>>"
    assert normalize_text("a==b") == "a==b"
    assert normalize_text("+84977123456") == "không chín bảy bảy một hai ba bốn năm sáu"

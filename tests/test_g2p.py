from vifasttts.frontend.g2p import (
    phoneme_inventory,
    syllable_to_phonemes,
    text_to_phonemes,
    to_phoneme_string,
)
from vifasttts.frontend.syllable import parse_syllable


def test_simple_syllables_north():
    assert syllable_to_phonemes(parse_syllable("hoa")) == ("h", "w", "a", "˧")
    assert syllable_to_phonemes(parse_syllable("quy")) == ("k", "w", "i", "˧")
    assert syllable_to_phonemes(parse_syllable("ăn")) == ("ă", "n", "˧")


def test_tone_contour_and_formats():
    parsed = parse_syllable("đường")
    assert syllable_to_phonemes(parsed) == ("ɗ", "ɯə", "ŋ", "˨˩")
    assert syllable_to_phonemes(parsed, with_tone=False) == ("ɗ", "ɯə", "ŋ")
    assert syllable_to_phonemes(parsed, tone_format="number") == ("ɗ", "ɯə", "ŋ", "2")
    assert syllable_to_phonemes(parsed, tone_format="none") == ("ɗ", "ɯə", "ŋ")


def test_six_tone_numbers_match_vig2p_convention():
    expected = {"ma": "1", "mà": "2", "mã": "3", "mả": "4", "má": "5", "mạ": "6"}
    for word, number in expected.items():
        assert syllable_to_phonemes(parse_syllable(word), tone_format="number")[-1] == number


def test_northern_mergers():
    # Hà Nội (Kirby 2011): d/gi/r → /z/, ch/tr → /tɕ/, s/x → /s/.
    for group in (("da", "gia", "ra"), ("cha", "tra"), ("sa", "xa")):
        phoneme_sets = {syllable_to_phonemes(parse_syllable(word)) for word in group}
        assert len(phoneme_sets) == 1, group


def test_southern_distinctions():
    assert syllable_to_phonemes(parse_syllable("da"), "south") == ("j", "a", "˦")
    assert syllable_to_phonemes(parse_syllable("gia"), "south") == ("j", "a", "˦")
    assert syllable_to_phonemes(parse_syllable("ra"), "south")[0] == "ʐ"
    assert syllable_to_phonemes(parse_syllable("sa"), "south")[0] == "ʂ"
    assert syllable_to_phonemes(parse_syllable("xa"), "south")[0] == "s"
    assert syllable_to_phonemes(parse_syllable("tra"), "south")[0] == "ʈ"
    assert syllable_to_phonemes(parse_syllable("cha"), "south")[0] == "c"


def test_southern_hoi_nga_merged():
    assert syllable_to_phonemes(parse_syllable("mả"), "south")[-1] == "˨˩˦"
    assert syllable_to_phonemes(parse_syllable("mã"), "south")[-1] == "˨˩˦"
    # Miền Bắc vẫn phân biệt hỏi và ngã.
    assert syllable_to_phonemes(parse_syllable("mả"))[-1] != syllable_to_phonemes(parse_syllable("mã"))[-1]


def test_narrow_style():
    assert syllable_to_phonemes(parse_syllable("mát"), style="narrow") == ("m", "a", "t̚", "˧˥")
    # Broad giữ nhãn "a"; narrow Bắc phải là [ăj] + coda tiền-velar
    # (Henderson 1965; Kirby 2011: /ɛŋ/→[ăjŋ̟], /ɛk/→[ăjk̟̚]).
    assert syllable_to_phonemes(parse_syllable("anh"), style="broad") == ("a", "ɲ", "˧")
    assert syllable_to_phonemes(parse_syllable("anh"), style="narrow") == ("ăj", "ŋ̟", "˧")
    assert syllable_to_phonemes(parse_syllable("ách"), style="narrow") == ("ăj", "k̟̚", "˧˥")
    assert syllable_to_phonemes(parse_syllable("ênh"), style="narrow") == ("ə̆j", "ŋ̟", "˧")
    assert syllable_to_phonemes(parse_syllable("tôi"), style="narrow")[0] == "t̪"


def test_gi_shared_i_reads_as_zi():
    # "giếng" = GI + IÊ + NG (i dùng chung), không phải z + e.
    assert syllable_to_phonemes(parse_syllable("giếng")) == ("z", "iə", "ŋ", "˧˥")
    assert syllable_to_phonemes(parse_syllable("giếng"), "south") == ("j", "iə", "ŋ", "˧˥")
    assert syllable_to_phonemes(parse_syllable("gì")) == ("z", "i", "˨˩")
    assert syllable_to_phonemes(parse_syllable("gìn")) == ("z", "i", "n", "˨˩")


def test_oo_loanword_has_distinct_long_nucleus():
    # "oo" (xoong, moóc) là nucleus OO riêng (o dài ɔː) — người bản ngữ phân biệt với O.
    # Không phải nguyên âm đôi: âm đôi thật chỉ có iê/uô/ươ (iə/uə/ɯə).
    assert syllable_to_phonemes(parse_syllable("xoong")) == ("s", "ɔː", "ŋ", "˧")
    assert syllable_to_phonemes(parse_syllable("xoong"), style="narrow") == ("s", "ɔː", "ŋ", "˧")
    assert syllable_to_phonemes(parse_syllable("xong"), style="narrow") == ("s", "ɔ", "ŋ", "˧")
    # Nam gộp /ɔː/ vào /ɔ/ (xoong ≈ xong); Trung xấp xỉ Bắc.
    assert syllable_to_phonemes(parse_syllable("xoong"), "south") == ("s", "ɔ", "ŋ", "˦")
    assert syllable_to_phonemes(parse_syllable("xong"), "south") == ("s", "ɔ", "ŋ", "˦")
    assert syllable_to_phonemes(parse_syllable("xoong"), "central") == ("s", "ɔː", "ŋ", "˧")
    assert syllable_to_phonemes(parse_syllable("moóc"), "south", style="narrow") == ("m", "ɔ", "k̚", "˧˥")


def test_narrow_hanoi_uu_uou_fronting():
    # Thompson (1965): Hà Nội đọc ưu/ươu ngả [iw]/[iəw] ở dạng narrow.
    assert syllable_to_phonemes(parse_syllable("cứu"), style="narrow") == ("k", "i", "w", "˧˥")
    assert syllable_to_phonemes(parse_syllable("hươu"), style="narrow") == ("h", "iə", "w", "˧")
    # Broad giữ âm vị /ɯw/; Nam giữ ɯ.
    assert syllable_to_phonemes(parse_syllable("cứu")) == ("k", "ɯ", "w", "˧˥")
    assert syllable_to_phonemes(parse_syllable("cứu"), "south", style="narrow") == ("k", "ɯ", "w", "˧˥")


def test_ai_ay_ao_au_have_length_contrast():
    # Help:IPA/Vietnamese: "ai" [aːj] vs "ay" [aj]; bảng nguyên âm lấy "may", "cau" = [a].
    assert syllable_to_phonemes(parse_syllable("tai")) == ("t", "a", "j", "˧")
    assert syllable_to_phonemes(parse_syllable("tay")) == ("t", "ă", "j", "˧")
    assert syllable_to_phonemes(parse_syllable("cao")) == ("k", "a", "w", "˧")
    assert syllable_to_phonemes(parse_syllable("cau")) == ("k", "ă", "w", "˧")
    assert syllable_to_phonemes(parse_syllable("quay")) == ("k", "w", "ă", "j", "˧")
    assert syllable_to_phonemes(parse_syllable("quai")) == ("k", "w", "a", "j", "˧")
    # coda /j/ giống nhau, chỉ nguyên âm khác (không phải khác âm cuối).
    assert syllable_to_phonemes(parse_syllable("tai"))[-2:] == syllable_to_phonemes(parse_syllable("tay"))[-2:]


def test_fallback_returns_empty():
    items = text_to_phonemes("Zibble Qwux")
    assert all(phonemes == () for phonemes in items)
    items = text_to_phonemes("Chào bạn")
    assert all(phonemes for phonemes in items)


def test_inventory_and_string():
    inventory = phoneme_inventory()
    for symbol in ("ɓ", "ɗ", "ŋ", "ɲ", "ʈ", "ʂ", "ʐ", "˧˥", "˨˩˦", "1", "6"):
        assert symbol in inventory, symbol
    assert to_phoneme_string(("h", "w", "a")) == "h w a"

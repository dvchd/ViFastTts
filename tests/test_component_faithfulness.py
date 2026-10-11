"""Bất biến của biểu diễn 5 thành phần — lý do không dùng IPA ở runtime.

Các test này khóa quyết định thiết kế: token hoá tiếng Việt thành
ONSET + MEDIAL + NUCLEUS + CODA + TONE là biểu diễn phonemic-native,
không mất thông tin so với IPA cho các đối lập âm vị học, trong khi
nhỏ, rời rạc và deterministic hơn một chuỗi IPA + G2P.
"""
from __future__ import annotations

import pytest

from vifasttts.frontend.normalizer import normalize_text
from vifasttts.frontend.orthography import compose_canonical
from vifasttts.frontend.syllable import parse_syllable


def _tuple(word: str):
    return parse_syllable(word).abstract


# Các đối lập âm vị học thật: thanh, phụ âm đầu, vần, coda.
DISTINCT_GROUPS = [
    ("ma", "mà", "má", "mả", "mã", "mạ"),
    ("hoa", "hoà", "hoá", "hoả", "hoã", "hoạ"),
    ("qua", "quà", "quá", "quả", "quã", "quạ"),
    ("sa", "xa"),
    ("cha", "tra"),
    ("căn", "cân"),
    ("lam", "lan"),
    ("cang", "canh"),
    ("bát", "bác"),
    ("chia", "chiên"),
    # uy (medial W + nucleus I) phân biệt với ui (nucleus U + coda J)
    ("khuy", "khui"),
    ("thuy", "thui"),
    ("tủy", "tủi"),
    # uy là vần đóng: khuya (W + IÊ) khác khua (UÔ)
    ("khuya", "khua"),
    # oo là nucleus riêng (xoong ≠ xong; moóc ≠ móc)
    ("xoong", "xong"),
    ("moóc", "móc"),
]


@pytest.mark.parametrize("group", DISTINCT_GROUPS)
def test_minimal_pairs_have_distinct_component_tuples(group):
    tuples = {_tuple(word) for word in group}
    assert len(tuples) == len(group)


# Biến thể chính tả cùng âm vị: i/y cuối, vị trí dấu thanh cũ/mới.
SAME_PHONEME_GROUPS = [
    ("tai", "tay"),
    ("mai", "may"),
    ("hai", "hay"),
    ("thủy", "thuỷ"),
    ("hòa", "hoà"),
]


@pytest.mark.parametrize("group", SAME_PHONEME_GROUPS)
def test_spelling_variants_share_component_tuple(group):
    tuples = {_tuple(word) for word in group}
    assert len(tuples) == 1


@pytest.mark.parametrize("word", [
    "thủy", "thuỷ", "hòa", "hoà", "quý", "nghiêng", "giếng", "xoài",
    "đường", "được", "ăn", "uống", "khuỷu", "ngoằn",
    "tủi", "củi", "chùi", "khuy", "khuya", "huynh", "xoong", "moóc",
    "gì", "gìn", "giêng", "giết", "già", "giữ",
])
def test_canonical_round_trip_is_stable(word):
    parsed = parse_syllable(word)
    canonical = compose_canonical(
        parsed.orthography.onset, parsed.orthography.medial,
        parsed.orthography.nucleus, parsed.orthography.coda, parsed.abstract.tone,
    )
    # normalize_text phải xuất đúng dạng canonical này...
    assert normalize_text(word) == canonical
    # ...và parse lại dạng canonical phải cho kết quả y hệt (idempotent).
    reparsed = parse_syllable(canonical)
    assert compose_canonical(
        reparsed.orthography.onset, reparsed.orthography.medial,
        reparsed.orthography.nucleus, reparsed.orthography.coda, reparsed.abstract.tone,
    ) == canonical


def test_gi_shared_i_matches_ieng_rhyme():
    """'giếng' phải cùng vần với 'tiếng': chữ i của "gi" thuộc vần iê."""
    gieng = parse_syllable("giếng").abstract
    tieng = parse_syllable("tiếng").abstract
    assert (gieng.nucleus, gieng.coda) == (tieng.nucleus, tieng.coda) == ("IÊ", "NG")
    # Nhưng âm đầu vẫn là GI (/z/), không phải G (/ɣ/).
    assert gieng.onset == "GI"


def test_component_layer_keeps_d_and_gi_apart():
    """IPA Bắc gộp d/gi thành /z/; thiết kế cố ý giữ riêng ở tầng thành phần
    (docs/FRONTEND.md: "Giữ D, GI, R riêng") để model nhất quán với chính tả."""
    assert _tuple("da") != _tuple("gia")
    from vifasttts.frontend.g2p import syllable_to_phonemes

    # Tầng IPA tùy chọn (chỉ phân tích) gộp đúng như âm vị học Bắc bộ.
    assert syllable_to_phonemes(parse_syllable("da"), "north") == syllable_to_phonemes(parse_syllable("gia"), "north")


def test_ipa_layer_distinguishes_what_components_distinguish():
    """Cross-check: với các cặp IPA phân biệt được, component cũng phải vậy
    (component không được "mù" hơn IPA ở các đối lập cơ bản)."""
    from vifasttts.frontend.g2p import syllable_to_phonemes

    for group in DISTINCT_GROUPS:
        ipa_sequences = {syllable_to_phonemes(parse_syllable(w)) for w in group}
        component_tuples = {_tuple(w) for w in group}
        if len(ipa_sequences) == len(group):
            assert len(component_tuples) == len(group)


def test_six_tones_have_distinct_contours():
    from vifasttts.frontend.g2p import TONE_CONTOURS_NORTH, TONE_CONTOURS_SOUTH

    assert len(set(TONE_CONTOURS_NORTH.values())) == 6
    # Miền Nam gộp hỏi–ngã (Huỳnh Công 2003) — đúng thực tế, không phải lỗi.
    assert TONE_CONTOURS_SOUTH["HOI"] == TONE_CONTOURS_SOUTH["NGA"]
    assert len(set(TONE_CONTOURS_SOUTH.values())) == 5


def test_component_vocab_is_small_and_discrete():
    """44.352 tổ hợp lý thuyết phủ toàn bộ âm tiết bản ngữ, không cần UNK."""
    from vifasttts.frontend.vocab import ComponentVocabs

    vocabs = ComponentVocabs.default()
    product = (len(vocabs.onset) * len(vocabs.medial) * len(vocabs.nucleus)
               * len(vocabs.coda) * len(vocabs.tone))
    assert product == 24 * 2 * 15 * 11 * 6
    assert product >= 40_000

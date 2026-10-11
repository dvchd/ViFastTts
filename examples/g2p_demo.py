"""Demo G2P: các khả năng của `frontend/g2p.py` và kết quả thật.

Chạy: python examples/g2p_demo.py
(Bảng IPA broad/narrow theo Kirby 2011 & Thompson 1965; xem docs/FRONTEND.md.)
"""
from __future__ import annotations

from vifasttts.frontend.g2p import (
    phoneme_inventory,
    syllable_to_phonemes,
    text_to_phonemes,
    to_phoneme_string,
)
from vifasttts.frontend.syllable import parse_syllable
from vifasttts.frontend.vocab import ComponentVocabs


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    section("1. Một âm tiết: Bắc vs Nam (broad, thanh Chao)")
    for word in ["hoa", "quy", "đường", "nguyễn", "rồi", "khuya"]:
        parsed = parse_syllable(word)
        north = to_phoneme_string(syllable_to_phonemes(parsed, "north"))
        south = to_phoneme_string(syllable_to_phonemes(parsed, "south"))
        print(f"{word:<9} Bắc: {north:<22} Nam: {south}")

    section("2. Gộp/tách âm đầu theo vùng (Bắc gộp, Nam tách)")
    for group in (("da", "gia", "ra"), ("cha", "tra"), ("sa", "xa")):
        north = " = ".join(to_phoneme_string(syllable_to_phonemes(parse_syllable(w), "north")) for w in group)
        south = " | ".join(to_phoneme_string(syllable_to_phonemes(parse_syllable(w), "south")) for w in group)
        print(f"{'/'.join(group):<12} Bắc: {north:<24} Nam: {south}")

    section("3. Sáu thanh: Chao letters và số (quy ước viG2P)")
    for word in ["ma", "mà", "mã", "mả", "má", "mạ"]:
        parsed = parse_syllable(word)
        contour = to_phoneme_string(syllable_to_phonemes(parsed, "north"))
        number = to_phoneme_string(syllable_to_phonemes(parsed, "north", tone_format="number"))
        no_tone = to_phoneme_string(syllable_to_phonemes(parsed, "north", tone_format="none"))
        print(f"{word:<6} {contour:<24} số: {number:<8} không thanh: {no_tone}")

    section("4. Broad vs narrow (chi tiết ngữ âm)")
    for word in ["mát", "anh", "tôi", "được"]:
        parsed = parse_syllable(word)
        broad = to_phoneme_string(syllable_to_phonemes(parsed))
        narrow = to_phoneme_string(syllable_to_phonemes(parsed, style="narrow"))
        print(f"{word:<8} broad: {broad:<22} narrow: {narrow}")

    section("5. Cả câu: một tuple IPA cho mỗi âm tiết")
    sentence = "Chào bạn, hôm nay trời đẹp."
    for dialect in ("north", "south"):
        phonemes = text_to_phonemes(sentence, dialect)
        print(f"{dialect:<6} {len(phonemes)} âm tiết:")
        print("      " + " | ".join(to_phoneme_string(item) for item in phonemes))

    section("6. OOV: trả tuple rỗng, không crash (để tầng ngoài xử lý)")
    for token in ["Hello", "RTX3080", "blorptang"]:
        print(f"{token:<10} -> {text_to_phonemes(token)}")

    section("7. Kho ký hiệu IPA (phoneme_inventory)")
    inventory = phoneme_inventory()
    print(f"{len(inventory)} ký hiệu: {' '.join(inventory[:14])} ...")

    section("8. Vì sao KHÔNG phải map 1-1 chữ cái -> IPA")
    print("Nhiều chữ viết -> cùng một âm vị:")
    for group, label in ((("ca", "kê", "quả"), "c / k / qu → /k/"),
                         (("ga", "ghế"), "g / gh → /ɣ/"),
                         (("nga", "nghệ"), "ng / ngh → /ŋ/")):
        readings = " = ".join(to_phoneme_string(syllable_to_phonemes(parse_syllable(w))) for w in group)
        print(f"  {' / '.join(group):<16} {readings:<34} ({label})")
    da, gia, ra = (to_phoneme_string(syllable_to_phonemes(parse_syllable(w))) for w in ("da", "gia", "ra"))
    print(f"  {'da / gia / ra':<16} {da} = {gia} = {ra}   (d / gi / r → /z/ ở Bắc)")

    print("Một chữ viết / một dấu -> khác nhau theo vị trí và phương ngữ:")
    print(f"  {'giếng':<16} Bắc: {to_phoneme_string(syllable_to_phonemes(parse_syllable('giếng')))}"
          f"   (i dùng chung: GI + iê, không phải z + e)")
    print(f"  {'gì / gìn':<16} Bắc: {to_phoneme_string(syllable_to_phonemes(parse_syllable('gì')))}"
          f" / {to_phoneme_string(syllable_to_phonemes(parse_syllable('gìn')))}   (i thuộc vần)")
    print(f"  {'quốc':<16} Bắc: {to_phoneme_string(syllable_to_phonemes(parse_syllable('quốc'), style='narrow'))}"
          f"   (qu = /k/ + /w/)")
    print(f"  {'tai vs tay':<16} {to_phoneme_string(syllable_to_phonemes(parse_syllable('tai')))}"
          f" vs {to_phoneme_string(syllable_to_phonemes(parse_syllable('tay')))}"
          f"   (ai = a dài [aː]; ay = ă ngắn [a])")
    print(f"  {'da':<16} Bắc: {to_phoneme_string(syllable_to_phonemes(parse_syllable('da'), 'north'))}"
          f" | Nam: {to_phoneme_string(syllable_to_phonemes(parse_syllable('da'), 'south'))}   (d khác nhau theo vùng)")
    print(f"  {'mả (dấu hỏi)':<16} Bắc: {to_phoneme_string(syllable_to_phonemes(parse_syllable('mả'), 'north'))}"
          f" | Nam: {to_phoneme_string(syllable_to_phonemes(parse_syllable('mả'), 'south'))}   (một dấu, hai đường nét)")

    print("1-1 thật sự nằm ở mức 5 thành phần (với một phương ngữ):")
    parsed = parse_syllable("giếng")
    print(f"  (GI, NONE, IÊ, NG, SẮC) + Bắc -> {to_phoneme_string(syllable_to_phonemes(parsed, 'north'))}")
    print(f"  (GI, NONE, IÊ, NG, SẮC) + Nam  -> {to_phoneme_string(syllable_to_phonemes(parsed, 'south'))}")

    section("9. Vì sao phải có hệ ký hiệu riêng, không lấy IPA làm khóa")
    header = ("từ", "bộ âm vị (trung lập phương ngữ)", "IPA Bắc", "IPA Nam")
    print(f"{header[0]:<8}{header[1]:<34}{header[2]:<16}{header[3]}")
    for word in ["da", "gia", "ra", "mả", "giếng"]:
        parsed = parse_syllable(word)
        a = parsed.abstract
        tup = "(" + f"{a.onset}, {a.medial}, {a.nucleus}, {a.coda}, {a.tone.value}" + ")"
        north = to_phoneme_string(syllable_to_phonemes(parsed, "north"))
        south = to_phoneme_string(syllable_to_phonemes(parsed, "south"))
        print(f"{word:<8}{tup:<34}{north:<16}{south}")
    print("Bắc: da/gia/ra là 3 tuple khác nhau nhưng cho cùng IPA /za/ -> IPA mất phân biệt.")
    print("Nam: một tuple vẫn dùng lại được, IPA thì đổi thành j / j / ʐ -> khóa IPA Bắc")
    print("     không dùng được cho checkpoint Nam, trong khi frontend là dùng chung.")
    vocabs = ComponentVocabs.default()
    encoded = vocabs.encode(parse_syllable("da"))
    print("Model không thấy chữ nào — chỉ thấy ID số (nhãn chỉ để người đọc):")
    print(f"  'da' -> vocab.encode = {encoded}")


if __name__ == "__main__":
    main()

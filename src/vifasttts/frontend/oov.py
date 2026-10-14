"""OOV resolver theo quy trình 5 bước của viG2P (viết lại, không dùng code GPL).

Thứ tự: (1) override/acronym/product và English fallback đã có ở normalizer;
(2) token giống từ tiếng Anh → respelling có sẵn; (3) viết hoa toàn bộ →
đánh vần từng chữ cái bằng tên chữ tiếng Việt; (4) còn lại trả None để giữ
FALLBACK_REQUIRED cho người dùng quyết định (log + bổ sung override).

Không gọi G2P ngoài ở runtime; xem docs/G2P_VIETNAMESE.md mục 6.4.
"""

from __future__ import annotations

from vifasttts.frontend.english_fallback import (
    lookup_case_insensitive_english,
    lookup_exact_acronym,
    lookup_product_name,
)

# Tên chữ cái tiếng Việt, mọi giá trị phải là âm tiết Việt hợp lệ.
LETTER_READINGS = {
    "a": "a", "ă": "á", "â": "ớ", "b": "bê", "c": "xê", "d": "dê", "đ": "đê",
    "e": "e", "ê": "ê", "f": "ép", "g": "giê", "h": "hát", "i": "i",
    "j": "giây", "k": "ca", "l": "lờ", "m": "em", "n": "en", "o": "o",
    "ô": "ô", "ơ": "ơ", "p": "pê", "q": "quy", "r": "rờ", "s": "ét",
    "t": "tê", "u": "u", "ư": "ư", "v": "vê", "w": "vê kép", "x": "ích",
    "y": "i", "z": "dét",
}


def spell_token(token: str) -> str:
    """Đánh vần tên chữ cái, ghép bằng khoảng trắng (đã là âm tiết Việt)."""
    return " ".join(LETTER_READINGS[c] for c in token.lower() if c in LETTER_READINGS)


def resolve_oov_token(token: str) -> str | None:
    """Trả về respelling tiếng Việt cho token OOV, hoặc None nếu không giải được."""
    for lookup in (lookup_exact_acronym, lookup_product_name):
        reading = lookup(token)
        if reading is not None:
            return reading
    english = lookup_case_insensitive_english(token)
    if english is not None:
        return english
    # V11 (P0.2, 14/10): đánh vần khi token "trông như viết tắt/từ lạ ngắn":
    #   (a) chữ HOA (mọi độ dài, gồm 1 ký tự — "B" → "bê"); (b) có số/ký hiệu ("COVID-19", "TP.HCM");
    #   (c) từ thường ngắn ≤4 ("wav"). Từ thường DÀI vẫn trả None để báo cáo OOV còn giá trị.
    letters = [c for c in token if c.isalpha()]
    mixed = len(letters) != len(token)
    if letters and all(c.lower() in LETTER_READINGS for c in letters) and (token.isupper() or mixed or len(letters) <= 4):
        spelled = spell_token("".join(letters))
        digits = "".join(c for c in token if c.isdigit())
        return f"{spelled} {digits}" if digits else spelled
    return None

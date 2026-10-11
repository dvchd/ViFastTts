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
    # Chỉ đánh vần khi toàn bộ là chữ cái (giữ chữ số lại cho người dùng xử lý)
    # và có ít nhất hai chữ cái viết hoa (tránh nhầm một chữ cái đơn).
    letters = [c for c in token if c.isalpha()]
    if token.isalpha() and token.isupper() and len(letters) >= 2 and all(c.lower() in LETTER_READINGS for c in letters):
        return spell_token(token)
    return None

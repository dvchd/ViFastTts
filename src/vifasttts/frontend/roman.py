"""Nhận dạng số La Mã viết hoa chuẩn (1..3999) để đọc thành số tiếng Việt."""

from __future__ import annotations

import re

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_CANONICAL_ROMAN = re.compile(
    r"^(?=.{2,15}$)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)


def roman_to_int(token: str) -> int | None:
    """Đổi số La Mã viết hoa chuẩn (≥2 ký tự) sang int; trả None nếu không hợp lệ.

    Yêu cầu viết HOA toàn bộ để không nuốt nhầm từ thường (`vi`, `mi`...).
    Ví dụ: `VI` → 6; `vi`, `VX`, `IIII` → None.
    """
    if token != token.upper() or not token.isascii() or not _CANONICAL_ROMAN.fullmatch(token):
        return None
    total = 0
    previous = 0
    for char in reversed(token):
        value = ROMAN_VALUES[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total if 1 <= total <= 3999 else None

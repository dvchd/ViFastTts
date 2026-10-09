from __future__ import annotations

import re

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_CANONICAL_ROMAN = re.compile(
    r"^(?=.{2,15}$)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)


def roman_to_int(token: str) -> int | None:
    """Convert canonical uppercase Roman numeral in range 1..3999.

    At least two characters are required by default. This preserves common
    single-letter tokens such as I, V, X, C, D and M for other normalization
    policies. `VI` is six, while `vi` and `Vi` are not Roman numerals.
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

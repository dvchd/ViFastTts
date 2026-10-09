from __future__ import annotations

import re
from dataclasses import dataclass, field

from vifasttts.frontend.dialect import Dialect, ReadingProfile, get_profile
from vifasttts.frontend.english_fallback import (
    lookup_case_insensitive_english, lookup_exact_acronym, lookup_product_name,
)
from vifasttts.frontend.roman import roman_to_int
from vifasttts.frontend.stem import normalize_stem
from vifasttts.frontend.unicode import normalize_unicode

SMALL = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+|[^A-Za-zÀ-ỹĐđ]+", re.UNICODE)


@dataclass(frozen=True)
class NormalizerOptions:
    dialect: Dialect = Dialect.NORTH
    canonical_tone_spelling: bool = True
    english_fallback: bool = True
    roman_numerals: bool = True
    stem_formulas: bool = True
    # Exact override giữ nguyên hoa thường. Lowercase override dùng cho từ Anh thường.
    exact_overrides: dict[str, str] = field(default_factory=dict)
    english_overrides: dict[str, str] = field(default_factory=dict)


def _under_100(n: int, p: ReadingProfile) -> str:
    if n < 10: return SMALL[n]
    q, r = divmod(n, 10)
    prefix = "mười" if q == 1 else f"{SMALL[q]} mươi"
    if r == 0: return prefix
    if r == 1 and q > 1: tail = "mốt"
    elif r == 4 and q > 1: tail = p.four_after_tens
    elif r == 5: tail = "lăm"
    else: tail = SMALL[r]
    return f"{prefix} {tail}"


def _under_1000(n: int, p: ReadingProfile, force_hundreds: bool = False) -> str:
    if n < 100 and not force_hundreds: return _under_100(n, p)
    q, r = divmod(n, 100); prefix = f"{SMALL[q]} trăm"
    if r == 0: return prefix
    if r < 10: return f"{prefix} {p.zero_tens} {SMALL[r]}"
    return f"{prefix} {_under_100(r, p)}"


def integer_to_words(n: int, dialect: str | Dialect = Dialect.NORTH) -> str:
    p = get_profile(dialect)
    if n < 0: return "âm " + integer_to_words(-n, dialect)
    if n == 0: return SMALL[0]
    units = [(1_000_000_000, "tỷ"), (1_000_000, "triệu"), (1000, p.thousand)]
    parts=[]; remaining=n; started=False
    for value, unit in units:
        group, remaining = divmod(remaining, value)
        if group:
            parts += [_under_1000(group, p), unit]; started=True
    if remaining:
        parts.append(_under_1000(remaining, p, force_hundreds=started and remaining < 100))
    return " ".join(parts)


def decimal_to_words(value: str, dialect: str | Dialect) -> str:
    left, right = re.split(r"[,.]", value, maxsplit=1)
    return f"{integer_to_words(int(left), dialect)} phẩy " + " ".join(SMALL[int(x)] for x in right)


def _canonicalize_vietnamese_token(token: str) -> str | None:
    from vifasttts.frontend.syllable import safe_parse_syllable
    from vifasttts.frontend.orthography import compose_canonical
    item = safe_parse_syllable(token)
    if item is None: return None
    o, a = item.orthography, item.abstract
    result = compose_canonical(o.onset, o.medial, o.nucleus, o.coda, a.tone)
    if token.isupper(): result = result.upper()
    elif token[:1].isupper(): result = result[:1].upper() + result[1:]
    return result


def normalize_word_token(token: str, options: NormalizerOptions) -> str:
    # 1. Exact user override has the highest priority.
    if token in options.exact_overrides:
        return options.exact_overrides[token]
    # 2. Acronym is exact and case-sensitive: AI is acronym, ai/Ai are Vietnamese.
    acronym = lookup_exact_acronym(token)
    if acronym is not None:
        return acronym
    # 3. Exact product spelling.
    product = lookup_product_name(token)
    if product is not None:
        return product
    # 4. Canonical uppercase Roman numeral, at least two letters: VI -> six.
    if options.roman_numerals:
        number = roman_to_int(token)
        if number is not None:
            return integer_to_words(number, options.dialect)
    # 5. Vietnamese syllable takes priority over case-insensitive English.
    vietnamese = _canonicalize_vietnamese_token(token)
    if vietnamese is not None:
        return vietnamese if options.canonical_tone_spelling else token
    # 6. User English override and built-in English fallback.
    if options.english_fallback:
        custom = options.english_overrides.get(token.lower())
        if custom is not None: return custom
        english = lookup_case_insensitive_english(token)
        if english is not None: return english
    return token


def normalize_words(text: str, options: NormalizerOptions) -> str:
    return "".join(normalize_word_token(t, options) if re.fullmatch(r"[A-Za-zÀ-ỹĐđ]+", t) else t for t in _TOKEN_RE.findall(text))


def canonicalize_vietnamese_tokens(text: str) -> str:
    options = NormalizerOptions(english_fallback=False, roman_numerals=False)
    return normalize_words(text, options)


def normalize_text(text: str, dialect: str | Dialect = Dialect.NORTH, *, options: NormalizerOptions | None = None) -> str:
    options = options or NormalizerOptions(dialect=Dialect(dialect))
    text = normalize_unicode(text)
    if options.stem_formulas:
        text = normalize_stem(text, lambda n: integer_to_words(n, options.dialect))
    text = re.sub(r"(?<!\w)(\d+[,.]\d+)\s*%", lambda m: decimal_to_words(m.group(1), options.dialect) + " phần trăm", text)
    text = re.sub(r"(?<!\w)(\d+)\s*%", lambda m: integer_to_words(int(m.group(1)), options.dialect) + " phần trăm", text)
    text = re.sub(r"(?<!\w)(\d+[,.]\d+)(?!\w)", lambda m: decimal_to_words(m.group(1), options.dialect), text)
    text = re.sub(r"(?<!\w)(-?\d+)(?![\w/.:])", lambda m: integer_to_words(int(m.group(1)), options.dialect), text)
    text = normalize_words(text, options)
    return re.sub(r"\s+", " ", text).strip()

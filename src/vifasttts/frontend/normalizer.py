"""Chuẩn hoá văn bản trước khi parse: số, ngày/giờ, tiền tệ, STEM, viết tắt.

Thứ tự xử lý quan trọng: giờ "14h30" trước STEM, ngày trước giờ:phút, phân số
trước phần trăm, tiền tệ trước số thường... xem comment trong `normalize_text`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vifasttts.frontend.dialect import Dialect, ReadingProfile, get_profile
from vifasttts.frontend.english_fallback import (
    lookup_case_insensitive_english,
    lookup_exact_acronym,
    lookup_product_name,
)
from vifasttts.frontend.roman import roman_to_int
from vifasttts.frontend.stem import normalize_stem
from vifasttts.frontend.unicode import normalize_unicode

SMALL = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+|[^A-Za-zÀ-ỹĐđ]+", re.UNICODE)


@dataclass(frozen=True)
class NormalizerOptions:
    """Tuỳ chọn chuẩn hoá: vùng miền, quy ước dấu thanh, fallback, override.

    - `dialect`: cách đọc số theo Bắc/Trung/Nam (nghìn/ngàn, linh/lẻ, tư/bốn).
    - `canonical_tone_spelling`: đưa vị trí dấu thanh về lối hiện đại (`tùy→tuỳ`).
    - `exact_overrides`: "mặt chữ → cách đọc", ưu tiên cao nhất, giữ nguyên hoa thường.
    - `english_overrides`: override cho từ tiếng Anh (so khớp chữ thường).
    """

    dialect: Dialect = Dialect.NORTH
    canonical_tone_spelling: bool = True
    english_fallback: bool = True
    roman_numerals: bool = True
    stem_formulas: bool = True
    # Override chính xác giữ nguyên hoa thường; override chữ thường dùng cho từ Anh.
    exact_overrides: dict[str, str] = field(default_factory=dict)
    english_overrides: dict[str, str] = field(default_factory=dict)


def _under_100(n: int, p: ReadingProfile) -> str:
    """Đọc số 0..99 (mười/mươi/mốt/lăm/tư tuỳ vùng)."""
    if n < 10:
        return SMALL[n]
    q, r = divmod(n, 10)
    prefix = "mười" if q == 1 else f"{SMALL[q]} mươi"
    if r == 0:
        return prefix
    if r == 1 and q > 1:
        tail = "mốt"
    elif r == 4 and q > 1:
        tail = p.four_after_tens
    elif r == 5:
        tail = "lăm"
    else:
        tail = SMALL[r]
    return f"{prefix} {tail}"


def _under_1000(n: int, p: ReadingProfile) -> str:
    """Đọc số 0..999 (trăm + hàng chục, kèm linh/lẻ khi hàng chục < 10)."""
    if n < 100:
        return _under_100(n, p)
    q, r = divmod(n, 100)
    prefix = f"{SMALL[q]} trăm"
    if r == 0:
        return prefix
    if r < 10:
        return f"{prefix} {p.zero_tens} {SMALL[r]}"
    return f"{prefix} {_under_100(r, p)}"


def _read_group(g: int, p: ReadingProfile, started: bool) -> str:
    """Đọc một nhóm ba chữ số.

    Nếu nhóm cao hơn đã đọc và nhóm này không có hàng trăm, tiếng Việt cần
    chèn "không trăm": 1.050.000 → "một triệu không trăm năm mươi nghìn".
    """
    if started and g < 100:
        if g < 10:
            return f"không trăm {p.zero_tens} {SMALL[g]}"
        return f"không trăm {_under_100(g, p)}"
    return _under_1000(g, p)


def _read_below_billion(n: int, p: ReadingProfile, *, started: bool = False) -> str:
    """Đọc số < 1 tỷ theo nhóm nghìn/triệu."""
    groups: list[int] = []
    while n > 0:
        n, r = divmod(n, 1000)
        groups.append(r)
    scales = ("", p.thousand, "triệu")
    parts: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        g = groups[index]
        if g == 0:
            continue
        text = _read_group(g, p, started)
        scale = scales[index]
        parts.append(f"{text} {scale}".strip())
        started = True
    return " ".join(parts)


def _read_number(n: int, p: ReadingProfile) -> str:
    """Đọc số nguyên bất kỳ (đệ quy theo "tỷ" cho số ≥ 10^9)."""
    if n == 0:
        return SMALL[0]
    if n >= 1_000_000_000:
        # Đọc đệ quy theo "… tỷ": 2.500.000.000.000 -> "hai nghìn năm trăm tỷ".
        high, rest = divmod(n, 1_000_000_000)
        text = _read_number(high, p) + " tỷ"
        if rest:
            text += " " + _read_below_billion(rest, p, started=True)
        return text
    return _read_below_billion(n, p, started=False)


def integer_to_words(n: int, dialect: str | Dialect = Dialect.NORTH) -> str:
    """Số nguyên → chữ tiếng Việt theo vùng (ví dụ 1500 → "một nghìn năm trăm")."""
    p = get_profile(dialect)
    if n < 0:
        return "âm " + integer_to_words(-n, dialect)
    return _read_number(n, p)


def decimal_to_words(value: str, dialect: str | Dialect) -> str:
    """Số thập phân → chữ, đọc từng chữ số sau "phẩy" (3.14 → "ba phẩy một bốn")."""
    left, right = re.split(r"[,.]", value, maxsplit=1)
    return f"{integer_to_words(int(left), dialect)} phẩy " + " ".join(SMALL[int(x)] for x in right)


def _phone_to_words(match: re.Match[str]) -> str:
    """Số điện thoại → đọc từng chữ số; chuẩn hoá +84 về 0."""
    digits = re.sub(r"\D", "", match.group(0))
    if digits.startswith("84") and len(digits) >= 11:
        digits = "0" + digits[2:]
    return " ".join(SMALL[int(d)] for d in digits)


def _currency_number_to_words(num_str: str, dialect: str | Dialect) -> str:
    """Phần số của tiền tệ → chữ (giữ nguyên phần thập phân nếu có)."""
    if re.search(r"[,.]", num_str):
        return decimal_to_words(num_str, dialect)
    return integer_to_words(int(num_str), dialect)


def _integer_token_to_words(match: re.Match[str], dialect: str | Dialect) -> str:
    """Token số nguyên thường → chữ; giữ số 0 đầu bằng cách đọc từng chữ số."""
    raw = match.group(1)
    # Mã kiểu 007 phải đọc từng chữ số để không mất thông tin.
    if len(raw) > 1 and raw.startswith("0"):
        return " ".join(SMALL[int(d)] for d in raw)
    return integer_to_words(int(raw), dialect)


def _canonicalize_vietnamese_token(token: str) -> str | None:
    """Đưa âm tiết Việt về chính tả canonical (dấu thanh); None nếu không parse được.

    Giữ hoa/thường như chữ gốc (VIỆT → VIỆT, Việt → Việt).
    """
    from vifasttts.frontend.orthography import compose_canonical
    from vifasttts.frontend.syllable import safe_parse_syllable
    item = safe_parse_syllable(token)
    if item is None: return None
    o, a = item.orthography, item.abstract
    result = compose_canonical(o.onset, o.medial, o.nucleus, o.coda, a.tone)
    if token.isupper(): result = result.upper()
    elif token[:1].isupper(): result = result[:1].upper() + result[1:]
    return result


def normalize_word_token(token: str, options: NormalizerOptions) -> str:
    """Chuẩn hoá một token chữ theo thứ tự ưu tiên 6 bước (xem comment trong hàm)."""
    # 1. Override của người dùng luôn có ưu tiên cao nhất.
    if token in options.exact_overrides:
        return options.exact_overrides[token]
    # 2. Viết tắt khớp chính xác, phân biệt hoa thường: "AI" là viết tắt, "ai"/"Ai" là tiếng Việt.
    acronym = lookup_exact_acronym(token)
    if acronym is not None:
        return acronym
    # 3. Tên sản phẩm viết đúng như bảng.
    product = lookup_product_name(token)
    if product is not None:
        return product
    # 4. Số La Mã viết hoa chuẩn (≥2 ký tự): VI → sáu.
    if options.roman_numerals:
        number = roman_to_int(token)
        if number is not None:
            return integer_to_words(number, options.dialect)
    # 5. Âm tiết tiếng Việt được ưu tiên hơn tiếng Anh không phân biệt hoa thường.
    vietnamese = _canonicalize_vietnamese_token(token)
    if vietnamese is not None:
        return vietnamese if options.canonical_tone_spelling else token
    # 6. Override tiếng Anh của người dùng rồi tới từ điển tiếng Anh có sẵn.
    if options.english_fallback:
        custom = options.english_overrides.get(token.lower())
        if custom is not None: return custom
        english = lookup_case_insensitive_english(token)
        if english is not None: return english
    return token


def normalize_words(text: str, options: NormalizerOptions) -> str:
    """Áp `normalize_word_token` cho mọi token chữ, giữ nguyên phần không phải chữ."""
    return "".join(normalize_word_token(t, options) if re.fullmatch(r"[A-Za-zÀ-ỹĐđ]+", t) else t for t in _TOKEN_RE.findall(text))


def canonicalize_vietnamese_tokens(text: str) -> str:
    """Chỉ chuẩn hoá chính tả tiếng Việt (tắt La Mã + English fallback)."""
    options = NormalizerOptions(english_fallback=False, roman_numerals=False)
    return normalize_words(text, options)


# --- Ngày, giờ, phân số, khoảng giá trị ---

# "14h30" và "14h" là cách viết giờ phổ biến; chạy trước STEM để không bị
# hiểu nhầm thành đơn vị vật lý. Phút yêu cầu đủ hai chữ số.
_HOUR_RE = re.compile(r"(?<!\w)(\d{1,2})h(\d{2})?(?!\w)")
_DATE_RE = re.compile(r"(?<![\d/.,-])(?:[Nn]gày\s+)?(\d{1,2})([/.-])(\d{1,2})\2(\d{4})(?!\d)")
_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_FRACTION_RE = re.compile(r"(?<![\d/.,])(\d+)/(\d+)(?![\d/.,])")
_RANGE_RE = re.compile(r"(?<![\d.,-])(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)(?:\s*(%))?(?![\d.,])")


def _hour_minute_words(hour: int, minute: int, dialect: str | Dialect) -> str:
    """Ghép "X giờ Y phút"; phút < 10 dùng linh/lẻ theo vùng."""
    parts = [f"{integer_to_words(hour, dialect)} giờ"]
    if minute:
        tail = integer_to_words(minute, dialect)
        parts.append(f"{get_profile(dialect).zero_tens} {tail} phút" if minute < 10 else f"{tail} phút")
    return " ".join(parts)


def _hour_notation_to_words(m: re.Match[str], dialect: str | Dialect) -> str:
    """Đọc "14h30"/"14h" thành "mười bốn giờ ba mươi phút"."""
    minute = int(m.group(2)) if m.group(2) else 0
    return _hour_minute_words(int(m.group(1)), minute, dialect)


def _date_to_words(m: re.Match[str], dialect: str | Dialect) -> str:
    """Đọc ngày dạng `25/12/2023` (giữ nguyên nếu ngày/tháng không hợp lệ)."""
    day, month, year = int(m.group(1)), int(m.group(3)), int(m.group(4))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return m.group(0)
    return (f"ngày {integer_to_words(day, dialect)} tháng {integer_to_words(month, dialect)} "
            f"năm {integer_to_words(year, dialect)}")


def _time_to_words(m: re.Match[str], dialect: str | Dialect) -> str:
    """Đọc giờ:phút `14:30` (giữ nguyên nếu giờ/phút không hợp lệ)."""
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return m.group(0)
    return _hour_minute_words(hour, minute, dialect)


def _fraction_to_words(m: re.Match[str], dialect: str | Dialect) -> str:
    """Đọc phân số `2/3` thành "hai phần ba"."""
    return f"{integer_to_words(int(m.group(1)), dialect)} phần {integer_to_words(int(m.group(2)), dialect)}"


def _number_token_to_words(value: str, dialect: str | Dialect) -> str:
    """Đọc token số (tự nhận thập phân hay nguyên) thành chữ."""
    if "," in value or "." in value:
        return decimal_to_words(value, dialect)
    return integer_to_words(int(value), dialect)


def _range_to_words(m: re.Match[str], dialect: str | Dialect) -> str:
    """Đọc khoảng giá trị `3-5`, `10-20%` thành "… đến …"."""
    text = f"{_number_token_to_words(m.group(1), dialect)} đến {_number_token_to_words(m.group(2), dialect)}"
    return text + " phần trăm" if m.group(3) else text


def normalize_text(text: str, dialect: str | Dialect = Dialect.NORTH, *, options: NormalizerOptions | None = None) -> str:
    """Chuẩn hoá cả đoạn văn: số/ngày/giờ/tiền/%; thứ tự các bước rất quan trọng."""
    options = options or NormalizerOptions(dialect=Dialect(dialect))
    text = normalize_unicode(text)
    # Giờ viết "14h30"/"14h" trước STEM để không lẫn với đơn vị vật lý.
    text = _HOUR_RE.sub(lambda m: _hour_notation_to_words(m, options.dialect), text)
    if options.stem_formulas:
        text = normalize_stem(text, lambda n: integer_to_words(n, options.dialect))
    # Dấu phân cách nghìn kiểu Việt ("." và khoảng trắng): 1.500.000 → 1500000.
    # Chỉ xoá dấu chấm/khoảng trắng khi theo sau đúng 3 chữ số, nên 3.14 giữ nguyên.
    text = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", text)
    text = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "", text)
    # Ngày/tháng/năm trước mọi phép tách "/"; giờ:phút sau đó.
    text = _DATE_RE.sub(lambda m: _date_to_words(m, options.dialect), text)
    text = _TIME_RE.sub(lambda m: _time_to_words(m, options.dialect), text)
    # Số điện thoại đọc từng chữ số: 0977123456, +84977123456.
    text = re.sub(r"(?<!\w)(?:\+84\d{9,10}|0\d{9,10})(?!\w)", _phone_to_words, text)
    # Tiền tệ: 50.000đ / 1000VND / $5 / 5$ → "… đồng" / "… đô la".
    # Chạy trước số thường để không khớp một phần.
    # Chữ "d" ASCII là cách viết tắt phổ biến của "đồng", nhưng chỉ với số ≥1000
    # để không phá "3d" (phim 3D) thành tiền tệ.
    text = re.sub(r"(?<!\w)(\d+(?:[,.]\d+)?)\s*(?:đồng|đ|vnd|VND)(?![A-Za-zÀ-ỹĐđ])", lambda m: _currency_number_to_words(m.group(1), options.dialect) + " đồng", text)
    text = re.sub(r"(?<!\w)(\d{4,}(?:[,.]\d+)?)\s*d(?![A-Za-zÀ-ỹĐđ])", lambda m: _currency_number_to_words(m.group(1), options.dialect) + " đồng", text)
    text = re.sub(r"\$\s*(\d+(?:[,.]\d+)?)", lambda m: _currency_number_to_words(m.group(1), options.dialect) + " đô la", text)
    text = re.sub(r"(?<!\w)(\d+(?:[,.]\d+)?)\s*\$(?![A-Za-zÀ-ỹĐđ])", lambda m: _currency_number_to_words(m.group(1), options.dialect) + " đô la", text)
    # Phân số "2/3" trước phần trăm để "1/2%" không bị cắt nửa chừng.
    text = _FRACTION_RE.sub(lambda m: _fraction_to_words(m, options.dialect), text)
    # Khoảng giá trị "3-5", "10-20%", "1873-1907".
    text = _RANGE_RE.sub(lambda m: _range_to_words(m, options.dialect), text)
    text = re.sub(r"(?<!\w)(\d+[,.]\d+)\s*%", lambda m: decimal_to_words(m.group(1), options.dialect) + " phần trăm", text)
    text = re.sub(r"(?<!\w)(\d+)\s*%", lambda m: integer_to_words(int(m.group(1)), options.dialect) + " phần trăm", text)
    text = re.sub(r"(?<!\w)(\d+[,.]\d+)(?!\w)", lambda m: decimal_to_words(m.group(1), options.dialect), text)
    text = re.sub(r"(?<!\w)(-?\d+)(?![\w/.:])", lambda m: _integer_token_to_words(m, options.dialect), text)
    # Thế hệ mạng di động "5G", "4G" đọc chữ cái G.
    text = re.sub(r"(?<!\w)([2-6])\s*G(?!\w)", lambda m: f"{SMALL[int(m.group(1))]} giê", text)
    text = normalize_words(text, options)
    return re.sub(r"\s+", " ", text).strip()

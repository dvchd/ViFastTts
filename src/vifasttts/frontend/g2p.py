"""G2P tuỳ chọn cho tiếng Việt — ánh xạ 5 thành phần âm tiết sang IPA chuẩn.

Bảng dưới đây là **broad (âm vị) và narrow (ngữ âm)** cho ba profile
Bắc (Hà Nội), Trung (xấp xỉ Huế) và Nam (TP.HCM), dựa trên:
- Kirby (2011) và Thompson (1965) qua "Vietnamese phonology" (Wikipedia),
- các đối chiếu thực nghiệm với sea-g2p / viG2P / WikiPron trong
  `docs/G2P_VIETNAMESE.md`.

Quy ước quan trọng:
- Bắc **gộp** d/gi/r → /z/, ch/tr → /tɕ/, s/x → /s/ (Hà Nội).
- Nam tách d/gi → /j/, r → /ʐ/, s → /ʂ/, x → /s/, tr → /ʈ/, ch → /c/.
- Trung hiện dùng âm đầu kiểu Nam và thanh điệu kiểu Bắc; chưa có bảng
  Huế đủ tin cậy — xem mục "Trung" trong docs/G2P_VIETNAMESE.md.
- Thanh điệu: mặc định ký hiệu Chao (˧ ˨˩ ...); có thể xuất số theo
  quy ước viG2P (1 ngang, 2 huyền, 3 ngã, 4 hỏi, 5 sắc, 6 nặng) để
  đối chiếu với hệ bên ngoài; `tone_format="number8"` thêm mã 7/8 cho
  **thanh nhập** (sắc nhập/nặng nhập — âm tiết kết thúc p/t/c/ch) theo
  quy ước gắn nhãn Praat. Đường nét theo Kirby 2011 (Hà Nội) và
  Huỳnh Công 2003 (TP.HCM); Help:IPA/Vietnamese có bảng khác đôi chỗ
  (ví dụ hỏi Hà Nội ˧˩ thay vì ˧˩˧) — dự án giữ nguồn đã chọn.
- Nguyên âm: `OO` là nucleus riêng [ɔː] (xoong, moóc); `ay/au` dùng ă
  ngắn [a] còn `ai/ao` dùng a dài [aː] (Help:IPA/Vietnamese).
- Phương ngữ: Nam gộp `OO`→[ɔ] (xoong ≈ xong với người Nam); Bắc narrow
  đọc `ưu/ươu` ngả [iw]/[iəw] (Thompson 1965) — broad giữ /ɯw/ɯəw/.
- Không có trọng âm từ vựng trong tiếng Việt; không thêm dấu ˈ.

LƯU Ý KIẾN TRÚC: module này KHÔNG được import bởi `models/`, `train.py`,
`infer.py` hay `pipeline.py`. Chuỗi IPA chỉ dùng cho phân tích,
kiểm chứng và công cụ build-time; đầu vào neural là 7 ID thành phần
(5 thành phần âm tiết + boundary + punctuation).
"""

from __future__ import annotations

from vifasttts.frontend.dialect import Dialect, get_profile
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.rules import STOP_CODAS
from vifasttts.frontend.syllable import parse_syllable
from vifasttts.types import ParsedSyllable, ParseStatus

ToneFormat = str  # "contour" | "number" | "number8" | "none"
Style = str  # "broad" | "narrow"

# --- Phụ âm đầu ---

ONSET_BROAD_NORTH: dict[str, tuple[str, ...]] = {
    "NONE": (), "B": ("ɓ",), "M": ("m",), "PH": ("f",), "V": ("v",),
    "T": ("t",), "TH": ("tʰ",), "Đ": ("ɗ",), "N": ("n",), "L": ("l",),
    # Hà Nội: d/gi/r đều là /z/; ch/tr đều là /tɕ/; s/x đều là /s/.
    "D": ("z",), "GI": ("z",), "R": ("z",),
    "X": ("s",), "S": ("s",),
    "CH": ("tɕ",), "TR": ("tɕ",),
    "NH": ("ɲ",), "NG": ("ŋ",), "G": ("ɣ",), "KH": ("x",), "H": ("h",),
    "K": ("k",), "P": ("p",),
}

ONSET_BROAD_SOUTH: dict[str, tuple[str, ...]] = {
    "NONE": (), "B": ("ɓ",), "M": ("m",), "PH": ("f",), "V": ("v",),
    "T": ("t",), "TH": ("tʰ",), "Đ": ("ɗ",), "N": ("n",), "L": ("l",),
    # TP.HCM: d/gi → /j/ (đa số), r → /ʐ/, s → /ʂ/, x → /s/, tr → /ʈ/, ch → /c/.
    "D": ("j",), "GI": ("j",), "R": ("ʐ",),
    "X": ("s",), "S": ("ʂ",),
    "CH": ("c",), "TR": ("ʈ",),
    "NH": ("ɲ",), "NG": ("ŋ",), "G": ("ɣ",), "KH": ("x",), "H": ("h",),
    "K": ("k",), "P": ("p",),
}

# Trung (xấp xỉ Huế): âm đầu kiểu Nam; xem ghi chú đầu file.
ONSET_BROAD_CENTRAL = ONSET_BROAD_SOUTH

# Biến thể narrow: t/th là denti-alveolar [t̪, t̪ʰ] (Kirby 2011).
ONSET_NARROW_OVERRIDES: dict[str, tuple[str, ...]] = {
    "T": ("t̪",), "TH": ("t̪ʰ",),
}

MEDIAL_BROAD: dict[str, tuple[str, ...]] = {"NONE": (), "W": ("w",)}

# --- Nguyên âm (vần) ---

# Kirby (2011): â/ă là nguyên âm ngắn [ɤ̆]/[ă]; ơ là [ɤ]; ư là [ɯ].
NUCLEUS_BROAD: dict[str, tuple[str, ...]] = {
    "A": ("a",), "Ă": ("ă",), "Â": ("ɤ̆",),
    "E": ("ɛ",), "Ê": ("e",),
    "I": ("i",), "O": ("ɔ",), "Ô": ("o",),
    "Ơ": ("ɤ",), "U": ("u",), "Ư": ("ɯ",),
    # "OO" là nucleus riêng của từ vay (xoong, boong, moóc): o dài [ɔː],
    # người bản ngữ phân biệt với "o" thường. Không phải nguyên âm đôi:
    # âm đôi thật của tiếng Việt chỉ có IÊ/UÔ/ƯƠ (iə/uə/ɯə).
    "OO": ("ɔː",),
    "IÊ": ("iə",), "UÔ": ("uə",), "ƯƠ": ("ɯə",),
}

# Phương ngữ Nam gộp /ɔː/ (OO) với /ɔ/ (O): "xoong" ≈ "xong" với người Nam.
# Trung hiện xấp xỉ bảng Bắc cho nguyên âm (xem ghi chú "Trung" cuối file).
NUCLEUS_BROAD_SOUTH: dict[str, tuple[str, ...]] = {**NUCLEUS_BROAD, "OO": ("ɔ",)}
NUCLEUS_BROAD_CENTRAL: dict[str, tuple[str, ...]] = dict(NUCLEUS_BROAD)

# Narrow Bắc: trước coda nh/ch, "a"/"ê" rút ngắn và có offglide ngạc [j]
# (Henderson 1965: [k̟, ŋ̟] tiền-velar; Kirby 2011: /ɛŋ/→[ăjŋ̟], /ek/→[ə̆jk̟̚]).
NARROW_J_OFFGLIDE: dict[str, str] = {"A": "ăj", "Ê": "ə̆j"}

# --- Phụ âm cuối ---

CODA_BROAD: dict[str, tuple[str, ...]] = {
    "NONE": (), "M": ("m",), "N": ("n",), "NG": ("ŋ",), "NH": ("ɲ",),
    "P": ("p",), "T": ("t",), "C": ("k",), "CH": ("c",),
    "J": ("j",), "W": ("w",),
}

# Narrow: âm tắc cuối không bật hơi [p̚ t̚ k̚]; ch/nh cuối là tiền ngạc
# [k̟̚, ŋ̟] theo Thompson (1965).
CODA_NARROW_OVERRIDES: dict[str, tuple[str, ...]] = {
    "P": ("p̚",), "T": ("t̚",), "C": ("k̚",),
    "CH": ("k̟̚",), "NH": ("ŋ̟",),
}

# --- Thanh điệu (Chao tone letters) ---

# Hà Nội (Kirby 2011; Thompson 1965).
TONE_CONTOURS_NORTH: dict[str, str] = {
    "NGANG": "˧", "HUYEN": "˨˩", "SAC": "˧˥",
    "HOI": "˧˩˧", "NGA": "˧ˀ˥", "NANG": "˧ˀ˨ʔ",
}

# TP.HCM (Huỳnh Công 2003 qua Wikipedia): hỏi–ngã gộp thành ˨˩˦.
TONE_CONTOURS_SOUTH: dict[str, str] = {
    "NGANG": "˦", "HUYEN": "˧˩", "SAC": "˧˥",
    "HOI": "˨˩˦", "NGA": "˨˩˦", "NANG": "˨˩˨",
}

# Trung: chưa có bảng Huế đủ tin cậy; tạm dùng bảng Bắc cho thanh điệu.
TONE_CONTOURS_CENTRAL = TONE_CONTOURS_NORTH

# Số thanh theo quy ước viG2P để đối chiếu với hệ ngoài.
TONE_NUMBERS: dict[str, str] = {
    "NGANG": "1", "HUYEN": "2", "NGA": "3", "HOI": "4", "SAC": "5", "NANG": "6",
}

# Mã 7/8 = thanh nhập (sắc nhập/nặng nhập) cho âm tiết kết thúc coda tắc p/t/c/ch —
# quy ước gắn nhãn Praat / phân tích 8 thanh; "number8" = "number" + 2 mã này.
ENTERING_NUMBERS: dict[str, str] = {"SAC": "7", "NANG": "8"}


def tone_table(dialect: str | Dialect) -> dict[str, str]:
    """Bảng đường nét thanh (Chao) theo phương ngữ."""
    if Dialect(dialect) is Dialect.SOUTH:
        return TONE_CONTOURS_SOUTH
    if Dialect(dialect) is Dialect.CENTRAL:
        return TONE_CONTOURS_CENTRAL
    return TONE_CONTOURS_NORTH


def onset_table(dialect: str | Dialect) -> dict[str, tuple[str, ...]]:
    """Bảng phụ âm đầu broad theo phương ngữ (Trung dùng chung bảng Nam)."""
    return ONSET_BROAD_SOUTH if Dialect(dialect) is not Dialect.NORTH else ONSET_BROAD_NORTH


def nucleus_table(dialect: str | Dialect) -> dict[str, tuple[str, ...]]:
    """Bảng nucleus broad theo phương ngữ (Nam gộp OO→ɔ như O)."""
    if Dialect(dialect) is Dialect.SOUTH:
        return NUCLEUS_BROAD_SOUTH
    if Dialect(dialect) is Dialect.CENTRAL:
        return NUCLEUS_BROAD_CENTRAL
    return NUCLEUS_BROAD


def nucleus_units(
    nucleus: str,
    dialect: str | Dialect = Dialect.NORTH,
    *,
    style: Style = "broad",
    coda: str = "NONE",
) -> tuple[str, ...]:
    """Đơn vị IPA cho nucleus, xử lý biến thể narrow theo phương ngữ.

    - Nam: `OO` → [ɔ] (gộp với `O`).
    - Bắc narrow + coda `W`: `ưu/ươu` ngả về [iw]/[iəw] (Thompson 1965).
    - Bắc narrow + coda `NH/CH`: `a`/`ê` rút ngắn + offglide ngạc [ăj]/[ə̆j]
      (Henderson 1965; Kirby 2011).
    """
    dialect_value = Dialect(dialect)
    if style == "narrow" and dialect_value is Dialect.NORTH:
        if coda in ("NH", "CH") and nucleus in NARROW_J_OFFGLIDE:
            return (NARROW_J_OFFGLIDE[nucleus],)
        if coda == "W":
            if nucleus == "Ư":
                return ("i",)
            if nucleus == "ƯƠ":
                return ("iə",)
    return nucleus_table(dialect_value)[nucleus]


def syllable_to_phonemes(
    parsed: ParsedSyllable,
    dialect: str | Dialect = Dialect.NORTH,
    *,
    with_tone: bool = True,
    tone_format: ToneFormat = "contour",
    style: Style = "broad",
) -> tuple[str, ...]:
    """Chuyển một âm tiết đã parse sang chuỗi IPA.

    - `style="broad"` (mặc định): dạng âm vị, gọn theo cách viết phổ biến.
    - `style="narrow"`: thêm chi tiết ngữ âm (t̪, p̚, k̟̚, ŋ̟).
    - `tone_format`: "contour" (Chao letters), "number" (viG2P 1–6),
      "number8" (viG2P + 7/8 cho thanh nhập: sắc nhập/nặng nhập) hoặc "none".
    - Trả về tuple rỗng cho âm tiết FALLBACK_REQUIRED (từ ngoại lai).
    """
    if parsed.status != ParseStatus.VALID:
        return ()
    get_profile(dialect)  # validate dialect
    a = parsed.abstract
    onset = list(onset_table(dialect)[a.onset])
    if style == "narrow":
        onset = list(ONSET_NARROW_OVERRIDES.get(a.onset, tuple(onset)))
    coda = list(CODA_BROAD[a.coda])
    if style == "narrow":
        coda = list(CODA_NARROW_OVERRIDES.get(a.coda, tuple(coda)))
    phonemes = (
        tuple(onset)
        + MEDIAL_BROAD[a.medial]
        + nucleus_units(a.nucleus, dialect, style=style, coda=a.coda)
        + tuple(coda)
    )
    if not with_tone or tone_format == "none":
        return phonemes
    if tone_format == "number":
        return phonemes + (TONE_NUMBERS[a.tone.value],)
    if tone_format == "number8":
        if a.coda in STOP_CODAS and a.tone.value in ENTERING_NUMBERS:
            return phonemes + (ENTERING_NUMBERS[a.tone.value],)
        return phonemes + (TONE_NUMBERS[a.tone.value],)
    return phonemes + (tone_table(dialect)[a.tone.value],)


def text_to_phonemes(
    text: str,
    dialect: str | Dialect = Dialect.NORTH,
    *,
    with_tone: bool = True,
    tone_format: ToneFormat = "contour",
    style: Style = "broad",
) -> list[tuple[str, ...]]:
    """Mỗi âm tiết một tuple IPA (tuple rỗng = OOV)."""
    return [
        syllable_to_phonemes(item, dialect, with_tone=with_tone, tone_format=tone_format, style=style)
        for item in parse_text(text, dialect=dialect)
    ]


def phoneme_inventory() -> tuple[str, ...]:
    """Toàn bộ ký hiệu IPA có thể xuất, cho vocab phoneme tùy chọn."""
    phonemes: set[str] = set()
    for table in (ONSET_BROAD_NORTH, ONSET_BROAD_SOUTH, MEDIAL_BROAD, NUCLEUS_BROAD, NUCLEUS_BROAD_SOUTH, CODA_BROAD):
        for value in table.values():
            phonemes.update(value)
    for overrides in (ONSET_NARROW_OVERRIDES, CODA_NARROW_OVERRIDES):
        for value in overrides.values():
            phonemes.update(value)
    phonemes.update(NARROW_J_OFFGLIDE.values())
    for tone_table in (TONE_CONTOURS_NORTH, TONE_CONTOURS_SOUTH):
        phonemes.update(tone_table.values())
    phonemes.update(TONE_NUMBERS.values())
    phonemes.update(ENTERING_NUMBERS.values())
    return tuple(sorted(phonemes))


def to_phoneme_string(phonemes: tuple[str, ...]) -> str:
    """Nối tuple âm vị IPA thành chuỗi cách nhau bằng dấu cách (để in/log)."""
    return " ".join(phonemes)


if __name__ == "__main__":  # pragma: no cover - tiện chạy tay
    for raw in ("hoa", "quy", "đường", "rồi", "nghiêng", "nguyễn"):
        item = parse_syllable(raw)
        print(f"{raw:10} north broad: {to_phoneme_string(syllable_to_phonemes(item))}")
        print(f"{'':10} south broad: {to_phoneme_string(syllable_to_phonemes(item, 'south'))}")

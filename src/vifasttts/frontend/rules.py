"""Bảng ánh xạ chính tả ↔ âm vị học: âm đầu, âm đệm, âm chính, âm cuối + bất biến.

Gồm cả các bảng compatibility dùng cho validator/audit (xem `frontend/lexicon.py`).
"""

from __future__ import annotations

# Dạng chính tả dài được kiểm trước. "q" đứng một mình không phải âm
# đầu hợp lệ trong tiếng Việt (chỉ tồn tại trong "qu"); "qa", "qi"... là fallback.
ONSET_SPELLINGS = (
    "ngh", "ng", "gh", "gi", "kh", "nh", "ph", "th", "tr", "ch", "qu",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x",
)

ONSET_ABSTRACT = {
    "": "NONE", "b": "B", "m": "M", "ph": "PH", "v": "V", "t": "T",
    "th": "TH", "đ": "Đ", "n": "N", "d": "D", "gi": "GI", "r": "R",
    "x": "X", "s": "S", "ch": "CH", "tr": "TR", "nh": "NH", "l": "L",
    "c": "K", "k": "K", "qu": "K", "kh": "KH", "ng": "NG",
    "ngh": "NG", "g": "G", "gh": "G", "h": "H", "p": "P",
}

CONSONANT_CODAS = ("nh", "ng", "ch", "m", "n", "p", "t", "c")
CODA_ABSTRACT = {
    "": "NONE", "m": "M", "n": "N", "ng": "NG", "nh": "NH",
    "p": "P", "t": "T", "c": "C", "ch": "CH", "i": "J", "y": "J",
    "u": "W", "o": "W",
}

NUCLEUS_ABSTRACT = {
    "a": "A", "ă": "Ă", "â": "Â", "e": "E", "ê": "Ê", "i": "I", "y": "I",
    "o": "O", "ô": "Ô", "ơ": "Ơ", "u": "U", "ư": "Ư",
    # "oo" tách riêng khỏi "o": người bản ngữ phân biệt "xoong" ≠ "xong"
    # (o dài [ɔː], từ vay/onomatopoeia: xoong, boong, moóc...).
    "oo": "OO",
    "iê": "IÊ", "yê": "IÊ", "ia": "IÊ", "ya": "IÊ",
    "uô": "UÔ", "ua": "UÔ", "ươ": "ƯƠ", "ưa": "ƯƠ",
}

# Ánh xạ cụm nguyên âm sau khi đã bỏ âm đầu và phụ âm cuối.
# Phần tử thứ nhất là chính tả âm đệm, thứ hai là chính tả âm chính.
SPECIAL_VOWELS = {
    "oa": ("o", "a"), "oă": ("o", "ă"), "oe": ("o", "e"),
    "uâ": ("u", "â"), "uê": ("u", "ê"), "uy": ("u", "y"),
    "uya": ("u", "ya"), "uyê": ("u", "yê"), "uơ": ("u", "ơ"),
}

STOP_CODAS = {"P", "T", "C", "CH"}

# Các tổ hợp hợp lệ dùng cho validator, báo cáo coverage và test parity Rust
# về sau. Tính hợp lệ chính tả vẫn nên đối chiếu lexicon sinh từ corpus trước
# khi huấn luyện production.
# W (âm đệm o/u) chỉ kết hợp với: oa, oă, oe, uâ, uê, uy, uyê, uô?, uơ — xác
# minh trên wordlist 39k từ: W+{A,Ă,Â,E,Ê,I,IÊ,Ô,Ơ} (Ô từ "quốc"); không có
# W+O hay W+U trong từ thật.
MEDIAL_NUCLEUS_COMPATIBILITY = {
    "NONE": set(NUCLEUS_ABSTRACT.values()),
    "W": {"A", "Ă", "Â", "E", "Ê", "I", "Ô", "Ơ", "IÊ"},
}

CODA_TONE_COMPATIBILITY = {
    "NONE": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "M": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "N": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "NG": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "NH": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "J": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "W": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "P": {"SAC", "NANG"}, "T": {"SAC", "NANG"},
    "C": {"SAC", "NANG"}, "CH": {"SAC", "NANG"},
}

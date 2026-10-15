"""Tách một âm tiết tiếng Việt thành 5 thành phần (âm đầu/đệm/chính/cuối/thanh).

Từ chữ viết → `ParsedSyllable` (chính tả gốc + dạng âm vị trừu tượng). Mọi luật
chính tả đặc thù (`gi`, `qu`, `uy`, `ay/au`...) nằm trong file này; bảng dữ liệu
nằm ở `rules.py`. Tổng quan: `docs/FRONTEND.md`, `docs/SYLLABLE_GUIDE.md`.
"""

from __future__ import annotations

import re

from vifasttts.frontend.rules import (
    CODA_ABSTRACT,
    CONSONANT_CODAS,
    MEDIAL_NUCLEUS_COMPATIBILITY,
    NUCLEUS_ABSTRACT,
    ONSET_ABSTRACT,
    ONSET_SPELLINGS,
    SPECIAL_VOWELS,
    STOP_CODAS,
)
from vifasttts.frontend.tone import extract_tone
from vifasttts.frontend.unicode import normalize_unicode
from vifasttts.types import AbstractSyllable, Orthography, ParsedSyllable, Tone

_EDGE_PUNCT = re.compile(r"^([^\wÀ-ỹĐđ]*)(.*?)([^\wÀ-ỹĐđ]*)$")
_VOWEL_LETTERS = set("aăâeêioôơuưy")


def _split_edge_punctuation(raw: str) -> tuple[str, str | None]:
    """Tách dấu câu ở hai đầu token, trả (phần lõi, dấu câu) để parse phần lõi."""
    m = _EDGE_PUNCT.match(raw)
    if not m:
        return raw, None
    core = m.group(2)
    punctuation = (m.group(1) + m.group(3)) or None
    return core, punctuation


def _match_onset(word: str) -> tuple[str, str, bool]:
    """Khớp âm đầu dài nhất (ngh/ng/gh/gi/kh/qu...).

    Trả về (chính tả onset, phần còn lại, có phải `qu` không).
    """
    # "gi": chữ i dùng chung giữa onset và vần. Khi sau nó là "ê", hết từ
    # (gì) hoặc một phụ âm (gìn), chữ i thuộc vần => onset chính tả là "g"
    # (âm vị vẫn là GI; xem xử lý trong parse_syllable). Trước nguyên âm khác
    # (già, giữ, giường) thì i là phần của digraph gi (âm /z/), câm.
    if word.startswith("gi") and (len(word) == 2 or word[2] == "ê" or word[2] not in _VOWEL_LETTERS):
        return "g", word[1:], False
    # "qu": chữ u là âm đệm W khi sau nó là nguyên âm KHÔNG tròn môi
    # (qua, quê, quy, quân, quơ...). Riêng "quô" (quốc): âm đệm /w/ không
    # kết hợp với âm tròn môi (u, ô, o) — chữ u thuộc nguyên âm đôi uô, nên
    # onset chính tả chỉ còn "q" (âm vị vẫn là K; xem parse_syllable).
    # Hệ quả: "quốc" = K+UÔ+C, đồng âm "cuốc" (thay cho K+W+Ô+C cũ — góp ý
    # của ThS Đinh Thị Hằng, Viện Ngôn ngữ học).
    if word.startswith("qu"):
        if word[2:3] == "ô":
            return "q", word[1:], False
        return "qu", word[2:], True
    for onset in ONSET_SPELLINGS:
        if word.startswith(onset):
            return onset, word[len(onset):], False
    return "", word, False


def _match_consonant_coda(remainder: str) -> tuple[str, str]:
    """Khớp phụ âm cuối dài nhất (nh/ng/ch...) ở cuối phần vần."""
    for coda in CONSONANT_CODAS:
        if remainder.endswith(coda) and len(remainder) > len(coda):
            return coda, remainder[:-len(coda)]
    return "", remainder


def _parse_vowel(vowel: str, qu_medial: bool) -> tuple[str | None, str, str, str]:
    """Return medial spelling, nucleus spelling, medial abstract, nucleus abstract."""
    if qu_medial:
        # Sau "qu", phần còn lại phải là nucleus hợp lệ (ví dụ "a" trong "qua",
        # "y"/"yê" trong "quy/quyên"). Bản thân "qu" đóng góp âm đệm W.
        # ("quô" không đi vào nhánh này — xem _match_onset: u thuộc âm đôi uô.)
        if vowel not in NUCLEUS_ABSTRACT:
            raise ValueError(f"Phần nguyên âm sau qu không hợp lệ: {vowel!r}")
        return "u", vowel, "W", NUCLEUS_ABSTRACT[vowel]

    # Ưu tiên cụm đặc biệt dài nhất trước.
    for seq in sorted(SPECIAL_VOWELS, key=len, reverse=True):
        if vowel == seq:
            medial, nucleus = SPECIAL_VOWELS[seq]
            return medial, nucleus, "W", NUCLEUS_ABSTRACT[nucleus]

    if vowel in NUCLEUS_ABSTRACT:
        return None, vowel, "NONE", NUCLEUS_ABSTRACT[vowel]
    raise ValueError(f"Cụm nguyên âm không hợp lệ hoặc chưa có quy tắc: {vowel!r}")


def _try_semivowel_coda(vowel_part: str) -> tuple[str, str]:
    """Tách i/y/u/o cuối thành coda trừu tượng J/W nếu phần nucleus còn lại hợp lệ."""
    if len(vowel_part) < 2:
        return "", vowel_part
    last = vowel_part[-1]
    base = vowel_part[:-1]
    if last in "iyuo" and (base in NUCLEUS_ABSTRACT or base in SPECIAL_VOWELS):
        return last, base
    return "", vowel_part


def validate(parsed: ParsedSyllable) -> None:
    """Kiểm bất biến âm vị: có nucleus, âm đệm hợp lệ, coda tắc chỉ sắc/nặng."""
    a = parsed.abstract
    if not a.nucleus or a.nucleus == "NONE":
        raise ValueError("Âm tiết phải có nucleus")
    if a.nucleus not in MEDIAL_NUCLEUS_COMPATIBILITY.get(a.medial, set()):
        raise ValueError(f"Âm đệm {a.medial} không kết hợp với nguyên âm {a.nucleus}")
    if a.coda in STOP_CODAS and a.tone not in {Tone.SAC, Tone.NANG}:
        raise ValueError("Coda tắc P/T/C/CH chỉ kết hợp với thanh sắc hoặc nặng")


def parse_syllable(raw: str) -> ParsedSyllable:
    """Parse một token thành ParsedSyllable (chữ gốc + chính tả + âm vị + dấu câu)."""
    normalized = normalize_unicode(raw.lower())
    core, punctuation = _split_edge_punctuation(normalized)
    if not core:
        raise ValueError("Không có âm tiết để phân tích")
    no_tone, tone, tone_location = extract_tone(core)
    onset_spelling, remainder, qu_medial = _match_onset(no_tone)
    coda_spelling, vowel_part = _match_consonant_coda(remainder)
    if not coda_spelling:
        # Ưu tiên bộ vần đặc biệt trước khi tách bán âm cuối: "uy" phải là
        # medial W + nucleus I (phân biệt với "ui" = nucleus U + coda J).
        try:
            _parse_vowel(vowel_part, qu_medial)
        except ValueError:
            coda_spelling, vowel_part = _try_semivowel_coda(vowel_part)
    medial_s, nucleus_s, medial_a, nucleus_a = _parse_vowel(vowel_part, qu_medial)
    if coda_spelling in ("y", "u") and nucleus_s == "a":
        # "ay", "au" (và "oay") viết bằng a nhưng đọc a NGẮN [ă], trong khi
        # "ai", "ao" đọc a DÀI [a] — xem Help:IPA/Vietnamese: "ai" [aːj] vs
        # "ay" [aj]; "may", "cau" dùng [a], "mai", "cao" dùng [aː].
        # Vì vậy abstract phải là Ă, còn chính tả vẫn giữ "a" để round-trip.
        nucleus_a = NUCLEUS_ABSTRACT["ă"]

    if onset_spelling == "q":
        # "q" trần chỉ xuất hiện khi u thuộc vần ("quốc" = q+uô+c):
        # âm vị đầu vẫn là K (xem _match_onset).
        abstract_onset = "K"
    else:
        abstract_onset = ONSET_ABSTRACT[onset_spelling]
    if onset_spelling == "g" and no_tone.startswith("gi"):
        # Chữ i dùng chung của "gi" (gì, gìn, giếng): chính tả ghi "g",
        # nhưng âm vị là GI (/z/) chứ không phải G (/ɣ/).
        abstract_onset = ONSET_ABSTRACT["gi"]

    parsed = ParsedSyllable(
        raw=raw,
        normalized=normalized,
        orthography=Orthography(
            onset=onset_spelling or None,
            medial=medial_s,
            nucleus=nucleus_s,
            coda=coda_spelling or None,
            tone_location=tone_location,
        ),
        abstract=AbstractSyllable(
            onset=abstract_onset,
            medial=medial_a,
            nucleus=nucleus_a,
            coda=CODA_ABSTRACT[coda_spelling],
            tone=tone,
        ),
        punctuation=punctuation,
    )
    validate(parsed)
    return parsed


def safe_parse_syllable(raw: str) -> ParsedSyllable | None:
    """Như `parse_syllable` nhưng trả None thay vì raise (dùng trong pipeline)."""
    try:
        return parse_syllable(raw)
    except ValueError:
        return None

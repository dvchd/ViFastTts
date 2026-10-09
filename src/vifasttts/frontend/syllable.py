from __future__ import annotations

import re
from dataclasses import replace

from vifasttts.frontend.rules import (
    CODA_ABSTRACT, CONSONANT_CODAS, NUCLEUS_ABSTRACT, ONSET_ABSTRACT,
    ONSET_SPELLINGS, SPECIAL_VOWELS, STOP_CODAS,
)
from vifasttts.frontend.tone import extract_tone
from vifasttts.frontend.unicode import normalize_unicode
from vifasttts.types import AbstractSyllable, Orthography, ParsedSyllable, ParseStatus, Tone

_EDGE_PUNCT = re.compile(r"^([^\wÀ-ỹĐđ]*)(.*?)([^\wÀ-ỹĐđ]*)$")


def _split_edge_punctuation(raw: str) -> tuple[str, str | None]:
    m = _EDGE_PUNCT.match(raw)
    if not m:
        return raw, None
    core = m.group(2)
    punctuation = (m.group(1) + m.group(3)) or None
    return core, punctuation


def _match_onset(word: str) -> tuple[str, str, bool]:
    for onset in ONSET_SPELLINGS:
        if word.startswith(onset):
            if onset == "qu":
                return onset, word[2:], True
            return onset, word[len(onset):], False
    return "", word, False


def _match_consonant_coda(remainder: str) -> tuple[str, str]:
    for coda in CONSONANT_CODAS:
        if remainder.endswith(coda) and len(remainder) > len(coda):
            return coda, remainder[:-len(coda)]
    return "", remainder


def _parse_vowel(vowel: str, qu_medial: bool) -> tuple[str | None, str, str, str]:
    """Return medial spelling, nucleus spelling, medial abstract, nucleus abstract."""
    if qu_medial:
        # After qu, remainder must be a legal nucleus (e.g. "a" in "qua",
        # "y"/"ye" in "quy/quyen"). qu itself contributes medial W.
        if vowel not in NUCLEUS_ABSTRACT:
            raise ValueError(f"Phần nguyên âm sau qu không hợp lệ: {vowel!r}")
        return "u", vowel, "W", NUCLEUS_ABSTRACT[vowel]

    # Longest special sequence first.
    for seq in sorted(SPECIAL_VOWELS, key=len, reverse=True):
        if vowel == seq:
            medial, nucleus = SPECIAL_VOWELS[seq]
            return medial, nucleus, "W", NUCLEUS_ABSTRACT[nucleus]

    if vowel in NUCLEUS_ABSTRACT:
        return None, vowel, "NONE", NUCLEUS_ABSTRACT[vowel]
    raise ValueError(f"Cụm nguyên âm không hợp lệ hoặc chưa có quy tắc: {vowel!r}")


def _try_semivowel_coda(vowel_part: str) -> tuple[str, str]:
    """Split final i/y/u/o as abstract J/W when the remaining nucleus is legal."""
    if len(vowel_part) < 2:
        return "", vowel_part
    last = vowel_part[-1]
    base = vowel_part[:-1]
    if last in "iyuo" and (base in NUCLEUS_ABSTRACT or base in SPECIAL_VOWELS):
        return last, base
    return "", vowel_part


def validate(parsed: ParsedSyllable) -> None:
    a = parsed.abstract
    if not a.nucleus or a.nucleus == "NONE":
        raise ValueError("Âm tiết phải có nucleus")
    if a.coda in STOP_CODAS and a.tone not in {Tone.SAC, Tone.NANG}:
        raise ValueError("Coda tắc P/T/C/CH chỉ kết hợp với thanh sắc hoặc nặng")


def parse_syllable(raw: str) -> ParsedSyllable:
    normalized = normalize_unicode(raw.lower())
    core, punctuation = _split_edge_punctuation(normalized)
    if not core:
        raise ValueError("Không có âm tiết để phân tích")
    no_tone, tone, tone_location = extract_tone(core)
    onset_spelling, remainder, qu_medial = _match_onset(no_tone)
    coda_spelling, vowel_part = _match_consonant_coda(remainder)
    if not coda_spelling:
        coda_spelling, vowel_part = _try_semivowel_coda(vowel_part)
    medial_s, nucleus_s, medial_a, nucleus_a = _parse_vowel(vowel_part, qu_medial)

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
            onset=ONSET_ABSTRACT[onset_spelling],
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
    try:
        return parse_syllable(raw)
    except ValueError:
        return None

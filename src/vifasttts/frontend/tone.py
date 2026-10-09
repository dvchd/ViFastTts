from __future__ import annotations

import unicodedata
from vifasttts.types import Tone

TONE_MARKS = {
    "\u0301": Tone.SAC,
    "\u0300": Tone.HUYEN,
    "\u0309": Tone.HOI,
    "\u0303": Tone.NGA,
    "\u0323": Tone.NANG,
}

QUALITY_MARKS = {"\u0302", "\u0306", "\u031B"}


def extract_tone(syllable: str) -> tuple[str, Tone, str | None]:
    """Remove only tone marks, preserving ă â ê ô ơ ư.

    Returns tone-stripped NFC text, tone and the vowel carrying the tone.
    """
    nfd = unicodedata.normalize("NFD", syllable)
    result: list[str] = []
    tone = Tone.NGANG
    tone_location: str | None = None
    last_base: str | None = None
    seen: set[Tone] = set()

    for ch in nfd:
        if not unicodedata.combining(ch):
            last_base = ch
            result.append(ch)
            continue
        if ch in TONE_MARKS:
            seen.add(TONE_MARKS[ch])
            tone = TONE_MARKS[ch]
            tone_location = last_base
        else:
            result.append(ch)

    if len(seen) > 1:
        raise ValueError(f"Âm tiết có nhiều dấu thanh: {syllable!r}")
    return unicodedata.normalize("NFC", "".join(result)), tone, tone_location

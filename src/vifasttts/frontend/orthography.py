from __future__ import annotations

import unicodedata
from vifasttts.types import Tone

TONE_COMBINING = {
    Tone.NGANG: "",
    Tone.HUYEN: "\u0300",
    Tone.SAC: "\u0301",
    Tone.HOI: "\u0309",
    Tone.NGA: "\u0303",
    Tone.NANG: "\u0323",
}


def _tone_target(nucleus: str) -> int:
    """Return index in grapheme nucleus for canonical main-vowel placement."""
    # Open variants ia, ua, ưa carry tone on the first vowel.
    if nucleus.lower() in {"ia", "ya", "ua", "ưa"}:
        return 0
    # Quality-marked main vowel has priority.
    for char in "êôơưăâÊÔƠƯĂÂ":
        if char in nucleus:
            return nucleus.index(char)
    # Canonical diphthongs and triphthongs.
    for char in "aAeEoOiIuUyY":
        if char in nucleus:
            return nucleus.index(char)
    return 0


def apply_tone(nucleus: str, tone: Tone) -> str:
    if tone is Tone.NGANG:
        return nucleus
    i = _tone_target(nucleus)
    nfd = unicodedata.normalize("NFD", nucleus[i])
    return nucleus[:i] + unicodedata.normalize("NFC", nfd + TONE_COMBINING[tone]) + nucleus[i + 1:]


def compose_canonical(onset: str | None, medial: str | None, nucleus: str, coda: str | None, tone: Tone) -> str:
    onset = onset or ""
    # qu already contains the written u that represents medial W.
    written_medial = "" if onset == "qu" else (medial or "")
    # Canonical "uy" puts tone on final y: thuy -> thuy with tone on y (thuy),
    # tuy -> tuy with tone on y (tuy). Parser yields nucleus "u" + coda "y",
    # so move tone to coda for this specific pattern.
    if (
        tone is not Tone.NGANG
        and coda is not None
        and coda.lower() in ("y", "i")
        and nucleus.lower() == "u"
    ):
        return onset + written_medial + nucleus + apply_tone(coda, tone)
    return onset + written_medial + apply_tone(nucleus, tone) + (coda or "")

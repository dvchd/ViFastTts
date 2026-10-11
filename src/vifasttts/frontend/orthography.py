"""Ghép chính tả canonical: đặt dấu thanh đúng âm chính và dựng lại từ các phần."""

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
    # Từ vay "oo" (xoong, moóc): dấu đặt trên chữ o thứ hai.
    if nucleus.lower() == "oo":
        return 1
    # Open variants ia, ua, ưa carry tone on the first vowel.
    if nucleus.lower() in {"ia", "ya", "ua", "ưa"}:
        return 0
    # Nguyên âm chính có dấu chất lượng được ưu tiên nhận thanh.
    for char in "êôơưăâÊÔƠƯĂÂ":
        if char in nucleus:
            return nucleus.index(char)
    # Nguyên âm đôi/ba chuẩn.
    for char in "aAeEoOiIuUyY":
        if char in nucleus:
            return nucleus.index(char)
    return 0


def apply_tone(nucleus: str, tone: Tone) -> str:
    """Đặt dấu thanh vào đúng chữ âm chính của vần (NFC, giữ dấu chất lượng)."""
    if tone is Tone.NGANG:
        return nucleus
    i = _tone_target(nucleus)
    nfd = unicodedata.normalize("NFD", nucleus[i])
    return nucleus[:i] + unicodedata.normalize("NFC", nfd + TONE_COMBINING[tone]) + nucleus[i + 1:]


def compose_canonical(onset: str | None, medial: str | None, nucleus: str, coda: str | None, tone: Tone) -> str:
    """Dựng chính tả canonical từ 5 phần rời (đã chuẩn hoá vị trí dấu thanh)."""
    onset = onset or ""
    # "qu" đã chứa sẵn chữ u viết cho âm đệm W, không ghi thêm lần nữa.
    written_medial = "" if onset == "qu" else (medial or "")
    # "uy" được parse thành medial W + nucleus I (xem syllable.py), nên dấu
    # thanh tự động rơi trên nucleus "y" — đúng chuẩn hiện đại (thuỷ, tuỳ).
    return onset + written_medial + apply_tone(nucleus, tone) + (coda or "")

"""Property test: mọi tổ hợp (âm đầu × vần × âm cuối × thanh) đều ổn định.

Sinh tổ hợp bằng `compose_canonical` rồi kiểm tra bất biến
`parse → compose → parse` (không FALLBACK, không trôi abstract).
Cùng logic với `tools/parser_audit.py`, chạy như test thường trực.
"""
from __future__ import annotations

from itertools import product

from vifasttts.frontend.orthography import compose_canonical
from vifasttts.frontend.rules import (
    CODA_ABSTRACT,
    NUCLEUS_ABSTRACT,
    ONSET_ABSTRACT,
    SPECIAL_VOWELS,
)
from vifasttts.frontend.syllable import _parse_vowel, safe_parse_syllable
from vifasttts.types import Tone

STOP_CODAS = {"p", "t", "c", "ch"}


def test_full_matrix_is_stable():
    problems: list[str] = []
    checked = 0
    onsets = [onset for onset in ONSET_ABSTRACT if onset]
    vowel_parts = list(NUCLEUS_ABSTRACT) + list(SPECIAL_VOWELS)
    for onset, vowel_part, coda, tone in product(onsets, vowel_parts, list(CODA_ABSTRACT), list(Tone)):
        if coda in STOP_CODAS and tone not in (Tone.SAC, Tone.NANG):
            continue
        try:
            medial_s, nucleus_s, _, _ = _parse_vowel(vowel_part, onset == "qu")
        except ValueError:
            continue
        checked += 1
        word = compose_canonical(onset, medial_s, nucleus_s, coda, tone)
        parsed = safe_parse_syllable(word)
        if parsed is None:
            problems.append(f"FALLBACK {word!r}")
            continue
        first = parsed.abstract
        canonical = compose_canonical(
            parsed.orthography.onset, parsed.orthography.medial,
            parsed.orthography.nucleus, parsed.orthography.coda, first.tone,
        )
        again = safe_parse_syllable(canonical)
        if again is None or again.abstract != first:
            problems.append(f"DRIFT {word!r} -> {canonical!r}")
            continue
        recanonical = compose_canonical(
            again.orthography.onset, again.orthography.medial,
            again.orthography.nucleus, again.orthography.coda, again.abstract.tone,
        )
        if recanonical != canonical:
            problems.append(f"UNSTABLE {word!r} -> {canonical!r} -> {recanonical!r}")
    assert checked > 40_000, checked
    assert not problems, problems[:20]

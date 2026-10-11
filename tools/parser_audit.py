"""Audit parser: vét cạn tổ hợp âm tiết và kiểm tra bất biến chuẩn hoá.

Sinh mọi tổ hợp (âm đầu × vần × âm cuối × thanh) bằng `compose_canonical`,
parse lại, rồi kiểm tra:
- không tổ hợp hợp lệ nào rơi vào FALLBACK;
- `parse → compose → parse` ổn định (cùng abstract, cùng canonical).

Chạy: python tools/parser_audit.py
Thoát mã 1 nếu có vấn đề (dùng được trong CI).
"""
from __future__ import annotations

import sys
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


def run() -> list[str]:
    problems: list[str] = []
    total = 0
    onsets = [onset for onset in ONSET_ABSTRACT if onset]
    vowel_parts = list(NUCLEUS_ABSTRACT) + list(SPECIAL_VOWELS)
    for onset, vowel_part, coda, tone in product(onsets, vowel_parts, list(CODA_ABSTRACT), list(Tone)):
        if coda in STOP_CODAS and tone not in (Tone.SAC, Tone.NANG):
            continue
        try:
            medial_s, nucleus_s, _, _ = _parse_vowel(vowel_part, onset == "qu")
        except ValueError:
            continue  # vần không hợp lệ sau qu
        word = compose_canonical(onset, medial_s, nucleus_s, coda, tone)
        parsed = safe_parse_syllable(word)
        total += 1
        if parsed is None:
            problems.append(f"FALLBACK: {word!r} ({onset}|{vowel_part}|{coda}|{tone.value})")
            continue
        first = parsed.abstract
        canonical = compose_canonical(
            parsed.orthography.onset, parsed.orthography.medial,
            parsed.orthography.nucleus, parsed.orthography.coda, first.tone,
        )
        again = safe_parse_syllable(canonical)
        if again is None:
            problems.append(f"REPARSE-FAIL: {word!r} -> {canonical!r}")
            continue
        if again.abstract != first:
            problems.append(f"ABSTRACT-DRIFT: {word!r} -> {canonical!r}: {first} != {again.abstract}")
            continue
        recanonical = compose_canonical(
            again.orthography.onset, again.orthography.medial,
            again.orthography.nucleus, again.orthography.coda, again.abstract.tone,
        )
        if recanonical != canonical:
            problems.append(f"UNSTABLE: {word!r} -> {canonical!r} -> {recanonical!r}")
    print(f"Đã kiểm {total} tổ hợp; {len(problems)} vấn đề")
    return problems


if __name__ == "__main__":
    found = run()
    for item in found[:20]:
        print(" ", item)
    sys.exit(1 if found else 0)

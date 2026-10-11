"""Audit parser: ma trận tổ hợp âm tiết và/hoặc danh sách từ thật.

Chạy:
    python tools/parser_audit.py                     # ma trận ~49k tổ hợp
    python tools/parser_audit.py --wordlist FILE     # audit trên 1 từ/dòng
    python tools/parser_audit.py --wordlist FILE --profile strict
    python tools/parser_audit.py --wordlist FILE --write-lexicon lex.json
    python tools/parser_audit.py --wordlist FILE --profile corpus --lexicon lex.json

Chế độ ma trận kiểm tra bất biến `parse → compose → parse`; thoát mã 1 nếu có
vấn đề (dùng được trong CI). Chế độ từ điển in fallback, nhóm va chạm, coverage
và (tuỳ chọn) cảnh báo theo profile open/strict/corpus.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from itertools import product

from vifasttts.frontend.orthography import compose_canonical
from vifasttts.frontend.rules import (
    CODA_ABSTRACT,
    CODA_TONE_COMPATIBILITY,
    MEDIAL_NUCLEUS_COMPATIBILITY,
    NUCLEUS_ABSTRACT,
    ONSET_ABSTRACT,
    SPECIAL_VOWELS,
)
from vifasttts.frontend.syllable import _parse_vowel, safe_parse_syllable
from vifasttts.types import Tone

_WORD_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+")


def run_matrix() -> list[str]:
    """Vét cạn mọi tổ hợp hợp lệ và kiểm bất biến; trả danh sách vấn đề (rỗng = đạt)."""
    problems: list[str] = []
    total = 0
    onsets = [onset for onset in ONSET_ABSTRACT if onset]
    vowel_parts = list(NUCLEUS_ABSTRACT) + list(SPECIAL_VOWELS)
    for onset, vowel_part, coda, tone in product(onsets, vowel_parts, list(CODA_ABSTRACT), list(Tone)):
        if tone.value not in CODA_TONE_COMPATIBILITY[CODA_ABSTRACT[coda]]:
            continue
        try:
            medial_s, nucleus_s, medial_a, nucleus_a = _parse_vowel(vowel_part, onset == "qu")
        except ValueError:
            continue  # vần không hợp lệ sau qu
        if nucleus_a not in MEDIAL_NUCLEUS_COMPATIBILITY[medial_a]:
            continue  # ví dụ qu + o/u: không hợp lệ (xem rules.py)
        word = compose_canonical(onset, medial_s, nucleus_s, coda, tone)
        parsed = safe_parse_syllable(word)
        total += 1
        if parsed is None:
            problems.append(f"FALLBACK: {word!r} ({onset}|{vowel_part}|{coda}|{tone.value})")
            continue
        first = parsed.abstract
        if first.nucleus not in MEDIAL_NUCLEUS_COMPATIBILITY[first.medial]:
            problems.append(f"MEDIAL-NUCLEUS: {word!r}: {first.medial}+{first.nucleus}")
        if first.tone.value not in CODA_TONE_COMPATIBILITY[first.coda]:
            problems.append(f"CODA-TONE: {word!r}: {first.coda}+{first.tone.value}")
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
    print(f"Ma trận: đã kiểm {total} tổ hợp; {len(problems)} vấn đề")
    return problems


def load_wordlist(path: str) -> set[str]:
    """Đọc file (mỗi dòng có thể chứa nhiều từ) thành tập token chữ Latin."""
    syllables: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            for token in re.split(r"[\s\-]+", line.strip()):
                if _WORD_RE.fullmatch(token):
                    syllables.add(token)
    return syllables


def run_wordlist(path: str) -> set[str]:
    """Audit ngữ liệu: fallback, va chạm tuple, coverage; trả tập token đã đọc."""
    syllables = load_wordlist(path)
    ok: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    fallback: list[str] = []
    for syllable in syllables:
        parsed = safe_parse_syllable(syllable)
        if parsed is None:
            fallback.append(syllable)
            continue
        a = parsed.abstract
        ok[(a.onset, a.medial, a.nucleus, a.coda, a.tone.value)].add(syllable)
    collisions = {key: value for key, value in ok.items() if len(value) > 1}
    print(f"Wordlist: {len(syllables)} âm tiết duy nhất | "
          f"parse OK: {len(syllables) - len(fallback)} | fallback: {len(fallback)}")
    print("Fallback:", sorted(fallback))
    print(f"Nhóm va chạm (nhiều chữ cùng tuple): {len(collisions)}")
    for key, words in sorted(collisions.items()):
        print("  ", key, sorted(words))

    # Coverage: ngữ liệu phủ bao nhiêu rime/shape, và phần parser "nhận rộng"
    accepted_rimes = {
        (medial, nucleus, coda)
        for medial, nuclei in MEDIAL_NUCLEUS_COMPATIBILITY.items()
        for nucleus in nuclei
        for coda in CODA_ABSTRACT.values()
    }
    seen_rimes = {(medial, nucleus, coda) for (_o, medial, nucleus, coda, _t) in ok}
    seen_shapes = {(onset, medial, nucleus, coda) for (onset, medial, nucleus, coda, _t) in ok}
    shape_words: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for (onset, medial, nucleus, coda, _tone), words in ok.items():
        shape_words[(onset, medial, nucleus, coda)].update(words)
    hapax = sum(1 for words in shape_words.values() if len(words) == 1)
    missing = sorted(accepted_rimes - seen_rimes)
    print(f"Coverage: shape (onset×rime) {len(seen_shapes)} | rime {len(seen_rimes)}/{len(accepted_rimes)} | "
          f"shape chỉ có 1 âm tiết: {hapax}")
    if missing:
        preview = ", ".join(f"{m}+{n}+{c}" for m, n, c in missing[:12])
        print(f"Rime bảng chấp nhận nhưng ngữ liệu chưa từng thấy ({len(missing)}): {preview}"
              + (" ..." if len(missing) > 12 else ""))
    return syllables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wordlist", help="File văn bản, mỗi dòng một từ/câu")
    parser.add_argument("--profile", choices=("open", "strict", "corpus"),
                        help="Kiểm luật chính tả/ngữ âm trên wordlist")
    parser.add_argument("--lexicon", help="File lexicon .json (cho profile corpus)")
    parser.add_argument("--write-lexicon", help="Ghi lexicon shape sinh từ wordlist")
    args = parser.parse_args()
    if args.wordlist:
        syllables = run_wordlist(args.wordlist)
        if args.write_lexicon or args.profile:
            from vifasttts.frontend.lexicon import Lexicon, profile_report

            if args.write_lexicon:
                lexicon = Lexicon.from_words(syllables)
                lexicon.save(args.write_lexicon)
                print(f"Lexicon: đã ghi {args.write_lexicon} "
                      f"({len(lexicon.shapes)} shape, {len(lexicon.rimes)} vần)")
            if args.profile:
                lexicon = Lexicon.load(args.lexicon) if args.lexicon else None
                if args.profile == "corpus" and lexicon is None:
                    print("--profile corpus cần --lexicon FILE (hoặc dùng --write-lexicon)",
                          file=sys.stderr)
                    sys.exit(2)
                report = profile_report(syllables, args.profile, lexicon)
                print(f"Profile {args.profile}: {report['total']} từ | cảnh báo: {report['flagged']}")
                for code, count in sorted(report["issues"].items(), key=lambda item: -item[1]):
                    examples = report["examples"].get(code, [])[:6]
                    print(f"  {code}: {count} vd={examples}")
        sys.exit(0)
    found = run_matrix()
    for item in found[:20]:
        print(" ", item)
    sys.exit(1 if found else 0)

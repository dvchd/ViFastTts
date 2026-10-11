"""Benchmark tầng biểu diễn frontend: char vs 5 slot vs phoneme vs vần.

So sánh định lượng trên cùng văn bản (không cần huấn luyện): số đơn vị mỗi
âm tiết, độ dài chuỗi theo câu (mean/p95) và kích thước vocab từng biểu diễn.
Dùng để chọn đơn vị đầu vào cho model và theo dõi "độ phình" khi đổi view
(xem docs/SYLLABLE_SYSTEMS_RESEARCH.md mục 6).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from vifasttts.frontend.dialect import Dialect
from vifasttts.frontend.g2p import syllable_to_phonemes
from vifasttts.frontend.pipeline import parse_text
from vifasttts.types import ParseStatus


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def representation_stats(
    texts: Iterable[str],
    dialect: str | Dialect = Dialect.NORTH,
) -> dict[str, Any]:
    """Thống kê các cách biểu diễn trên một tập câu."""
    counts: Counter[str] = Counter()
    units: dict[str, int] = Counter()
    sentence_lengths: dict[str, list[int]] = defaultdict(list)
    vocab: dict[str, Counter] = {
        "char": Counter(), "slot": Counter(), "phoneme": Counter(), "rime": Counter(),
    }
    sentences = 0
    for text in texts:
        if not text.strip():
            continue
        sentences += 1
        per_sentence: dict[str, int] = Counter()
        for item in parse_text(text, dialect=dialect):
            counts["syllables"] += 1
            if item.status != ParseStatus.VALID:
                counts["fallback"] += 1
                counts["tokens"] += 1
                per_sentence["char"] += max(1, len(item.raw))
                per_sentence["slot"] += 5
                per_sentence["phoneme"] += 0
                per_sentence["rime"] += 2
                continue
            counts["tokens"] += 1
            parsed_units = {
                "char": len(item.normalized),
                "slot": 5,
                "phoneme": len(syllable_to_phonemes(item, dialect)),
                "rime": 2,  # (vần, thanh)
            }
            for name, value in parsed_units.items():
                units[name] += value
                per_sentence[name] += value
            a = item.abstract
            vocab["char"].update(item.normalized)
            vocab["slot"].update((a.onset, a.medial, a.nucleus, a.coda, a.tone.value))
            vocab["phoneme"].update(syllable_to_phonemes(item, dialect))
            vocab["rime"].update({("+".join(a.rime), a.tone.value)})
        for name, value in per_sentence.items():
            sentence_lengths[name].append(value)

    syllable_count = max(1, counts["tokens"])
    representations: dict[str, dict[str, Any]] = {}
    for name in ("char", "slot", "phoneme", "rime"):
        representations[name] = {
            "units_total": units[name],
            "units_per_syllable": round(units[name] / syllable_count, 3),
            "p95_units_per_sentence": _p95(sentence_lengths[name]),
            "vocab_size": len(vocab[name]),
        }
    return {
        "sentences": sentences,
        "tokens": counts["tokens"],
        "syllables": counts["syllables"],
        "fallback": counts["fallback"],
        "representations": representations,
    }


def format_stats(stats: dict[str, Any]) -> str:
    """Render bảng gọn cho CLI/log."""
    lines = [
        f"Câu: {stats['sentences']} | token: {stats['tokens']} | fallback: {stats['fallback']}",
        f"{'biểu diễn':<10} {'/âm tiết':>8} {'p95/câu':>8} {'vocab':>6}",
    ]
    for name, values in stats["representations"].items():
        lines.append(
            f"{name:<10} {values['units_per_syllable']:>8} "
            f"{values['p95_units_per_sentence']:>8} {values['vocab_size']:>6}"
        )
    return "\n".join(lines)

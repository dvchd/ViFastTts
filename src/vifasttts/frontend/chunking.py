"""Chia văn bản dài thành đoạn để tổng hợp (kèm gợi ý khoảng nghỉ).

Câu dài một hơi dễ méo và tràn bộ nhớ; tách theo câu → mệnh đề → từ, giữ
dấu câu trong đoạn để normalizer/parser vẫn thấy ngữ cảnh.
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])[\s]+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:])[\s]+")
_SENTENCE_END = ".!?…"
_CLAUSE_END = ",;:"


def pause_seconds(chunk: str) -> float:
    """Khoảng nghỉ gợi ý sau một đoạn, theo dấu câu cuối."""
    stripped = chunk.rstrip()
    if not stripped:
        return 0.0
    last = stripped[-1]
    if last in _SENTENCE_END:
        return 0.15
    if last in _CLAUSE_END:
        return 0.08
    return 0.05


def _pack(parts: list[str], max_chars: int) -> list[str]:
    """Gộp các phần nhỏ liền nhau cho tới `max_chars`."""
    chunks: list[str] = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= max_chars:
            current = f"{current} {part}"
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """Chia `text` thành các đoạn ≤ `max_chars` (xấp xỉ), ưu tiên ranh giới câu.

    Trả về `[]` cho chuỗi rỗng/khoảng trắng.
    """
    if max_chars <= 0:
        raise ValueError("max_chars phải > 0")
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    pieces: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(normalized):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        for clause in _CLAUSE_SPLIT.split(sentence):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) <= max_chars:
                pieces.append(clause)
                continue
            buffer = ""
            for word in clause.split(" "):
                if not word:
                    continue
                if buffer and len(buffer) + 1 + len(word) > max_chars:
                    pieces.append(buffer)
                    buffer = word
                else:
                    buffer = f"{buffer} {word}".strip()
            if buffer:
                pieces.append(buffer)
    return _pack(pieces, max_chars)

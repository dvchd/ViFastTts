"""Test chunk văn bản dài và khoảng nghỉ theo dấu câu."""
from __future__ import annotations

import pytest

from vifasttts.frontend.chunking import (
    chunk_text,
    pause_seconds,
    split_clauses,
    steady_pause_seconds,
)


def test_short_text_single_chunk():
    assert chunk_text("Chào bạn") == ["Chào bạn"]
    assert chunk_text("  ") == []


def test_sentence_split_and_merge():
    text = "Câu một. Câu hai! Câu ba? Câu bốn."
    # max_chars nhỏ: mỗi câu một đoạn.
    assert chunk_text(text, max_chars=15) == ["Câu một.", "Câu hai!", "Câu ba?", "Câu bốn."]
    # max_chars lớn: gộp lại thành một đoạn.
    assert chunk_text(text, max_chars=100) == [text]


def test_long_sentence_splits_at_clause_then_word():
    sentence = "Đây là một câu rất dài, " + " ".join(f"từ{i}" for i in range(40)) + "."
    chunks = chunk_text(sentence, max_chars=40)
    assert len(chunks) > 2
    assert all(len(chunk) <= 40 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ") == sentence.replace("  ", " ")


def test_invalid_max_chars():
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=0)


def test_pause_seconds_by_punctuation():
    assert pause_seconds("Câu một.") == 0.15
    assert pause_seconds("rồi,") == 0.08
    assert pause_seconds("cuối cùng") == 0.05
    assert pause_seconds("   ") == 0.0
    assert pause_seconds('Kết thúc…') == 0.15


def test_split_clauses_keeps_punctuation():
    text = "Câu một. Câu hai, vế ba; vế bốn: hết! Sao?"
    assert split_clauses(text) == ["Câu một.", "Câu hai,", "vế ba;", "vế bốn:", "hết!", "Sao?"]
    assert split_clauses("   ") == []
    # mảnh quá dài vẫn bị cắt tiếp cho vừa max_chars
    long = "một hai ba " * 30 + "."
    pieces = split_clauses(long, max_chars=40)
    assert all(len(piece) <= 40 for piece in pieces)


def test_steady_pause_seconds():
    assert steady_pause_seconds("Câu một.") == 0.55
    assert steady_pause_seconds("rồi,") == 0.25
    assert steady_pause_seconds("hết;") == 0.25
    assert steady_pause_seconds("Sao?") == 0.55
    assert steady_pause_seconds("cuối cùng") == 0.08
    assert steady_pause_seconds("   ") == 0.0
    # tuỳ chỉnh
    assert steady_pause_seconds("rồi,", comma=0.3) == 0.3
    assert steady_pause_seconds("xong.", period=0.7) == 0.7
    assert steady_pause_seconds("không dấu", default=0.12) == 0.12

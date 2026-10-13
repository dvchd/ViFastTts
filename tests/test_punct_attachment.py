"""Test gắn dấu câu vào âm tiết — ghi lại luật hiện tại (Addendum 4 §2.4, Addendum 5 §4).

Luật (pipeline.py:62-90, vocab.py:47) — ghi vào test để không đổi ngầm:
- Dấu SAU gắn vào âm tiết đứng trước; token cuối ∈ ".!?" → boundary SENTENCE,
  ",;:" → PHRASE, còn lại giữ WORD.
- Dấu MỞ ĐẦU (pending) gắn vào âm tiết đứng sau (dạng tiền tố).
- OOV respell: dấu giữ trên âm tiết cuối của token respell.
- Nhiều dấu liền nhau gộp một nhãn theo ưu tiên ? > ! > , > . > OTHER
  ("?!" → QUESTION, "..." → PERIOD); boundary theo token cuối cùng.
- Ghi nhận cho v2.1: "..." giữa câu hiện cũng thành SENTENCE (có thể làm pause
  predictor học nghỉ dài ở chỗ lẽ ra nghỉ ngắn — cùng SIL token).
"""

from __future__ import annotations

from vifasttts.frontend.normalizer import NormalizerOptions
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.vocab import ComponentVocabs

OPTIONS = NormalizerOptions(dialect="north")
VOCABS = ComponentVocabs.default()
_PUNCT_INVERSE = {value: key for key, value in VOCABS.punctuation.items()}


def _parse(text: str, resolve_oov: bool = False):
    """Parse + assert không sinh âm tiết rỗng."""
    items = parse_text(text, dialect="north", options=OPTIONS, resolve_oov=resolve_oov)
    assert all(item.normalized for item in items), "có âm tiết rỗng"
    return items


def _punct_id(item) -> str:
    """Tên nhãn punctuation sau mã hoá (COMMA/PERIOD/QUESTION/...)."""
    return _PUNCT_INVERSE[VOCABS.encode(item)[6]]


def test_dau_sau_gan_am_tiet_truoc():
    """'Xin chào, bạn khỏe không?' — dấu phẩy ở 'chào', dấu hỏi ở 'không'."""
    items = _parse("Xin chào, bạn khỏe không?")
    assert [item.normalized for item in items] == ["xin", "chào", "bạn", "khoẻ", "không"]
    assert items[1].punctuation == ","
    assert items[1].boundaries.phrase_end and not items[1].boundaries.sentence_end
    assert _punct_id(items[1]) == "COMMA"
    assert items[4].punctuation == "?"
    assert items[4].boundaries.sentence_end
    assert _punct_id(items[4]) == "QUESTION"


def test_dau_mo_dau_gan_am_tiet_sau():
    """Dấu mở đầu (pending) gắn vào âm tiết đứng sau, dạng tiền tố."""
    items = _parse('"Xin chào" bạn')
    assert items[0].punctuation == '"' and items[1].punctuation == '"'
    assert items[2].punctuation is None
    assert _punct_id(items[0]) == "OTHER"


def test_oov_respell_giu_dau():
    """OOV respell: 'wifi' → 'quai phai', dấu phẩy nằm ở âm tiết cuối token respell."""
    items = _parse("wifi, bạn khỏe không.", resolve_oov=True)
    assert [item.normalized for item in items][:2] == ["quai", "phai"]
    assert items[1].punctuation == ","
    assert _punct_id(items[1]) == "COMMA"
    assert items[-1].punctuation == "."
    assert _punct_id(items[-1]) == "PERIOD"


def test_nhieu_dau_lien_nhau():
    """'?!' → QUESTION/SENTENCE; '...' → PERIOD/SENTENCE; không âm tiết rỗng."""
    items = _parse("Thật sao?!")
    assert items[-1].punctuation == "?!"
    assert _punct_id(items[-1]) == "QUESTION"
    assert items[-1].boundaries.sentence_end

    items = _parse("Ừ...")
    assert items[-1].punctuation == "..."
    assert _punct_id(items[-1]) == "PERIOD"
    assert items[-1].boundaries.sentence_end

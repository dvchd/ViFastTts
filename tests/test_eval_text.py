"""Test hàm dựng text eval bảo toàn dấu câu (v18.37)."""

from __future__ import annotations

import pytest

from vifasttts.eval_text import build_eval_text, legacy_raw_join, text_record
from vifasttts.frontend.custom_reading import apply_custom_reading
from vifasttts.frontend.pipeline import parse_text
from vifasttts.infer import Synthesizer


def _row(*items):
    return {"syllables": [{"raw": raw, "normalized": raw.lower(), "punctuation": punct}
                          for raw, punct in items]}


def test_giu_dau_phay_cham_hoi():
    row = _row(("Chào", None), ("bạn", ","), ("hôm", None), ("nay", None),
               ("thế", None), ("nào", "?"), ("Tôi", None), ("ổn", "."))
    text = build_eval_text(row)
    assert text == "Chào bạn, hôm nay thế nào? Tôi ổn."
    parsed = parse_text(text, resolve_oov=True)
    assert [item.punctuation for item in parsed] == [None, ",", None, None, None, "?", None, "."]


def test_khong_dau_cau():
    row = _row(("Xin", None), ("chào", None), ("các", None), ("bạn", None))
    assert build_eval_text(row) == "Xin chào các bạn"
    assert legacy_raw_join(row) == "Xin chào các bạn"


def test_khong_lap_dau_cau():
    row = _row(("bạn.", "."), ("nhé", "!"), ("ừ", "?!"))
    assert build_eval_text(row) == "bạn. nhé! ừ?!"
    assert legacy_raw_join(row) == "bạn. nhé ừ"


def test_dau_cau_nhieu_ky_tu_va_token_trong():
    row = {"syllables": [
        {"raw": "một", "punctuation": "++"},
        {"raw": "", "normalized": "hai", "punctuation": "."},
        {"raw": None, "normalized": "ba", "punctuation": None},
    ]}
    assert build_eval_text(row) == "một++ hai. ba"
    assert legacy_raw_join(row) == "một hai ba"


def test_custom_reading_va_thu_tu():
    row = _row(("Dùng", None), ("VBA", ","), ("và", None), ("VB", "."))
    text = build_eval_text(row)
    assert text == "Dùng VBA, và VB."
    rewritten = apply_custom_reading(text)
    assert rewritten == "Dùng vi be a, và vi be."
    parsed = parse_text(rewritten, resolve_oov=True)
    assert [item.normalized for item in parsed] == ["dùng", "vi", "be", "a", "và", "vi", "be"]


def test_thieu_token_bao_loi():
    with pytest.raises(ValueError):
        build_eval_text({"syllables": [{"raw": None, "normalized": None, "punctuation": "."}]})


def test_text_record_co_hai_duong():
    row = _row(("Chào", None), ("bạn", "."))
    record = text_record(row)
    assert record == {"text": "Chào bạn.", "text_legacy": "Chào bạn", "text_mode": "punctuation"}


def test_sentence_gaps_theo_dau_cau():
    items = parse_text("Chào bạn, hôm nay thế nào? Tôi ổn.")
    gaps = Synthesizer.sentence_gaps(items, 24000, 256)
    assert gaps.tolist() == [0, 23, 0, 0, 0, 52, 0, 0]
    assert Synthesizer.sentence_gaps(parse_text("Xin chào các bạn"), 24000, 256).tolist() == [0, 0, 0, 0]

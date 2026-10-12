"""Test bộ câu eval cố định `data/eval/cau_eval_v1.txt` (đủ 6 thanh + coda tắc, không fallback)."""
from __future__ import annotations

from pathlib import Path

from vifasttts.frontend.pipeline import parse_text

EVAL_FILE = Path(__file__).resolve().parents[1] / "data" / "eval" / "cau_eval_v1.txt"
STOP_CODAS = {"P", "T", "C", "CH"}


def _lines() -> list[str]:
    return [line.strip() for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_bo_eval_du_so_cau_va_khong_fallback():
    lines = _lines()
    assert len(lines) >= 100
    for line in lines:
        items = parse_text(line)
        assert items, f"câu rỗng: {line!r}"
        for item in items:
            assert str(item.status) == "VALID", f"{line!r} có token fallback: {item.raw!r}"


def test_bo_eval_du_6_thanh_va_coda_tac():
    tones, codas = set(), set()
    for line in _lines():
        for item in parse_text(line):
            tones.add(str(item.abstract.tone))
            codas.add(str(item.abstract.coda))
    assert tones == {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"}
    assert STOP_CODAS <= codas


def test_bo_eval_co_cau_hoi_va_cam_than():
    text = "\n".join(_lines())
    assert "?" in text and "!" in text and "," in text

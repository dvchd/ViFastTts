"""Test cách đọc riêng (custom) — không đụng từ điển chuẩn."""

from __future__ import annotations

from vifasttts.frontend.custom_reading import apply_custom_reading
from vifasttts.frontend.pipeline import parse_text

TABLE = {"vba": "vi be a", "vb": "vi be", "vps": "vi pi ét", "ps": "pi es", "ch": "xê hát"}


def test_khop_cum_dai_truoc():
    assert apply_custom_reading("Dùng VBA để mở", TABLE) == "Dùng vi be a để mở"
    assert apply_custom_reading("dùng VB", TABLE) == "dùng vi be"


def test_khong_sinh_them_am_tiet():
    out = apply_custom_reading("PS và VPS khác nhau", TABLE)
    assert out == "pi es và vi pi ét khác nhau"
    assert out.count("es") == 1


def test_ranh_gioi_tu():
    assert apply_custom_reading("VPSX", TABLE) == "VPSX"
    assert apply_custom_reading("(VPS)", TABLE) == "(vi pi ét)"


def test_parse_khong_lap_tu():
    """Chuỗi thay KHÔNG lặp; normalizer vẫn tách "es" (âm tiết ngoài hệ Việt) thành 2 âm tiết.

    => ghi nhận hạn chế: muốn "es" đọc đúng 1 âm tiết phải sửa normalizer (chưa làm).
    """
    raw = [s.raw for s in parse_text("PS và VPS khác nhau", resolve_oov=True)]
    assert raw[0] == "pi"
    assert raw.count("vi") == 1 and raw.count("pi") == 2

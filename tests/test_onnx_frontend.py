"""Frontend ONNX phải khớp PyTorch production, kể cả resolve OOV (v18.41)."""

from __future__ import annotations

import numpy as np
import pytest

from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.runtime.onnx_engine import OnnxAcousticModel


def _features(items) -> np.ndarray:
    vocabs = ComponentVocabs.default()
    return np.array([[vocabs.encode(item) for item in items]], dtype=np.int64)


def _onnx(text: str, *, dialect: str = "north", resolve_oov: bool | None = None) -> np.ndarray:
    # `features_from_text` không cần session; tạo object rỗng để test frontend thuần.
    engine = object.__new__(OnnxAcousticModel)
    kwargs = {} if resolve_oov is None else {"resolve_oov": resolve_oov}
    return engine.features_from_text(text, dialect=dialect, **kwargs)


@pytest.mark.parametrize("text", [
    "VBA.",              # custom reading
    "OpenAI.",           # English fallback trong normalizer
    "John.",             # OOV resolver đánh vần tên ngắn
    "XYZ.",              # OOV resolver đánh vần chữ hoa
    "Năm 2026, giá 15,5%.",
    "Xin chào các bạn.",
])
def test_onnx_frontend_khop_pytorch_resolve_oov(text: str):
    expected = _features(parse_text(text, resolve_oov=True))
    got = _onnx(text)
    assert np.array_equal(got, expected), (text, got, expected)


def test_onnx_frontend_khop_dialect_south():
    assert np.array_equal(_onnx("1500.", dialect="south"),
                          _features(parse_text("1500.", dialect="south", resolve_oov=True)))


def test_onnx_frontend_mac_dinh_resolve_oov():
    """Mặc định phải resolve; đây là hành vi production, không phải legacy."""
    assert np.array_equal(_onnx("XYZ."), _features(parse_text("XYZ.", resolve_oov=True)))
    assert not np.array_equal(_onnx("XYZ.", resolve_oov=False),
                              _features(parse_text("XYZ.", resolve_oov=True)))


def test_onnx_frontend_khong_am_tiet_bao_loi():
    with pytest.raises(ValueError, match="Không có âm tiết"):
        _onnx("...")

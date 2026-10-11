"""Test tùy chọn precision khi export ONNX (phần nặng nằm ở test_neural_smoke)."""
from __future__ import annotations

import pytest

from vifasttts.export import PRECISIONS, export_onnx


def test_precision_list():
    assert set(PRECISIONS) == {"fp32", "fp16", "int8"}


def test_invalid_precision_rejected_early():
    # Kiểm tra tham số trước khi cần torch/checkpoint.
    with pytest.raises(ValueError):
        export_onnx({}, "missing.pt", "out.onnx", precision="bf16")

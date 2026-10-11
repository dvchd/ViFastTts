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


def test_provider_resolution_cpu_and_invalid():
    pytest.importorskip("onnxruntime")
    from vifasttts.runtime.onnx_engine import resolve_providers

    assert resolve_providers("cpu") == ["CPUExecutionProvider"]
    with pytest.raises(ValueError):
        resolve_providers("tpu")


def test_provider_resolution_matches_availability():
    ort = pytest.importorskip("onnxruntime")
    from vifasttts.runtime.onnx_engine import resolve_providers

    available = set(ort.get_available_providers())
    auto = resolve_providers("auto")
    if "CUDAExecutionProvider" in available:
        assert auto[0] == ("CUDAExecutionProvider", {}) or auto[0] == "CUDAExecutionProvider"
    else:
        assert auto == ["CPUExecutionProvider"]

    if "CUDAExecutionProvider" not in available:
        with pytest.raises(ValueError):
            resolve_providers("cuda")

    if "CoreMLExecutionProvider" in available:
        providers = resolve_providers("coreml")
        assert providers[0][0] == "CoreMLExecutionProvider"
        assert providers[0][1]["MLComputeUnits"] == "ALL"
        assert providers[-1] == "CPUExecutionProvider"
    else:
        with pytest.raises(ValueError):
            resolve_providers("coreml")

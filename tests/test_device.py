"""Test chọn thiết bị auto|cpu|cuda|mps."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vifasttts.device import resolve_device


def test_explicit_cpu():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device(None) in {"cpu", "cuda", "mps"}


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_auto_matches_availability():
    result = resolve_device("auto")
    if torch.cuda.is_available():
        assert result == "cuda"
    elif torch.backends.mps.is_available():
        assert result == "mps"
    else:
        assert result == "cpu"


def test_cuda_override_requires_availability():
    if not torch.cuda.is_available():
        with pytest.raises(ValueError):
            resolve_device("cuda")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS không khả dụng")
def test_mps_override():
    assert resolve_device("mps") == "mps"
    with pytest.raises(ValueError):
        resolve_device("cuda") if not torch.cuda.is_available() else None

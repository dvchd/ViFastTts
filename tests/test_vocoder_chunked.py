"""Test chọn checkpoint sạch để resume (`tools/train_vocoder_chunked._pick_checkpoint`)."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tools.train_vocoder_chunked import _pick_checkpoint, read_step


def _save(path: Path, step: int, **extra) -> None:
    torch.save({"step": step, "vocoder": {}, **extra}, path)


def test_ưu_tiên_last_pt_sạch(tmp_path: Path):
    _save(tmp_path / "last.pt", 120, clean=True, last_loss=1.0)
    _save(tmp_path / "prev.pt", 100, clean=True, last_loss=1.2)
    picked = _pick_checkpoint(tmp_path)
    assert picked is not None and picked.name == "last.pt"
    assert read_step(picked) == 120


def test_bỏ_qua_checkpoint_nghi_nhiễm_nan(tmp_path: Path):
    _save(tmp_path / "last.pt", 130, clean=False, last_loss=1.0)
    _save(tmp_path / "prev.pt", 120, clean=True, last_loss=1.1)
    picked = _pick_checkpoint(tmp_path)
    assert picked is not None and picked.name == "prev.pt"

    # last_loss không hữu hạn cũng bị bỏ qua
    _save(tmp_path / "last.pt", 140, clean=True, last_loss=float("nan"))
    picked = _pick_checkpoint(tmp_path)
    assert picked is not None and picked.name == "prev.pt"


def test_không_có_checkpoint(tmp_path: Path):
    assert _pick_checkpoint(tmp_path) is None


def test_checkpoint_cũ_không_có_cờ_clean_vẫn_dùng_được(tmp_path: Path):
    _save(tmp_path / "last.pt", 50)  # checkpoint cũ: không có clean/last_loss
    picked = _pick_checkpoint(tmp_path)
    assert picked is not None and picked.name == "last.pt"

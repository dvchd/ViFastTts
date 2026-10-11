"""Đọc/kiểm tra config YAML: `load_config` + `validate_config` báo lỗi rõ ràng."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_MEL_KEYS = ("n_fft", "win_length", "hop_length", "n_mels", "f_min", "f_max")
_MODEL_KEYS = (
    "component_dim", "hidden_size", "encoder_layers", "decoder_layers", "attention_heads",
    "ffn_size", "conv_kernel_size", "dropout", "duration_layers", "variance_layers",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Đọc YAML thành dict; lỗi nếu file không phải mapping."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError("Config phải là YAML mapping")
    return data


def validate_config(config: dict[str, Any], kind: str = "acoustic") -> None:
    """Kiểm config trước khi dựng model — báo lỗi rõ ràng thay vì KeyError giữa chừng.

    - `kind="acoustic"`: cần `sample_rate`, `mel` (đủ khóa) và `model` (đủ khóa kiến trúc).
    - `kind="vocoder"`: chỉ cần `sample_rate` + `mel` (vocoder độc lập acoustic model).
    """
    if kind not in ("acoustic", "vocoder"):
        raise ValueError(f"kind không hợp lệ: {kind!r} (dùng acoustic|vocoder)")
    problems: list[str] = []
    if not isinstance(config, dict):
        raise TypeError("Config phải là mapping")
    if "sample_rate" not in config:
        problems.append("thiếu sample_rate")
    mel = config.get("mel")
    if not isinstance(mel, dict):
        problems.append("thiếu mục mel")
    else:
        problems.extend(f"thiếu mel.{key}" for key in _MEL_KEYS if key not in mel)
    if kind == "acoustic":
        model = config.get("model")
        if not isinstance(model, dict):
            problems.append("thiếu mục model")
        else:
            problems.extend(f"thiếu model.{key}" for key in _MODEL_KEYS if key not in model)
    if problems:
        raise ValueError("Config không hợp lệ:\n- " + "\n- ".join(problems))

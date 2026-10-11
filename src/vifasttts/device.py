"""Chọn thiết bị chạy: cuda > mps > cpu, có override tường minh."""

from __future__ import annotations

VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


def resolve_device(setting: str | None = None) -> str:
    """Trả về 'cuda' | 'mps' | 'cpu' theo cài đặt `auto|cpu|cuda|mps`.

    - `auto` (mặc định): CUDA nếu có, ngược lại Apple MPS, cuối cùng CPU.
    - Đặt tường minh nhưng thiết bị không khả dụng → ValueError rõ ràng.
    """
    import torch

    value = (setting or "auto").strip().lower()
    if value not in VALID_DEVICES:
        raise ValueError(f"device không hợp lệ: {setting!r} (dùng auto|cpu|cuda|mps)")
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Yêu cầu device=cuda nhưng CUDA không khả dụng")
        return "cuda"
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Yêu cầu device=mps nhưng MPS không khả dụng")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:  # pragma: no cover - torch quá cũ
        pass
    return "cpu"

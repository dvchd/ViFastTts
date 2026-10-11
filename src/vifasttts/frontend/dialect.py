"""Phương ngữ cho normalizer: Bắc/Trung/Nam chọn cách đọc số và viết tắt."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Dialect(StrEnum):
    """Ba vùng miền normalizer hỗ trợ (parser không phụ thuộc vùng)."""

    NORTH = "north"
    CENTRAL = "central"
    SOUTH = "south"


@dataclass(frozen=True)
class ReadingProfile:
    """Khác biệt đọc số theo vùng: nghìn/ngàn, linh/lẻ, tư/bốn."""

    thousand: str
    zero_tens: str
    four_after_tens: str


PROFILES = {
    Dialect.NORTH: ReadingProfile(thousand="nghìn", zero_tens="linh", four_after_tens="tư"),
    Dialect.CENTRAL: ReadingProfile(thousand="ngàn", zero_tens="lẻ", four_after_tens="tư"),
    Dialect.SOUTH: ReadingProfile(thousand="ngàn", zero_tens="lẻ", four_after_tens="bốn"),
}


def get_profile(value: str | Dialect) -> ReadingProfile:
    """Lấy profile đọc số theo tên vùng; lỗi rõ ràng nếu vùng không hợp lệ."""
    try:
        return PROFILES[Dialect(value)]
    except ValueError as exc:
        raise ValueError("dialect phải là north, central hoặc south") from exc

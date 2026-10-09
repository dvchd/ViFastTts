from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Dialect(StrEnum):
    NORTH = "north"
    CENTRAL = "central"
    SOUTH = "south"


@dataclass(frozen=True)
class ReadingProfile:
    thousand: str
    zero_tens: str
    four_after_tens: str


PROFILES = {
    Dialect.NORTH: ReadingProfile(thousand="nghìn", zero_tens="linh", four_after_tens="tư"),
    Dialect.CENTRAL: ReadingProfile(thousand="ngàn", zero_tens="lẻ", four_after_tens="tư"),
    Dialect.SOUTH: ReadingProfile(thousand="ngàn", zero_tens="lẻ", four_after_tens="bốn"),
}


def get_profile(value: str | Dialect) -> ReadingProfile:
    try:
        return PROFILES[Dialect(value)]
    except ValueError as exc:
        raise ValueError("dialect phải là north, central hoặc south") from exc

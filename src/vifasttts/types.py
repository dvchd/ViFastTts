"""Kiểu dữ liệu lõi của ViFastTts.

Mỗi âm tiết (`ParsedSyllable`) mang đồng thời ba dạng của cùng một âm tiết:

1. `Orthography` — dạng chính tả (cách ghi gốc).
2. `AbstractSyllable` — **dạng âm vị học**: 5 thành phần là các category
   trung lập phương ngữ, dùng làm khoá vocab cho model.
3. Dạng ngữ âm (IPA) — sinh một chiều bởi `frontend/g2p.py`, theo vùng.

Thuật ngữ đầy đủ và quy ước đặt tên: docs/GLOSSARY.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Tone(StrEnum):
    NGANG = "NGANG"
    HUYEN = "HUYEN"
    SAC = "SAC"
    HOI = "HOI"
    NGA = "NGA"
    NANG = "NANG"


class ParseStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    FALLBACK_REQUIRED = "FALLBACK_REQUIRED"


@dataclass(frozen=True)
class Orthography:
    """Dạng chính tả: giữ cách ghi gốc (gi/qu, i/y, dấu cũ/mới...)."""

    onset: str | None
    medial: str | None
    nucleus: str
    coda: str | None
    tone_location: str | None = None


@dataclass(frozen=True)
class AbstractSyllable:
    """Dạng âm vị học: 5 category trung lập phương ngữ.

    Ứng với thuật ngữ ngữ văn: âm đầu / âm đệm / âm chính / âm cuối /
    thanh điệu. Đây là khoá vocab cho model (qua `ComponentVocabs`);
    KHÔNG phải IPA — xem docs/GLOSSARY.md.
    """

    onset: str
    medial: str
    nucleus: str
    coda: str
    tone: Tone

    @property
    def rime(self) -> tuple[str, str, str]:
        """Vần = (âm đệm, âm chính, âm cuối) — phép chiếu sang mô hình
        "onset + vần + thanh" mà không mất thông tin thành phần."""
        return (self.medial, self.nucleus, self.coda)

    @property
    def rime_key(self) -> str:
        """Khoá vần dạng chuỗi, tiện làm dict key/report."""
        return "+".join(self.rime)


@dataclass(frozen=True)
class Boundaries:
    word_end: bool = True
    phrase_end: bool = False
    sentence_end: bool = False


@dataclass
class ParsedSyllable:
    raw: str
    normalized: str
    orthography: Orthography
    abstract: AbstractSyllable  # dạng âm vị học (phonological form)
    boundaries: Boundaries = field(default_factory=Boundaries)
    status: ParseStatus = ParseStatus.VALID
    confidence: float = 1.0
    punctuation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

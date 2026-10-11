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
    """Sáu thanh điệu tiếng Việt; giá trị chuỗi dùng thẳng làm ID trong vocab."""

    NGANG = "NGANG"
    HUYEN = "HUYEN"
    SAC = "SAC"
    HOI = "HOI"
    NGA = "NGA"
    NANG = "NANG"


class ParseStatus(StrEnum):
    """Trạng thái parse của một token (hợp lệ / không hợp lệ / cần fallback)."""

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
    """Ranh giới ngữ điệu dùng sinh ID `boundary`: hết từ / hết mệnh đề / hết câu."""

    word_end: bool = True
    phrase_end: bool = False
    sentence_end: bool = False


@dataclass
class ParsedSyllable:
    """Kết quả parse một âm tiết: chữ gốc + chính tả + dạng âm vị + trạng thái.

    `orthography` giữ cách viết người dùng (i/y, gi/qu, dấu cũ...);
    `abstract` là 5 thành phần trung lập phương ngữ — khoá vocab cho model.
    """

    raw: str
    normalized: str
    orthography: Orthography
    abstract: AbstractSyllable  # dạng âm vị học (phonological form)
    boundaries: Boundaries = field(default_factory=Boundaries)
    status: ParseStatus = ParseStatus.VALID
    confidence: float = 1.0
    punctuation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Chuyển sang dict (asdict) để ghi manifest JSONL / in CLI."""
        return asdict(self)

"""Segment Expander: mở 5 thành phần âm tiết thành chuỗi segment cấp âm vị.

Mục đích (xem docs/TTS_ARCHITECTURE_RESEARCH.md §9.4):
- tách nucleus đôi `IÊ/UÔ/ƯƠ` thành hai đích (i/u/ɯ + ə) để phân bổ
  thời lượng và điều kiện hoá phonation cấp âm vị;
- gán **trọng số thời lượng theo role** (onset/đệm/chính/cuối) làm prior
  cho aligner/duration model;
- cung cấp chuỗi phoneme mịn hơn 5 ID nhưng vẫn tất định từ `AbstractSyllable`.

Đây là "view cấp segment" của cùng mô hình 5 thành phần — không thay thế
đầu vào 7 ID của model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from vifasttts.frontend.dialect import Dialect
from vifasttts.frontend.g2p import (
    CODA_BROAD,
    CODA_NARROW_OVERRIDES,
    MEDIAL_BROAD,
    nucleus_units,
    onset_table,
)
from vifasttts.types import ParsedSyllable, ParseStatus

Style = Literal["broad", "narrow"]

ROLE_ONSET = "onset"
ROLE_MEDIAL = "medial"
ROLE_NUCLEUS = "nucleus"
ROLE_CODA = "coda"
ROLE_TONE = "tone"

# Ưu tiên cấp frame khi thiếu (B2 — khớp `models/slot_ops.PRIORITY`): N > O > C > M.
ROLE_PRIORITY = {ROLE_NUCLEUS: 0, ROLE_ONSET: 1, ROLE_CODA: 2, ROLE_MEDIAL: 3}

# Trọng số thời lượng tương đối theo role — prior khởi điểm, hiệu chỉnh
# bằng forced alignment khi có. Tone không chiếm frame riêng (0.0).
WEIGHT_ONSET = 0.4
WEIGHT_MEDIAL = 0.3
WEIGHT_NUCLEUS = 1.0
WEIGHT_CODA_NASAL = 0.6
WEIGHT_CODA_STOP = 0.35
WEIGHT_CODA_GLIDE = 0.45
WEIGHT_TONE = 0.0

# Nucleus đôi: hai đích (mở đầu + trung tâm ə), tổng ~1.2 > nguyên âm đơn.
DIPHTHONG_PARTS: dict[str, tuple[tuple[str, float], ...]] = {
    "IÊ": (("i", 0.5), ("ə", 0.7)),
    "UÔ": (("u", 0.5), ("ə", 0.7)),
    "ƯƠ": (("ɯ", 0.5), ("ə", 0.7)),
}

_CODA_WEIGHTS = {"M": WEIGHT_CODA_NASAL, "N": WEIGHT_CODA_NASAL, "NG": WEIGHT_CODA_NASAL,
                 "NH": WEIGHT_CODA_NASAL, "P": WEIGHT_CODA_STOP, "T": WEIGHT_CODA_STOP,
                 "C": WEIGHT_CODA_STOP, "CH": WEIGHT_CODA_STOP, "J": WEIGHT_CODA_GLIDE,
                 "W": WEIGHT_CODA_GLIDE}

# ── Mask F0 v2 (Addendum 6 — docs/ACOUSTIC_V2.md §19) ───────────────────────────
# Đo trên train_seg (400 row, cache v2): `vuv` của pyworld bật ~90-94% frame của onset
# vô thanh (closure ngắn trong speech liên tục) với f0 jitter 3-4× nhóm hữu thanh
# (năng lượng KHÔNG tách được: < -15 dB chỉ 0,0%) → slot O chỉ tính F0 loss khi onset
# thuộc nhóm **hữu thanh thật** dưới đây (kèm điều kiện `voiced ratio ≥ 0,5` của §6).
# Coda giữ luôn luật §6: coda tắc tiếng Việt không bung hơi → f0 là đuôi nguyên âm
# (jitter 0,039 chỉ ~1,9× sonorant nên vẫn dùng được).
F0_VOICED_ONSETS = frozenset({"B", "D", "G", "GI", "L", "M", "N", "NG", "NH", "R", "V"})
F0_VOICED_CODAS = frozenset({"M", "N", "NG", "NH", "W", "J"})  # để báo cáo/chẩn đoán


@dataclass(frozen=True)
class Segment:
    """Một đơn vị cấp âm vị để phân bổ thời lượng/điều kiện hoá.

    - `unit`: nhãn trừu tượng (`K`, `W`, `Ô`, `C`, `SAC`...; hai nửa của
      nucleus đôi dùng cùng nhãn nucleus kèm `part`).
    - `role`: onset/medial/nucleus/coda/tone.
    - `weight`: trọng số thời lượng tương đối (0.0 = không chiếm frame).
    - `phoneme`: ký hiệu IPA broad/narrow (tùy `style`), None nếu không có.
    - `part`: chỉ số phần trong nucleus đôi (0/1) hoặc None.
    """

    unit: str
    role: str
    weight: float
    phoneme: str | None = None
    part: int | None = None


def expand_syllable(
    parsed: ParsedSyllable,
    dialect: str | Dialect = Dialect.NORTH,
    *,
    style: Style = "broad",
    with_phonemes: bool = True,
) -> list[Segment]:
    """Mở một âm tiết hợp lệ thành chuỗi segment (rỗng nếu fallback)."""
    if parsed.status != ParseStatus.VALID:
        return []
    a = parsed.abstract
    segments: list[Segment] = []

    if a.onset != "NONE":
        onset_units = onset_table(dialect)[a.onset]
        phoneme = onset_units[0] if (with_phonemes and onset_units) else None
        segments.append(Segment(unit=a.onset, role=ROLE_ONSET, weight=WEIGHT_ONSET, phoneme=phoneme))

    if a.medial != "NONE":
        medial_units = MEDIAL_BROAD[a.medial]
        phoneme = medial_units[0] if (with_phonemes and medial_units) else None
        segments.append(Segment(unit=a.medial, role=ROLE_MEDIAL, weight=WEIGHT_MEDIAL, phoneme=phoneme))

    units = nucleus_units(a.nucleus, dialect, style=style, coda=a.coda)
    parts = DIPHTHONG_PARTS.get(a.nucleus)
    if (
        parts is not None
        and style == "narrow"
        and Dialect(dialect) is Dialect.NORTH
        and a.coda == "W"
        and a.nucleus == "ƯƠ"
    ):
        # "ươu" Hà Nội narrow đọc ngả [iəw] — một đích thay vì tách ɯ+ə.
        parts = None
    if parts is not None:
        for index, (phoneme, weight) in enumerate(parts):
            segments.append(Segment(unit=a.nucleus, role=ROLE_NUCLEUS, weight=weight,
                                    phoneme=phoneme if with_phonemes else None, part=index))
    else:
        first = units[0] if (with_phonemes and units) else None
        weight = 1.2 if a.nucleus == "OO" else WEIGHT_NUCLEUS
        segments.append(Segment(unit=a.nucleus, role=ROLE_NUCLEUS, weight=weight, phoneme=first))

    if a.coda != "NONE":
        coda_units = CODA_BROAD[a.coda]
        if style == "narrow":
            coda_units = CODA_NARROW_OVERRIDES.get(a.coda, coda_units)
        phoneme = coda_units[0] if (with_phonemes and coda_units) else None
        segments.append(Segment(unit=a.coda, role=ROLE_CODA, weight=_CODA_WEIGHTS[a.coda], phoneme=phoneme))

    segments.append(Segment(unit=a.tone.value, role=ROLE_TONE, weight=WEIGHT_TONE, phoneme=None))
    return segments


def expand_text(
    items: Iterable[ParsedSyllable],
    dialect: str | Dialect = Dialect.NORTH,
    *,
    style: Style = "broad",
    with_phonemes: bool = True,
) -> list[list[Segment]]:
    """Mở cả câu; âm tiết fallback trả về danh sách rỗng (caller xử lý)."""
    return [
        expand_syllable(item, dialect, style=style, with_phonemes=with_phonemes)
        for item in items
    ]


def distribute_frames(frames: int, segments: Sequence[Segment]) -> list[int]:
    """Chia `frames` cho segment theo trọng số — **cùng thuật toán B2** với `models/slot_ops`.

    - Segment weight 0 (tone) luôn nhận 0 frame.
    - Thiếu frame: cấp 1 frame cho segment ưu tiên cao nhất theo thứ tự **N, O, C, M** (B2).
    - Nếu đủ frame, mỗi segment dương nhận tối thiểu 1 frame.
    - Tổng kết quả đúng bằng `frames`; lớn nhất còn dư làm tròn xuống, hòa phá theo vị trí.
    """
    result = [0] * len(segments)
    positives = [index for index, segment in enumerate(segments) if segment.weight > 0]
    if frames <= 0 or not positives:
        return result
    count = len(positives)
    if frames < count:
        ordered = sorted(positives, key=lambda i: (ROLE_PRIORITY.get(segments[i].role, 9), i))
        for index in ordered[:frames]:
            result[index] = 1
        return result

    weights = [segments[index].weight for index in positives]
    total_weight = sum(weights)
    remaining = frames - count
    raw = [weight / total_weight * remaining for weight in weights]
    floors = [int(value) for value in raw]
    leftover = remaining - sum(floors)
    order = sorted(range(count), key=lambda k: (-(raw[k] - floors[k]), k))
    for k in order[:leftover]:
        floors[k] += 1
    for position, index in enumerate(positives):
        result[index] = 1 + floors[position]
    return result


def segment_summary(segments: Sequence[Segment]) -> dict[str, float | int | str]:
    """Tóm tắt nhanh cho log/test: số segment, tổng trọng số, chuỗi phoneme."""
    return {
        "count": len(segments),
        "weight": round(sum(segment.weight for segment in segments), 3),
        "phonemes": " ".join(segment.phoneme or "∅" for segment in segments),
    }

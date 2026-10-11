"""Lexicon và profile kiểm tra chặt/chùng cho âm tiết.

Parser cố tình *permissive* (nhận cả từ vay, chính tả lỏng) để không chặn
dữ liệu thật. Module này thêm lớp kiểm tra tùy chọn:

- ``open``   — như parser hiện tại, không cảnh báo gì.
- ``strict`` — cảnh báo theo luật chính tả/ngữ âm bản ngữ (c/k/q, g/gh,
  ng/ngh, môi + /w/, coda nh/ch, tổ hợp đứng một mình phải viết yê...).
- ``corpus`` — thêm ràng buộc shape (onset, âm đệm, âm chính, âm cuối) phải
  có trong ``Lexicon`` (sinh từ từ điển/ngữ liệu của dự án).

Thiết kế: ``check_syllable`` trả về danh sách **mã cảnh báo** (ổn định để
test/CLI dịch ra tiếng người). Không dùng để từ chối dữ liệu ở pipeline —
dùng cho QA ngữ liệu, log và dataset curation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from vifasttts.frontend.syllable import safe_parse_syllable
from vifasttts.types import AbstractSyllable, ParsedSyllable, ParseStatus

Profile = Literal["open", "strict", "corpus"]
PROFILES: tuple[Profile, ...] = ("open", "strict", "corpus")

# Nhóm luật strict (mã cảnh báo -> mô tả người đọc).
ISSUE_DESCRIPTIONS = {
    "not-valid": "token không phải âm tiết hợp lệ (fallback)",
    "spelling-c-before-front": "'c' trước i/e/ê phải viết 'k'",
    "spelling-k-before-back": "'k' trước a/o/u/ơ/ư/ă/â phải viết 'c'",
    "spelling-g-before-front": "'g' trước e/ê phải viết 'gh'",
    "spelling-gh-before-back": "'gh' trước nguyên âm khác phải viết 'g'",
    "spelling-ng-before-front": "'ng' trước i/e/ê phải viết 'ngh'",
    "spelling-ngh-before-back": "'ngh' trước nguyên âm khác phải viết 'ng'",
    "labial-w": "phụ âm môi (b/m/ph/v/p) + âm đệm /w/ là từ vay",
    "coda-nh-ch-after-nonfront": "coda nh/ch chỉ đứng sau a, ê, i",
    "standalone-ie": "iê đứng một mình phải viết yê",
    "open-double-vowel-spelling": "iê/uô/ươ mở phải viết ia/ua/ưa",
    "shape-not-in-lexicon": "shape (onset+vần) không có trong lexicon",
}

_K_GROUP = {"I", "IÊ", "E", "Ê"}
_GH_GROUP = {"I", "IÊ", "E", "Ê"}
_E_GROUP = {"E", "Ê"}
_LABIAL_ONSETS = {"B", "M", "PH", "V", "P"}
_OPEN_ALIASES = {"iê": "ia", "uô": "ua", "ươ": "ưa"}


@dataclass(frozen=True)
class Lexicon:
    """Tập shape âm tiết (onset, medial, nucleus, coda) đã quan sát."""

    shapes: frozenset[tuple[str, str, str, str]]

    @classmethod
    def from_words(cls, words: Iterable[str]) -> Lexicon:
        shapes = set()
        for word in words:
            parsed = safe_parse_syllable(str(word))
            if parsed is not None and parsed.status == ParseStatus.VALID:
                a = parsed.abstract
                shapes.add((a.onset, a.medial, a.nucleus, a.coda))
        return cls(frozenset(shapes))

    @classmethod
    def from_manifest(cls, manifest: str | Path) -> Lexicon:
        shapes = set()
        with Path(manifest).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for syllable in row.get("syllables", []):
                    a = syllable.get("abstract", {})
                    if syllable.get("status", "VALID") == "VALID" and a.get("nucleus"):
                        shapes.add((a["onset"], a["medial"], a["nucleus"], a["coda"]))
        return cls(frozenset(shapes))

    def has(self, abstract: AbstractSyllable) -> bool:
        return (abstract.onset, abstract.medial, abstract.nucleus, abstract.coda) in self.shapes

    @property
    def rimes(self) -> frozenset[tuple[str, str, str]]:
        return frozenset((medial, nucleus, coda) for (_o, medial, nucleus, coda) in self.shapes)

    def save(self, path: str | Path) -> None:
        payload = {"version": 1, "shapes": sorted([list(shape) for shape in self.shapes])}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Lexicon:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(frozenset(tuple(shape) for shape in payload["shapes"]))


def strict_warnings(parsed: ParsedSyllable) -> list[str]:
    """Cảnh báo theo luật chính tả/ngữ âm bản ngữ (không cần lexicon)."""
    issues: list[str] = []
    if parsed.status != ParseStatus.VALID:
        return ["not-valid"]
    a = parsed.abstract
    o = parsed.orthography
    onset_spelling = o.onset or ""
    nucleus_spelling = o.nucleus

    # c/k/q: phân bố bổ khuyết theo chính tả (c trước a/o/u..., k trước i/e/ê).
    if onset_spelling == "c" and a.nucleus in _K_GROUP:
        issues.append("spelling-c-before-front")
    if onset_spelling == "k" and a.nucleus not in _K_GROUP:
        issues.append("spelling-k-before-back")
    # g/gh, ng/ngh: xét chữ viết ngay sau onset — âm đệm (o/u) đứng xen vào
    # làm "ngoe", "nguyễn" trở thành hợp lệ.
    if onset_spelling == "g" and o.medial is None and a.nucleus in _E_GROUP:
        issues.append("spelling-g-before-front")
    if onset_spelling == "gh" and (o.medial is not None or a.nucleus not in _GH_GROUP):
        issues.append("spelling-gh-before-back")
    if onset_spelling == "ng" and o.medial is None and a.nucleus in _GH_GROUP:
        issues.append("spelling-ng-before-front")
    if onset_spelling == "ngh" and (o.medial is not None or a.nucleus not in _GH_GROUP):
        issues.append("spelling-ngh-before-back")

    # Giới hạn ngữ âm bản ngữ: /w/ không đi sau phụ âm môi.
    if a.medial == "W" and a.onset in _LABIAL_ONSETS:
        issues.append("labial-w")

    # Coda nh/ch chỉ sau a, ê, i.
    if a.coda in {"NH", "CH"} and a.nucleus not in {"A", "I", "Ê"}:
        issues.append("coda-nh-ch-after-nonfront")

    # Tổ hợp đứng một mình không có onset.
    if o.onset is None and a.nucleus == "IÊ" and a.coda != "NONE" and nucleus_spelling.startswith("i"):
        issues.append("standalone-ie")
    if o.onset is None and a.coda == "NONE" and nucleus_spelling in _OPEN_ALIASES:
        issues.append("open-double-vowel-spelling")
    return issues


def check_syllable(
    parsed: ParsedSyllable,
    profile: Profile = "strict",
    lexicon: Lexicon | None = None,
) -> list[str]:
    """Trả danh sách mã cảnh báo cho một âm tiết theo profile."""
    if profile == "open":
        return []
    if profile == "strict":
        return strict_warnings(parsed)
    if profile == "corpus":
        if lexicon is None:
            raise ValueError("profile 'corpus' cần lexicon")
        issues = strict_warnings(parsed)
        if parsed.status == ParseStatus.VALID and not lexicon.has(parsed.abstract):
            issues.append("shape-not-in-lexicon")
        return issues
    raise ValueError(f"Profile không hợp lệ: {profile!r} (dùng open/strict/corpus)")


def profile_report(
    words: Iterable[str],
    profile: Profile = "strict",
    lexicon: Lexicon | None = None,
    *,
    max_examples: int = 8,
) -> dict[str, Any]:
    """Thống kê cảnh báo trên một danh sách từ (dùng cho audit/tool)."""
    total = 0
    flagged = 0
    counter: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for word in words:
        parsed = safe_parse_syllable(str(word))
        if parsed is None:
            total += 1
            counter["not-valid"] += 1
            flagged += 1
            examples.setdefault("not-valid", [])
            if len(examples["not-valid"]) < max_examples:
                examples["not-valid"].append(str(word))
            continue
        total += 1
        issues = check_syllable(parsed, profile, lexicon)
        if issues:
            flagged += 1
        for issue in issues:
            counter[issue] += 1
            bucket = examples.setdefault(issue, [])
            if len(bucket) < max_examples:
                bucket.append(str(word))
    return {
        "profile": profile,
        "total": total,
        "flagged": flagged,
        "issues": dict(counter),
        "examples": examples,
    }

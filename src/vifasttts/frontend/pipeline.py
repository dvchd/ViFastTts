"""Đường ống parse văn bản: chuẩn hoá → token → âm tiết kèm dấu câu/ranh giới."""

from __future__ import annotations

import re

from vifasttts.frontend.custom_reading import apply_custom_reading
from vifasttts.frontend.dialect import Dialect
from vifasttts.frontend.normalizer import NormalizerOptions, normalize_text
from vifasttts.frontend.syllable import safe_parse_syllable
from vifasttts.types import (
    AbstractSyllable,
    Boundaries,
    Orthography,
    ParsedSyllable,
    ParseStatus,
    Tone,
)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _fallback_syllable(token: str) -> ParsedSyllable:
    """Tạo entry fallback cho token không phải tiếng Việt hoặc không parse được.

    Giữ raw để debug, đánh dấu FALLBACK_REQUIRED với confidence 0, và dùng ID
    hợp lệ (NONE/NONE/A/NONE/NGANG) để encode không bao giờ crash.
    Caller gắn dấu câu sau đó.
    """
    return ParsedSyllable(
        raw=token,
        normalized=token.lower(),
        orthography=Orthography(onset=None, medial=None, nucleus=token, coda=None),
        abstract=AbstractSyllable(onset="NONE", medial="NONE", nucleus="A", coda="NONE", tone=Tone.NGANG),
        status=ParseStatus.FALLBACK_REQUIRED,
        confidence=0.0,
    )


def parse_text(
    text: str,
    dialect: str | Dialect = Dialect.NORTH,
    *,
    options: NormalizerOptions | None = None,
    resolve_oov: bool = False,
) -> list[ParsedSyllable]:
    """Parse text into structured syllables.

    Cách đọc riêng (`data/codelung/custom_reading.json`) được áp TRƯỚC chuẩn hoá; không có tệp
    thì hành vi y như cũ.

    Normalizer xử lý cách đọc theo vùng miền trước, sau đó mới token hoá và parse.
    Dấu câu đứng trước gắn vào âm tiết kế tiếp, dấu câu đứng sau gắn vào âm tiết
    trước đó kèm ranh giới phrase/câu. Token lạ trở thành FALLBACK_REQUIRED thay
    vì raise, nên build manifest và suy luận không bao giờ crash.

    Với `resolve_oov=True`, token OOV được thử respell bằng
    `frontend/oov.py` (acronym/English/tên chữ cái) trước khi fallback.
    """
    from vifasttts.frontend.oov import resolve_oov_token

    normalized = normalize_text(apply_custom_reading(text), dialect, options=options)
    output: list[ParsedSyllable] = []
    pending_punctuation = ""
    for token in _TOKEN_RE.findall(normalized):
        if re.fullmatch(r"[^\w\s]", token):
            if output:
                output[-1].punctuation = (output[-1].punctuation or "") + token
                if token in ".!?":
                    output[-1].boundaries = Boundaries(word_end=True, phrase_end=True, sentence_end=True)
                elif token in ",;:":
                    output[-1].boundaries = Boundaries(word_end=True, phrase_end=True, sentence_end=False)
            else:
                pending_punctuation += token
            continue
        parsed = safe_parse_syllable(token)
        expanded: list[ParsedSyllable] = []
        if parsed is None and resolve_oov:
            respelled = resolve_oov_token(token)
            if respelled:
                expanded = parse_text(respelled, dialect, options=options)
                for item in expanded:
                    item.raw = token
                    item.confidence = min(item.confidence, 0.5)
        if expanded:
            items = expanded
        else:
            if parsed is None:
                parsed = _fallback_syllable(token)
            items = [parsed]
        if pending_punctuation:
            items[0].punctuation = pending_punctuation + (items[0].punctuation or "")
            pending_punctuation = ""
        output.extend(items)
    return output

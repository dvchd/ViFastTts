from __future__ import annotations

import re
from vifasttts.frontend.dialect import Dialect
from vifasttts.frontend.normalizer import NormalizerOptions, normalize_text
from vifasttts.frontend.syllable import safe_parse_syllable
from vifasttts.types import AbstractSyllable, Boundaries, Orthography, ParsedSyllable, ParseStatus, Tone

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+|[^\w\s]", re.UNICODE)


def _fallback_syllable(token: str) -> ParsedSyllable:
    """Create a fallback entry for non-Vietnamese or invalid tokens.

    Keeps raw text for debugging, marks FALLBACK_REQUIRED with zero
    confidence, and uses valid vocab IDs (NONE/NONE/A/NONE/NGANG) so
    downstream encoding does not crash. Caller attaches punctuation.
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
) -> list[ParsedSyllable]:
    """Parse text into structured syllables.

    Normalizer handles dialect-specific readings, then tokens are parsed.
    Leading punctuation is attached to the next syllable, trailing
    punctuation to the previous one with phrase/sentence boundaries.
    Invalid or foreign tokens become FALLBACK_REQUIRED entries instead
    of raising, so manifest building and inference never crash.
    """
    normalized = normalize_text(text, dialect, options=options)
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
        if parsed is None:
            parsed = _fallback_syllable(token)
        if pending_punctuation:
            parsed.punctuation = pending_punctuation + (parsed.punctuation or "")
            pending_punctuation = ""
        output.append(parsed)
    return output

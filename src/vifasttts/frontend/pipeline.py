from __future__ import annotations

import re
from vifasttts.frontend.dialect import Dialect
from vifasttts.frontend.normalizer import NormalizerOptions, normalize_text
from vifasttts.frontend.syllable import parse_syllable
from vifasttts.types import Boundaries, ParsedSyllable

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+|[^\w\s]", re.UNICODE)


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
        parsed = parse_syllable(token)
        if pending_punctuation:
            parsed.punctuation = pending_punctuation + (parsed.punctuation or "")
            pending_punctuation = ""
        output.append(parsed)
    return output

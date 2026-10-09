from __future__ import annotations

import re
from vifasttts.frontend.dialect import Dialect
from vifasttts.frontend.normalizer import NormalizerOptions, normalize_text
from vifasttts.frontend.syllable import parse_syllable
from vifasttts.types import ParsedSyllable, Boundaries

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+|[^\w\s]", re.UNICODE)


def parse_text(text: str, dialect: str | Dialect = Dialect.NORTH, *, options: NormalizerOptions | None = None) -> list[ParsedSyllable]:
    normalized = normalize_text(text, dialect, options=options)
    output: list[ParsedSyllable] = []
    for token in _TOKEN_RE.findall(normalized):
        if re.fullmatch(r"[^\w\s]", token):
            if output:
                output[-1].punctuation = (output[-1].punctuation or "") + token
                if token in ".!?":
                    output[-1].boundaries = Boundaries(word_end=True, phrase_end=True, sentence_end=True)
                elif token in ",;:":
                    output[-1].boundaries = Boundaries(word_end=True, phrase_end=True, sentence_end=False)
            continue
        output.append(parse_syllable(token))
    return output

from __future__ import annotations

from dataclasses import dataclass
from vifasttts.types import ParsedSyllable


def _map(values: list[str]) -> dict[str, int]:
    return {value: i for i, value in enumerate(values)}


@dataclass(frozen=True)
class ComponentVocabs:
    onset: dict[str, int]
    medial: dict[str, int]
    nucleus: dict[str, int]
    coda: dict[str, int]
    tone: dict[str, int]
    boundary: dict[str, int]
    punctuation: dict[str, int]

    @classmethod
    def default(cls) -> "ComponentVocabs":
        return cls(
            onset=_map(["NONE", "B", "M", "PH", "V", "T", "TH", "Đ", "N", "D", "GI", "R", "X", "S", "CH", "TR", "NH", "L", "K", "KH", "NG", "G", "H", "P"]),
            medial=_map(["NONE", "W"]),
            nucleus=_map(["A", "Ă", "Â", "E", "Ê", "I", "O", "Ô", "Ơ", "U", "Ư", "IÊ", "UÔ", "ƯƠ"]),
            coda=_map(["NONE", "M", "N", "NG", "NH", "P", "T", "C", "CH", "J", "W"]),
            tone=_map(["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]),
            boundary=_map(["NONE", "WORD", "PHRASE", "SENTENCE"]),
            punctuation=_map(["NONE", "COMMA", "PERIOD", "QUESTION", "EXCLAMATION", "OTHER"]),
        )

    def encode(self, item: ParsedSyllable) -> list[int]:
        boundary = "SENTENCE" if item.boundaries.sentence_end else "PHRASE" if item.boundaries.phrase_end else "WORD"
        p = item.punctuation or ""
        punct = "QUESTION" if "?" in p else "EXCLAMATION" if "!" in p else "COMMA" if "," in p else "PERIOD" if "." in p else "OTHER" if p else "NONE"
        a = item.abstract
        return [
            self.onset[a.onset], self.medial[a.medial], self.nucleus[a.nucleus],
            self.coda[a.coda], self.tone[a.tone.value], self.boundary[boundary],
            self.punctuation[punct],
        ]

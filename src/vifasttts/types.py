from dataclasses import dataclass,field,asdict
from enum import StrEnum
class Tone(StrEnum): NGANG="NGANG"; HUYEN="HUYEN"; SAC="SAC"; HOI="HOI"; NGA="NGA"; NANG="NANG"
class ParseStatus(StrEnum): VALID="VALID"; INVALID="INVALID"; FALLBACK_REQUIRED="FALLBACK_REQUIRED"
@dataclass(frozen=True)
class Orthography: onset:str|None; medial:str|None; nucleus:str; coda:str|None; tone_location:str|None=None
@dataclass(frozen=True)
class AbstractSyllable: onset:str; medial:str; nucleus:str; coda:str; tone:Tone
@dataclass(frozen=True)
class Boundaries: word_end:bool=True; phrase_end:bool=False; sentence_end:bool=False
@dataclass
class ParsedSyllable:
 raw:str; normalized:str; orthography:Orthography; abstract:AbstractSyllable; boundaries:Boundaries=field(default_factory=Boundaries); status:ParseStatus=ParseStatus.VALID; confidence:float=1.0; punctuation:str|None=None
 def to_dict(self): return asdict(self)

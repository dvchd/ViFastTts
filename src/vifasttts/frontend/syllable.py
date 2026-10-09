import re
from vifasttts.frontend.rules import *
from vifasttts.frontend.tone import extract_tone
from vifasttts.frontend.unicode import normalize_unicode
from vifasttts.types import *
def _onset(w):
 for o in ONSET_SPELLINGS:
  if w.startswith(o): return o,w[len(o):],o=="qu"
 return "",w,False
def _coda(r):
 for c in CONSONANT_CODAS:
  if r.endswith(c) and len(r)>len(c): return c,r[:-len(c)]
 return "",r
def _semivowel(v):
 if v in SPECIAL_VOWELS: return "",v
 if len(v)>1 and v[-1] in "iyuo" and (v[:-1] in NUCLEUS_ABSTRACT or v[:-1] in SPECIAL_VOWELS): return v[-1],v[:-1]
 return "",v
def _vowel(v,qu):
 if qu:
  if v not in NUCLEUS_ABSTRACT: raise ValueError(v)
  return "u",v,"W",NUCLEUS_ABSTRACT[v]
 if v in SPECIAL_VOWELS:
  m,n=SPECIAL_VOWELS[v];return m,n,"W",NUCLEUS_ABSTRACT[n]
 if v in NUCLEUS_ABSTRACT:return None,v,"NONE",NUCLEUS_ABSTRACT[v]
 raise ValueError(v)
def parse_syllable(raw):
 norm=normalize_unicode(raw.lower()); core=re.sub(r"^[^\wÀ-ỹĐđ]+|[^\wÀ-ỹĐđ]+$","",norm)
 plain,tone,loc=extract_tone(core); o,rem,qu=_onset(plain); c,v=_coda(rem)
 if not c:c,v=_semivowel(v)
 ms,ns,ma,na=_vowel(v,qu)
 p=ParsedSyllable(raw,norm,Orthography(o or None,ms,ns,c or None,loc),AbstractSyllable(ONSET_ABSTRACT[o],ma,na,CODA_ABSTRACT[c],tone))
 if p.abstract.coda in STOP_CODAS and tone not in {Tone.SAC,Tone.NANG}:raise ValueError("stop coda")
 return p
def safe_parse_syllable(raw):
 try:return parse_syllable(raw)
 except ValueError:return None

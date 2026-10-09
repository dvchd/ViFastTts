import unicodedata
from vifasttts.types import Tone
MARK={"\u0301":Tone.SAC,"\u0300":Tone.HUYEN,"\u0309":Tone.HOI,"\u0303":Tone.NGA,"\u0323":Tone.NANG}
def extract_tone(syllable):
 out=[]; tone=Tone.NGANG; loc=None; last=None; seen=set()
 for ch in unicodedata.normalize("NFD",syllable):
  if not unicodedata.combining(ch): last=ch;out.append(ch)
  elif ch in MARK: tone=MARK[ch];loc=last;seen.add(tone)
  else: out.append(ch)
 if len(seen)>1: raise ValueError("multiple tones")
 return unicodedata.normalize("NFC",''.join(out)),tone,loc

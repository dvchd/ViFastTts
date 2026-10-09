from dataclasses import dataclass
def mp(x):return {v:i for i,v in enumerate(x)}
@dataclass(frozen=True)
class ComponentVocabs:
 onset:dict;medial:dict;nucleus:dict;coda:dict;tone:dict;boundary:dict;punctuation:dict
 @classmethod
 def default(cls):return cls(mp(["NONE","B","M","PH","V","T","TH","Đ","N","D","GI","R","X","S","CH","TR","NH","L","K","KH","NG","G","H","P"]),mp(["NONE","W"]),mp(["A","Ă","Â","E","Ê","I","O","Ô","Ơ","U","Ư","IÊ","UÔ","ƯƠ"]),mp(["NONE","M","N","NG","NH","P","T","C","CH","J","W"]),mp(["NGANG","HUYEN","SAC","HOI","NGA","NANG"]),mp(["NONE","WORD","PHRASE","SENTENCE"]),mp(["NONE","COMMA","PERIOD","QUESTION","EXCLAMATION","OTHER"]))
 def encode(self,x):
  a=x.abstract;return [self.onset[a.onset],self.medial[a.medial],self.nucleus[a.nucleus],self.coda[a.coda],self.tone[a.tone.value],self.boundary["WORD"],self.punctuation["NONE"]]

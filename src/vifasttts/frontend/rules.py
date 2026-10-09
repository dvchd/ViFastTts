ONSET_SPELLINGS=("ngh","ng","gh","gi","kh","nh","ph","th","tr","ch","qu","b","c","d","đ","g","h","k","l","m","n","p","q","r","s","t","v","x")
ONSET_ABSTRACT={"":"NONE","b":"B","m":"M","ph":"PH","v":"V","t":"T","th":"TH","đ":"Đ","n":"N","d":"D","gi":"GI","r":"R","x":"X","s":"S","ch":"CH","tr":"TR","nh":"NH","l":"L","c":"K","k":"K","q":"K","qu":"K","kh":"KH","ng":"NG","ngh":"NG","g":"G","gh":"G","h":"H","p":"P"}
CONSONANT_CODAS=("nh","ng","ch","m","n","p","t","c")
CODA_ABSTRACT={"":"NONE","m":"M","n":"N","ng":"NG","nh":"NH","p":"P","t":"T","c":"C","ch":"CH","i":"J","y":"J","u":"W","o":"W"}
NUCLEUS_ABSTRACT={"a":"A","ă":"Ă","â":"Â","e":"E","ê":"Ê","i":"I","y":"I","o":"O","ô":"Ô","ơ":"Ơ","u":"U","ư":"Ư","iê":"IÊ","yê":"IÊ","ia":"IÊ","ya":"IÊ","uô":"UÔ","ua":"UÔ","ươ":"ƯƠ","ưa":"ƯƠ"}
SPECIAL_VOWELS={"oa":("o","a"),"oă":("o","ă"),"oe":("o","e"),"uâ":("u","â"),"uê":("u","ê"),"uy":("u","y"),"uyê":("u","yê"),"uơ":("u","ơ")}
STOP_CODAS={"P","T","C","CH"}


# Productive legal combinations used by validators, coverage reports and future
# Rust parity tests. Orthographic validity should still be checked against a
# generated syllable lexicon before production training.
MEDIAL_NUCLEUS_COMPATIBILITY = {
    "NONE": set(NUCLEUS_ABSTRACT.values()),
    "W": {"A", "Ă", "Â", "E", "Ê", "I", "Ơ", "IÊ"},
}

CODA_TONE_COMPATIBILITY = {
    "NONE": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "M": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "N": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "NG": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "NH": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "J": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "W": {"NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"},
    "P": {"SAC", "NANG"}, "T": {"SAC", "NANG"},
    "C": {"SAC", "NANG"}, "CH": {"SAC", "NANG"},
}

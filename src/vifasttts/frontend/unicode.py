import re,unicodedata
def normalize_unicode(text):
 text=unicodedata.normalize("NFC",text).replace("…","...").replace("“",'"').replace("”",'"').replace("’","'")
 return re.sub(r"\s+"," ",text).strip()

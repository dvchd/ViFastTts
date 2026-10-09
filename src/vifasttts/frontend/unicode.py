import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("…", "...").replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return _SPACE_RE.sub(" ", text).strip()

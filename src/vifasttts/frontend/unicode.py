"""Chuẩn hoá Unicode cho văn bản tiếng Việt (NFC + dấu nháy + khoảng trắng)."""

import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """NFC toàn chuỗi, đưa dấu nháy/ellipsis về ASCII, gộp khoảng trắng."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("…", "...").replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return _SPACE_RE.sub(" ", text).strip()

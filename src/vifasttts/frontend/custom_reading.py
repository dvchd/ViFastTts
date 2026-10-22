"""Cách đọc RIÊNG (custom) cho từ viết tắt — không đụng từ điển chuẩn.

Vì sao: cùng một người đọc có thể đọc chữ cái không nhất quán (S lúc "ét" lúc "es",
R đọc theo từ "rust"...). Ta không sửa `LETTER_READINGS` toàn cục; chỉ thay ở mức
CHUỖI TRƯỚC KHI chuẩn hoá, theo một tệp JSON riêng (mặc định
`data/codelung/custom_reading.json`, không commit).

Tệp: {"vb": "vi be", "vps": "vi pi ét", "ps": "pi es", "vs": "vi es",
       "ini": "i ni", "ch": "xê hát"}
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path("data/codelung/custom_reading.json")


_VALIDATING = False


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict[str, str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    table = {str(k).lower(): str(v) for k, v in data.items() if str(k).strip()}
    _validate(table)
    return table


def _validate(table: dict[str, str]) -> None:
    """Mọi giá trị phải parse thành ĐÚNG số âm tiết Việt bằng số từ (không lặng lẽ tách)."""
    global _VALIDATING
    if _VALIDATING:
        return
    _VALIDATING = True
    try:
        from vifasttts.frontend.pipeline import parse_text

        bad = []
        for key, value in table.items():
            words = value.split()
            parsed = parse_text(value, resolve_oov=True)
            if len(parsed) != len(words):
                bad.append(f"{key}={value!r}: {len(words)} từ → {len(parsed)} âm tiết")
        if bad:
            raise ValueError("custom_reading không hợp lệ: " + "; ".join(bad))
    finally:
        _VALIDATING = False


def custom_map(path: str | Path | None = None) -> dict[str, str]:
    """Nạp bảng đọc riêng (rỗng nếu không có tệp). Không raise khi tệp hỏng."""
    target = Path(path) if path is not None else DEFAULT_PATH
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return {}
    return _load(str(target), mtime)


def apply_custom_reading(text: str, table: dict[str, str] | None = None) -> str:
    """Thay token viết tắt bằng cách đọc riêng — tách token MỘT LẦN, không quét lại phần đã thay.

    Khớp theo token nguyên vẹn (ranh giới từ), ưu tiên khoá DÀI trước; phần đã thay không bị
    so khớp lại nên "PS và VPS" không sinh thêm âm tiết.
    """
    table = table if table is not None else custom_map()
    if not table:
        return text
    keys = {k.lower(): v for k, v in table.items()}
    pieces = re.split(r"([^\wÀ-ỹ]+)", text)  # giữ nguyên dấu phân cách
    out = []
    for piece in pieces:
        replacement = keys.get(piece.lower())
        out.append(replacement if replacement is not None else piece)
    return "".join(out)

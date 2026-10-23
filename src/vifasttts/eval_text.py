"""Dựng text cho đánh giá từ syllable — bảo toàn dấu câu (v18.37).

Vì sao có module này: các script eval cũ dựng `" ".join(raw)` nên **mất `punctuation`**
(`s["punctuation"]` là trường riêng, không nằm trong `raw`). Hệ quả: `parse_text` không
thấy dấu câu, `iter_synthesize_sentence` nhận `gaps = 0`, và mọi kết luận pause/duration
của free-run bị nhiễu. Hàm ở đây là nguồn duy nhất cho text eval; đường cũ vẫn giữ dưới
tên `legacy_raw_join` để làm đối chứng có nhãn.

Quy ước:
- token lấy `raw`, fallback `normalized`; thiếu cả hai → `ValueError` (không bỏ token im lặng);
- punctuation gắn trực tiếp vào token trước, không thêm khoảng trắng: `"bạn" + "." → "bạn."`;
- nếu token đã kết thúc bằng punctuation đó thì không lặp: `"bạn." + "." → "bạn."`;
- punctuation nhiều ký tự (`".?"`, `"++"`) được giữ, chỉ bỏ phần đầu đã trùng hậu tố.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _syllables(source: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Nhận row manifest hoặc list syllable; trả list để dùng chung."""
    if isinstance(source, Mapping):
        value = source.get("syllables")
        if value is None:
            raise ValueError("row không có trường 'syllables'")
        return list(value)
    return list(source)


def _token(syllable: Mapping[str, Any], index: int) -> str:
    """Lấy token thật; không im lặng bỏ token rỗng."""
    for key in ("raw", "normalized"):
        value = syllable.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f"syllable[{index}] thiếu cả 'raw' và 'normalized'")


def _attach_punctuation(token: str, punctuation: str | None) -> str:
    """Gắn punctuation vào cuối token, tránh lặp hậu tố đã có."""
    if not punctuation:
        return token
    suffix = str(punctuation).strip()
    if not suffix:
        return token
    if token.endswith(suffix):
        return token
    # punctuation nhiều ký tự: bỏ các ký tự đầu đã trùng hậu tố token.
    rest = suffix
    while rest and token.endswith(rest[0]):
        rest = rest[1:]
    return token + rest


def build_eval_text(
    source: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    preserve_punctuation: bool = True,
) -> str:
    """Dựng text eval từ row/list syllable.

    `preserve_punctuation=True` (mặc định): text dùng để chạy frontend, có dấu câu.
    `False`: tương đương raw-join cũ, **chỉ dùng làm đối chứng legacy**.
    """
    items = _syllables(source)
    if not preserve_punctuation:
        return legacy_raw_join(items)
    tokens: list[str] = []
    pending = ""
    for index, syllable in enumerate(items):
        token = _token(syllable, index)
        if pending:
            token = pending + token
            pending = ""
        tokens.append(_attach_punctuation(token, syllable.get("punctuation")))
    if pending:  # punctuation đứng trước nhưng không có token kế tiếp
        tokens.append(pending)
    return " ".join(tokens)


def legacy_raw_join(source: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> str:
    """Đường cũ `" ".join(raw)` — giữ nguyên để đối chứng, không dùng cho eval mới."""
    items = _syllables(source)
    return " ".join(_token(syllable, index) for index, syllable in enumerate(items))


def text_record(source: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Metadata text để ghi vào `items.json`: text dùng thật + bản legacy có nhãn."""
    return {
        "text": build_eval_text(source, preserve_punctuation=True),
        "text_legacy": legacy_raw_join(source),
        "text_mode": "punctuation",
    }

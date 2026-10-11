"""Fallback tiếng Anh viết tắt/sản phẩm: bảng tra chính xác + không phân biệt hoa thường."""

from __future__ import annotations

# Acronym phải khớp chính xác, có phân biệt hoa thường. Vì vậy AI khác ai/Ai.
ACRONYM_READINGS = {
    "AI": "ây ai",
    "API": "ây pi ai",
    "CPU": "xi pi diu",
    "GPU": "giê pi diu",
    "HTML": "hát tê em lờ",
    "HTTP": "hát tê tê pi",
    "HTTPS": "hát tê tê pi ét",
    "JSON": "giây sần",
    "NASA": "na xa",
    "RAM": "ram",
    "ROM": "rom",
    "SSD": "ét ét đi",
    "URL": "diu a lờ",
    "USB": "diu ét bi",
    "XML": "ích em lờ",
    "GPT": "giê pê tê",
    "OTP": "ô tê pê",
    "PDF": "pê đê ép",
    "VPN": "vê pê en",
}

# Từ tiếng Anh thông thường được phép khớp không phân biệt hoa thường,
# nhưng chỉ sau khi token không phải âm tiết tiếng Việt và không phải số La Mã.
CASE_INSENSITIVE_ENGLISH = {
    "app": "áp",
    "audio": "ô đi ô",
    "bluetooth": "bờ lu tút",
    "browser": "bờ rao dơ",
    "chat": "chát",
    "check": "chéc",
    "cloud": "cờ lao",
    "data": "đa ta",
    "email": "i meo",
    "facebook": "phây búc",
    "fan": "phan",
    "file": "phai",
    "game": "ghêm",
    "google": "gu gồ",
    "grab": "gờ ráp",
    "internet": "in tơ nét",
    "laptop": "láp tốp",
    "livestream": "lai sờ trim",
    "meeting": "mi tinh",
    "ok": "ô kê",
    "oke": "ô kê",
    "online": "on lai",
    "podcast": "pót cát",
    "sale": "sêu",
    "server": "xơ vơ",
    "shopee": "sóp pi",
    "software": "xóp que",
    "tiktok": "tíc tốc",
    "video": "vi đi ô",
    "virus": "vi rút",
    "voucher": "vâu chờ",
    "web": "quếp",
    "website": "quếp sai",
    "wifi": "quai phai",
    "windows": "quin đô",
    "youtube": "diu túp",
    "zalo": "da lô",
    "zoom": "dum",
}

# Tên sản phẩm ưu tiên cách viết chính xác. Có thể thêm bằng user override.
EXACT_PRODUCT_NAMES = {
    "Mac": "mác",
    "MacBook": "mác búc",
    "Microsoft": "mai cờ rô xóp",
    "YouTube": "diu túp",
    "Facebook": "phây búc",
    "Google": "gu gồ",
}


def lookup_exact_acronym(token: str) -> str | None:
    """Tra viết tắt đúng hoa thường (AI → "ây ai"); None nếu không có."""
    return ACRONYM_READINGS.get(token)


def lookup_product_name(token: str) -> str | None:
    """Tra tên sản phẩm viết đúng như bảng; None nếu không có."""
    return EXACT_PRODUCT_NAMES.get(token)


def lookup_case_insensitive_english(token: str) -> str | None:
    """Tra từ tiếng Anh không phân biệt hoa thường; None nếu không có."""
    return CASE_INSENSITIVE_ENGLISH.get(token.lower())

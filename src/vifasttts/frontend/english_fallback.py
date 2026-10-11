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
    "SDK": "ét đê ca",
    "UI": "iu ai",
    "VPN": "vê pê en",
}

# Từ tiếng Anh thông thường được phép khớp không phân biệt hoa thường,
# nhưng chỉ sau khi token không phải âm tiết tiếng Việt và không phải số La Mã.
CASE_INSENSITIVE_ENGLISH = {
    "app": "áp",
    "audio": "ô đi ô",
    "binary": "bai na ri",
    "bluetooth": "bờ lu tút",
    "browser": "bờ rao dơ",
    "build": "biu",
    "chat": "chát",
    "check": "chéc",
    "clone": "cờ lon",
    "cloud": "cờ lao",
    "code": "cốt",
    "comment": "cô men",
    "community": "cô miu ni ti",
    "copy": "cô pi",
    "data": "đa ta",
    "deps": "đép",
    "desktop": "đép tốp",
    "download": "đao lô",
    "email": "i meo",
    "facebook": "phây búc",
    "fan": "phan",
    "file": "phai",
    "folder": "phôn đờ",
    "game": "ghêm",
    "github": "gít húp",
    "google": "gu gồ",
    "grab": "gờ ráp",
    "internet": "in tơ nét",
    "laptop": "láp tốp",
    "link": "linh",
    "livestream": "lai sờ trim",
    "meeting": "mi tinh",
    "net": "nét",
    "next": "nếch",
    "ninja": "nin da",
    "ok": "ô kê",
    "oke": "ô kê",
    "online": "on lai",
    "paste": "pết",
    "podcast": "pót cát",
    "program": "pờ rô gờ ram",
    "sale": "sêu",
    "server": "xơ vơ",
    "shopee": "sóp pi",
    "software": "xóp que",
    "studio": "xờ tu đi ô",
    "system": "xít tờm",
    "tiktok": "tíc tốc",
    "upload": "áp lô",
    "video": "vi đi ô",
    "view": "viu",
    "virus": "vi rút",
    "visual": "vi du an",
    "voucher": "vâu chờ",
    "web": "quếp",
    "website": "quếp sai",
    "wifi": "quai phai",
    "win": "uyn",
    "windows": "quin đô",
    "youtube": "diu túp",
    "zalo": "da lô",
    "zip": "díp",
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

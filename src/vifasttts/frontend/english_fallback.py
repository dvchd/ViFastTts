from __future__ import annotations

# Acronym phải khớp chính xác, có phân biệt hoa thường. Vì vậy AI khác ai/Ai.
ACRONYM_READINGS = {
    "AI": "ây ai",
    "API": "ây pi ai",
    "CPU": "xi pi diu",
    "GPU": "gi pi diu",
    "HTML": "hát tê em eo",
    "HTTP": "hát tê tê pi",
    "HTTPS": "hát tê tê pi ét",
    "JSON": "giây sần",
    "NASA": "na xa",
    "RAM": "ram",
    "ROM": "rom",
    "SSD": "ét ét đi",
    "URL": "diu a eo",
    "USB": "diu ét bi",
    "XML": "ích em eo",
}

# Từ tiếng Anh thông thường được phép khớp không phân biệt hoa thường,
# nhưng chỉ sau khi token không phải âm tiết tiếng Việt và không phải số La Mã.
CASE_INSENSITIVE_ENGLISH = {
    "app": "áp",
    "audio": "ô đi ô",
    "bluetooth": "blu tút",
    "browser": "brao dơ",
    "chat": "chát",
    "cloud": "cờ lao",
    "email": "i meo",
    "facebook": "phây búc",
    "file": "phai",
    "google": "gu gồ",
    "internet": "in tơ nét",
    "online": "on lai",
    "podcast": "pót cát",
    "server": "xơ vơ",
    "software": "xóp que",
    "video": "vi đi ô",
    "web": "quếp",
    "website": "quếp sai",
    "wifi": "quai phai",
    "windows": "quin đô",
    "youtube": "diu túp",
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
    return ACRONYM_READINGS.get(token)


def lookup_product_name(token: str) -> str | None:
    return EXACT_PRODUCT_NAMES.get(token)


def lookup_case_insensitive_english(token: str) -> str | None:
    return CASE_INSENSITIVE_ENGLISH.get(token.lower())

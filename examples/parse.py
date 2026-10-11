"""Ví dụ tối thiểu: parse một câu và in cấu trúc từng âm tiết."""

from pprint import pprint

from vifasttts.frontend.pipeline import parse_text

for item in parse_text("Hôm nay trời đẹp."):
    pprint(item.to_dict())

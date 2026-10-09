from pprint import pprint
from vifasttts.frontend.pipeline import parse_text

for item in parse_text("Hôm nay trời đẹp."):
    pprint(item.to_dict())

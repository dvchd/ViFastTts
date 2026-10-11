"""Benchmark nhanh các biểu diễn frontend trên một file văn bản.

Chạy:
    python tools/frontend_benchmark.py --text-file data.txt
    python tools/frontend_benchmark.py --text "Chào bạn hôm nay"
    python tools/frontend_benchmark.py --text-file data.txt --dialect south
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vifasttts.frontend.benchmark import format_stats, representation_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text-file", help="File văn bản, mỗi dòng một câu")
    source.add_argument("--text", help="Một câu truyền trực tiếp")
    parser.add_argument("--dialect", default="north", choices=["north", "central", "south"])
    args = parser.parse_args()

    if args.text:
        texts = [args.text]
    else:
        texts = [line for line in Path(args.text_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = representation_stats(texts, dialect=args.dialect)
    print(format_stats(stats))


if __name__ == "__main__":
    main()

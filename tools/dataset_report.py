"""Báo cáo nhanh dataset YouTube: per-video, tổng giờ, fallback, clip cần xem lại.

Chạy:
    python tools/dataset_report.py --dataset data/codelung
"""

from __future__ import annotations

import argparse

from vifasttts.data.report import dataset_summary, format_report


def main() -> None:
    """Đọc tham số rồi in báo cáo dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/youtube", help="Thư mục dataset")
    parser.add_argument("--flag-fallback", type=float, default=0.2,
                        help="Ngưỡng tỉ lệ fallback để gắn cờ clip cần xem lại")
    args = parser.parse_args()
    print(format_report(dataset_summary(args.dataset, flag_fallback=args.flag_fallback)))


if __name__ == "__main__":
    main()

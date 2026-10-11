"""Chia manifest thành train/val **theo video nguồn** (tránh rò rỉ dữ liệu).

Chạy:
    python tools/split_manifest.py --manifest data/codelung/manifest.jsonl \
        --out-dir data/codelung --val-videos 4
    python tools/split_manifest.py --manifest data/codelung/manifest.jsonl \
        --out-dir data/codelung --val-ratio 0.1
"""

from __future__ import annotations

import argparse

from vifasttts.data.split import write_split


def main() -> None:
    """Đọc tham số, ghi train.jsonl/val.jsonl và in thống kê."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Manifest JSONL đầy đủ")
    parser.add_argument("--out-dir", required=True, help="Thư mục ghi train.jsonl/val.jsonl")
    parser.add_argument("--val-videos", type=int, default=None, help="Số video cho validation")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Tỉ lệ video cho validation (nếu không đặt --val-videos)")
    parser.add_argument("--seed", type=int, default=13, help="Seed chọn video val (tất định)")
    parser.add_argument("--clips", default=None,
                        help="clips.jsonl để tính số giờ (mặc định cạnh manifest)")
    args = parser.parse_args()

    stats = write_split(
        args.manifest, args.out_dir,
        val_videos=args.val_videos, val_ratio=args.val_ratio, seed=args.seed, clips=args.clips,
    )
    print(f"Train: {stats['train']} mẫu ({stats['train_hours']:.2f} giờ)")
    print(f"Val:   {stats['val']} mẫu ({stats['val_hours']:.2f} giờ)")
    print(f"Video val: {', '.join(stats['val_videos']) or '(không có — dữ liệu < 2 video)'}")
    print(f"Đã ghi: {args.out_dir}/train.jsonl và {args.out_dir}/val.jsonl")


if __name__ == "__main__":
    main()

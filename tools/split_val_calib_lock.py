"""V11.A6: chia val fixtail (hàng `start`..`end`) thành val_calib / val_lock **theo clip cha**.

Quy tắc: mọi đoạn của cùng một clip cha vào CÙNG một tập (tránh rò rỉ giữa hiệu chuẩn và
khoá); loại clip cha đã xuất hiện trong val300 (300 hàng đầu — tập eval đang dùng) để hai tập
không giao val300. Thứ tự manifest (tất định); `val_calib` nhận dần theo nhóm cho tới ~nửa số hàng.

Dùng: `.venv/bin/python tools/split_val_calib_lock.py`
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def split(manifest: Path, start: int = 300, end: int = 1773) -> tuple[list[dict], list[dict]]:
    """Trả (calib, lock) — tất định theo thứ tự manifest; giữ nguyên hàng gốc."""
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    val300_parents = {row["parent"] for row in rows[:start]}
    pool = [row for row in rows[start:end] if row["parent"] not in val300_parents]
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in pool:
        if row["parent"] not in groups:
            order.append(row["parent"])
        groups.setdefault(row["parent"], []).append(row)
    target = len(pool) / 2
    calib: list[dict] = []
    lock: list[dict] = []
    for parent in order:
        (calib if len(calib) < target else lock).extend(groups[parent])
    return calib, lock


@app.command()
def main(manifest: str = "data/codelung/val_clean_v2_dur_seg_slot4_fixtail.jsonl",
         out_dir: str = "data/codelung", start: int = 300, end: int = 1773) -> None:
    calib, lock = split(Path(manifest), start, end)
    for name, part in (("calib", calib), ("lock", lock)):
        path = Path(out_dir) / f"val_clean_v2_dur_seg_slot4_fixtail_{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in part),
                        encoding="utf-8")
        sha = hashlib.sha1(path.read_bytes()).hexdigest()
        print(f"{path}: {len(part)} hàng · {len({row['parent'] for row in part})} clip cha · sha1 {sha}")


if __name__ == "__main__":
    app()

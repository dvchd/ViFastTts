"""Parse lại `abstract`/`orthography` của từng âm tiết trong manifest (từ `text`).

Dùng khi parser được sửa ở tầng âm vị (vd "quốc": K+W+Ô+C → K+UÔ+C — góp ý của
ThS Đinh Thị Hằng, Viện Ngôn ngữ học, 16/10). Nguyên tắc **phẫu thuật**:

- chỉ hai trường `abstract` + `orthography` được cập nhật;
- mọi trường khác (`boundaries`, `punctuation`, `confidence`...) **giữ nguyên
  từng byte** — nếu khác với parse hiện tại (file thời kỳ cũ) thì chỉ đếm và
  cảnh báo là `stale`, KHÔNG sửa;
- bất biến: cùng `text` phải ra cùng số âm tiết và đúng `raw`/`normalized` cũ,
  sai thì thoát mã lỗi (vd `text` đã sửa nhưng `syllables` chưa sync).

Lưu ý: đổi byte của manifest = vỡ sha256 ghim cho các mốc bench cũ, nên chỉ
chạy `--write` khi chuẩn bị manifest cho lần train mới.

Chạy:
    python tools/reparse_syllables.py data/codelung/train_clean_v2_dur_seg_slot4.jsonl
    python tools/reparse_syllables.py --write data/codelung/train_clean_v2_dur_seg_slot4.jsonl

`--write` tạo sao lưu `<tên>.pre-reparse.bak` (một lần) trước khi ghi đè.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for extra in (_ROOT, _ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from vifasttts.frontend.pipeline import parse_text

_UPDATED_KEYS = ("abstract", "orthography")


def reparse_manifest(path: str | Path, *, write: bool = False) -> dict[str, object]:
    """Cập nhật `abstract`/`orthography` từ `text`; trả thống kê thay đổi."""
    manifest = Path(path)
    rows: list[dict[str, object]] = []
    changed = 0
    stale = 0
    total = 0
    examples: list[str] = []
    stale_examples: list[str] = []
    with manifest.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not text:
                raise ValueError(f"{manifest}:{lineno}: thiếu 'text' để parse lại")
            new = [item.to_dict() for item in parse_text(str(text))]
            old = row.get("syllables") or []
            if len(new) != len(old):
                raise ValueError(f"{manifest}:{lineno}: số âm tiết đổi {len(old)} → {len(new)}")
            for before, after in zip(old, new):
                total += 1
                if (before.get("raw") or "") != (after.get("raw") or "") or (
                    before.get("normalized") or ""
                ) != (after.get("normalized") or ""):
                    raise ValueError(
                        f"{manifest}:{lineno}: token đổi — "
                        f"{before.get('raw')!r} → {after.get('raw')!r} "
                        f"(text và syllables chưa đồng bộ?)"
                    )
                other_keys = sorted(
                    key
                    for key in set(before) | set(after)
                    if key not in _UPDATED_KEYS and before.get(key) != after.get(key)
                )
                if other_keys:
                    stale += 1
                    if len(stale_examples) < 5:
                        stale_examples.append(f"{after.get('normalized')}: khác ở {other_keys}")
                if any(before.get(key) != after.get(key) for key in _UPDATED_KEYS):
                    changed += 1
                    if len(examples) < 10:
                        examples.append(
                            f"{after.get('normalized')}: "
                            f"{before.get('abstract')} → {after.get('abstract')}"
                        )
                    for key in _UPDATED_KEYS:
                        before[key] = after[key]
            row["syllables"] = old
            rows.append(row)
    stats: dict[str, object] = {
        "manifest": str(manifest),
        "rows": len(rows),
        "syllables": total,
        "changed": changed,
        "stale": stale,
        "examples": examples,
        "stale_examples": stale_examples,
    }
    if write and changed:
        backup = manifest.with_name(manifest.name + ".pre-reparse.bak")
        if not backup.exists():
            shutil.copy2(manifest, backup)
        with manifest.open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        stats["backup"] = str(backup)
    return stats


def main() -> None:
    """Đọc tham số, parse lại các manifest, in và (tuỳ chọn) ghi báo cáo JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", help="Manifest JSONL (có 'text' + 'syllables')")
    parser.add_argument("--write", action="store_true", help="Ghi đè tại chỗ (kèm .pre-reparse.bak)")
    parser.add_argument("--report", help="Ghi báo cáo JSON (vd artifacts/reparse_syllables.json)")
    args = parser.parse_args()
    reports = []
    exit_code = 0
    for path in args.manifests:
        try:
            stats = reparse_manifest(path, write=args.write)
        except ValueError as error:
            print(f"LỖI: {error}")
            exit_code = 1
            continue
        reports.append(stats)
        print(
            f"{stats['manifest']}: {stats['rows']} row | {stats['syllables']} âm tiết | "
            f"đổi {stats['changed']} | stale (giữ nguyên) {stats['stale']}"
        )
        for example in stats["examples"]:
            print("   ", example)
        for example in stats["stale_examples"]:
            print("    STALE:", example)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

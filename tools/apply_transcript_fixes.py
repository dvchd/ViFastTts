"""tools/apply_transcript_fixes.py — áp danh sách sửa lỗi transcript (đã duyệt tay) lên manifest.

Nguồn lỗi + cách tìm: `tools/audit_bigrams.py` (bigram hiếm có biến thể phổ biến) →
duyệt tay → `resources/transcript_fixes.json`. Vì sao cần: model đọc đúng cái sai của phụ đề
(vd "thư **dục**"), và **mọi checkpoint sai giống nhau** — không sửa được bằng train.

Chỉ thay **từ nguyên vẹn** (không thay giữa từ), so khớp không phân biệt hoa/thường, giữ nguyên
khoảng trắng/dấu câu. KHÔNG tự áp lên val/holdout (giữ mốc so sánh + tránh "tự thưởng" CER).

Chạy:
    .venv/bin/python tools/apply_transcript_fixes.py \
        --manifest data/codelung/train_clean_v2_dur_seg_slot4_fixtail.jsonl \
        --out data/codelung/train_clean_v2_dur_seg_slot4_fixtail_typo1.jsonl \
        --report artifacts/dataset_v2/apply_fixes_typo1.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _apply(text: str, pairs: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, list[str]]:
    """Thay cặp từ (không phân biệt hoa/thường) — trả (text mới, danh sách cặp đã dùng)."""
    matches = list(_WORD_RE.finditer(text))
    words = [m.group(0).lower() for m in matches]
    used: list[str] = []
    pieces: list[str] = []
    cursor = 0
    index = 0
    while index < len(matches):
        pair = (words[index], words[index + 1]) if index + 1 < len(words) else None
        replacement = pairs.get(pair) if pair else None
        if replacement is not None:
            start, end = matches[index].start(), matches[index + 1].end()
            pieces.append(text[cursor:start])
            # Chèn CẢ cụm thay thế (cặp có thể 1→2 từ), giữ kiểu hoa/thường của từ gốc đầu.
            first = matches[index].group(0)
            right = " ".join(replacement)
            pieces.append(right.capitalize() if first[:1].isupper() else right)
            cursor = end
            used.append(f"{pair[0]} {pair[1]}")
            index += 2
        else:
            index += 1
    pieces.append(text[cursor:])
    return "".join(pieces), used


def apply_fixes(manifest: str | Path, fixes: str | Path, out: str | Path,
                report_path: str | Path | None = None) -> dict:
    """Áp danh sách sửa lên `manifest`, ghi manifest mới + báo cáo."""
    payload = json.loads(Path(fixes).read_text(encoding="utf-8"))
    pairs = {tuple(item["wrong"].split()): tuple(item["right"].split()) for item in payload["pairs"]}
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line]
    changed = 0
    fired: Counter[str] = Counter()
    examples: list[dict] = []
    out_rows = []
    for row in rows:
        text = row.get("text") or ""
        new_text, used = _apply(text, pairs)
        if used:
            changed += 1
            fired.update(used)
            if len(examples) < 40:
                examples.append({"id": row.get("id"), "truoc": text[:150], "sau": new_text[:150]})
            row = {**row, "text": new_text}
        out_rows.append(row)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "manifest": str(manifest), "fixes": str(fixes), "out": str(out),
        "rows": len(rows), "rows_changed": changed,
        "pairs_total": len(pairs), "pairs_fired": len(fired),
        "fired": [{"pair": key, "n": value} for key, value in fired.most_common()],
        "pairs_never_fired": [f"{w} {r}" for (w, r) in pairs if f"{w} {r}" not in fired],
        "examples": examples,
    }
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main() -> None:
    """CLI: áp fixes, in tóm tắt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fixes", default="resources/transcript_fixes.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    report = apply_fixes(args.manifest, args.fixes, args.out, args.report)
    print(f"row {report['rows_changed']}/{report['rows']} đổi text · cặp khớp "
          f"{report['pairs_fired']}/{report['pairs_total']}")
    for item in report["fired"][:8]:
        print(f"  {item['pair']}: {item['n']} row")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()

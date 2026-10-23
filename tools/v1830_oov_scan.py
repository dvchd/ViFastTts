"""tools/v1830_oov_scan.py — liệt kê token OOV (status != VALID) trong manifest.

Vì sao (A0 v18.30 B3): trang nghe phải loại câu có OOV; đồng thời biết token nào cần
thêm cách đọc vào `data/codelung/custom_reading.json`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--out", default="artifacts/bench/v1830_oov_val.json")
    args = parser.parse_args()

    rows = [json.loads(x) for x in Path(args.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    counter: Counter[str] = Counter()
    bad_rows: list[dict] = []
    for row in rows:
        tokens = [s.get("raw") or s.get("normalized") for s in row.get("syllables", [])
                  if str(s.get("status", "VALID")) != "VALID"]
        if not tokens:
            continue
        counter.update(tokens)
        bad_rows.append({"id": row["id"], "tokens": tokens, "text": row.get("text", "")})
    report = {
        "manifest": args.manifest,
        "rows": len(rows),
        "rows_with_oov": len(bad_rows),
        "rows_pct": round(100.0 * len(bad_rows) / max(1, len(rows)), 2),
        "tokens": counter.most_common(50),
        "items": bad_rows[:200],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "rows_with_oov", "rows_pct")}, ensure_ascii=False))
    print("top tokens:", counter.most_common(12))


if __name__ == "__main__":
    main()

"""tools/syllable_coverage.py — độ phủ âm tiết trong TRAIN (A0 v11, Việc 7/2.1).

Vì sao: nếu "ách"/"ếch"/"rồi" xuất hiện quá ít trong train thì lỗi nghe là do ĐỘ PHỦ DỮ LIỆU,
không sửa được bằng loss. Tool đếm 3 mức: đúng âm tiết (kể cả thanh), cùng vần bỏ thanh, cùng coda.

Chạy:
  .venv/bin/python tools/syllable_coverage.py \
      --manifest data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl \
      --texts examples/drift_texts.json --focus "ách,ếch,rồi,bấm,treo,tường,ao" \
      --out artifacts/bench/coverage.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _rhyme(syllable: dict) -> str:
    orth = syllable.get("orthography") or {}
    abstract = syllable.get("abstract") or {}
    nuc = orth.get("nucleus") or abstract.get("nucleus") or ""
    coda = orth.get("coda") or abstract.get("coda") or ""
    return f"{nuc}{coda}".lower()


def _coda(syllable: dict) -> str:
    orth = syllable.get("orthography") or {}
    abstract = syllable.get("abstract") or {}
    return str(orth.get("coda") or abstract.get("coda") or "").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--texts", default="examples/drift_texts.json")
    parser.add_argument("--focus", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    exact: Counter[str] = Counter()
    rhyme: Counter[str] = Counter()
    coda: Counter[str] = Counter()
    total = 0
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for syllable in row.get("syllables", []):
            normalized = str(syllable.get("normalized") or syllable.get("raw") or "").lower()
            exact[normalized] += 1
            rhyme[_rhyme(syllable)] += 1
            coda[_coda(syllable)] += 1
            total += 1

    focus = [w.strip().lower() for w in args.focus.split(",") if w.strip()]
    sentences = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    wanted: set[str] = set(focus)
    if args.texts and Path(args.texts).exists():
        for row in sentences:
            for word in str(row.get("text", "")).replace(",", " ").replace(".", " ").split():
                wanted.add(word.lower())

    rows = []
    for word in sorted(wanted):
        rows.append({
            "syllable": word,
            "exact": exact.get(word, 0),
            "rhyme": None,
            "coda": None,
        })
    # Vần/coda cần parse âm tiết — dùng frontend cho các từ cần soi.
    from vifasttts.frontend.syllable import parse_syllable

    for row in rows:
        try:
            parsed = parse_syllable(row["syllable"])
            orth = parsed.orthography if hasattr(parsed, "orthography") else None
        except Exception:  # noqa: BLE001 - từ không phải âm tiết (dấu câu, số)
            orth = None
        if orth is None:
            row["rhyme"] = None
            row["coda"] = None
            continue
        key = f"{orth.nucleus}{orth.coda}".lower() if orth.coda else str(orth.nucleus).lower()
        row["rhyme"] = rhyme.get(key, 0)
        row["coda"] = coda.get(str(orth.coda).lower(), 0) if orth.coda else 0

    rare = [r for r in rows if r["exact"] < 10]
    result = {"manifest": args.manifest, "total_tokens": total, "n_types": len(exact),
              "focus": focus, "rows": rows, "rare_lt10": rare}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train: {total} token · {len(exact)} type · thiếu (<10 lần): {len(rare)}")
    for r in rows:
        if r["exact"] < 10:
            print(f"  {r['syllable']:10s} exact {r['exact']:5d} · vần {r['rhyme']} · coda {r['coda']}")


if __name__ == "__main__":
    main()

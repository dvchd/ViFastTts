"""tools/audit_slot_c.py — slot C nhận bao nhiêu frame theo coda, tách theo việc MFA có/không có phone cuối.

Vì sao (A0 lượt 5, §3.4): ~28 % token coda T bị MFA bỏ phone cuối (`t̚`/`k̚`) — cần biết rule hybrid
đang gán slot C bao nhiêu frame cho các token đó (0? tối thiểu?) so với token có phone. Nếu slot C
= 0 frame thì model không có chỗ để học coda ⇒ phải sửa rule, không phải re-align.

Cách làm: ghép mỗi segment trong manifest với các word trong TextGrid của clip cha (theo thời gian
`frame_range`), so khớp chuỗi từ bằng difflib, rồi đọc `seg_durations[i][3]` (slot C) tương ứng.

Chạy:
  .venv/bin/python tools/audit_slot_c.py --manifest data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl \
      --textgrid-dir data/codelung/mfa_v2c_vn_aligned --sample 400 \
      --out artifacts/dataset_v2/slot_c_audit.json
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_EXPECTED = {"NH": "ɲ", "N": "n", "NG": "ŋ", "M": "m", "CH": "c", "T": "t̚", "P": "p", "K": "k̚"}


def _stats(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.array(values)
    return {"n": len(values), "median": float(np.median(arr)), "mean": round(float(arr.mean()), 2),
            "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
            "zero_pct": round(float((arr == 0).mean() * 100), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--textgrid-dir", required=True)
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_mfa_coda import _parse

    from vifasttts.frontend.syllable import parse_syllable

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    step = max(1, len(rows) // args.sample)
    rows = rows[::step][: args.sample]
    by_coda: dict[str, list[int]] = defaultdict(list)
    split: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"present": [], "missing": []})
    n_matched = n_unmatched = 0
    for row in rows:
        tg_path = Path(args.textgrid_dir) / f"{row.get('parent', '')}.TextGrid"
        if not tg_path.exists() or not row.get("frame_range"):
            continue
        tiers = _parse(tg_path)
        start = row["frame_range"][0] * args.hop / args.sample_rate
        end = row["frame_range"][1] * args.hop / args.sample_rate
        words = [(a, b, c) for a, b, c in tiers.get("words", []) if c.strip() and a >= start - 0.05 and b <= end + 0.05]
        phones = [(a, b, c) for a, b, c in tiers.get("phones", []) if c.strip()]
        syllables = row["syllables"]
        raws = [str(s.get("raw", "")).strip().lower() for s in syllables]
        word_texts = [w[2].strip().lower() for w in words]
        matcher = difflib.SequenceMatcher(a=raws, b=word_texts, autojunk=False)
        pairs = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                pairs.extend(zip(range(i1, i2), range(j1, j2)))
        if not pairs:
            n_unmatched += 1
            continue
        n_matched += 1
        for i, j in pairs:
            try:
                coda = parse_syllable(str(syllables[i].get("raw", ""))).abstract.coda
            except (ValueError, KeyError, TypeError):
                continue
            slot_c = int(row["seg_durations"][i][3])
            by_coda[coda or "NONE"].append(slot_c)
            w_start, w_end, _ = words[j]
            inside = [(a, b, c) for a, b, c in phones
                      if min(b, w_end) - max(a, w_start) > 0.5 * max(1e-6, b - a)]
            expected = _EXPECTED.get(coda, "")
            present = bool(inside) and inside[-1][2] == expected
            split[coda or "NONE"]["present" if present else "missing"].append(slot_c)

    result = {"segments": len(rows), "matched": n_matched, "unmatched": n_unmatched,
              "by_coda": {coda: _stats(v) for coda, v in sorted(by_coda.items())},
              "split": {coda: {k: _stats(v) for k, v in parts.items()} for coda, parts in sorted(split.items())}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"segment khớp {n_matched}/{n_matched + n_unmatched}")
    print(f"{'coda':6s} {'n':>6s} {'median':>7s} {'p10':>5s} {'p90':>5s} {'C=0%':>6s} | "
          f"{'có phone':>18s} {'mất phone':>18s}")
    for coda, s in result["by_coda"].items():
        if s["n"] < 20:
            continue
        pres = result["split"][coda]["present"]
        miss = result["split"][coda]["missing"]
        fmt = lambda d: f"n={d['n']:4d} med={d.get('median', float('nan')):.1f}" if d["n"] else "n=0"
        print(f"{coda:6s} {s['n']:6d} {s['median']:7.1f} {s['p10']:5.1f} {s['p90']:5.1f} {s['zero_pct']:5.1f}% | "
              f"{fmt(pres):>18s} {fmt(miss):>18s}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()

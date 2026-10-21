"""tools/audit_mfa_coda.py — kiểm chứng MFA có phát âm COD A đúng chính tả không (A0 lượt 4, Việc 1b.2.2).

Vì sao: nếu aligner chọn phone sai (vd `anh` → /a n/ thay vì /a ɲ/), toàn bộ target
duration/boundary của âm tiết đó bị lệch ⇒ model học ngọng nh↔n. Tool đếm từ TextGrid:
mỗi word có coda **NH** (theo frontend) thì phone cuối MFA chọn là `ɲ`, `n` hay khác;
tương tự **CH** (`c` vs `t`) và **NG** (`ŋ` vs `n`). Tách theo vần (bỏ thanh).

Chạy:
  .venv/bin/python tools/audit_mfa_coda.py --dir data/codelung/mfa_v2c_vn_aligned \
      --out artifacts/dataset_v2/mfa_coda_audit.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_INTERVAL = re.compile(r'xmin = ([\d.eE+-]+)\s*\n\s*xmax = ([\d.eE+-]+)\s*\n\s*text = "(.*?)"', re.DOTALL)
_TIER = re.compile(r'item \[\d+\]:', re.DOTALL)
_EXPECTED = {"NH": "ɲ", "N": "n", "NG": "ŋ", "M": "m", "CH": "c", "T": "t", "P": "p", "K": "k"}


def _parse(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Trả {tên tier: [(xmin, xmax, text)]} — parser tối giản cho TextGrid chuẩn MFA."""
    text = path.read_text(encoding="utf-8")
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    starts = [m.start() for m in _TIER.finditer(text)] + [len(text)]
    for begin, end in itertools.pairwise(starts):
        block = text[begin:end]
        name_match = re.search(r'name = "(.*?)"', block)
        if not name_match:
            continue
        tiers[name_match.group(1)] = [(float(a), float(b), c) for a, b, c in _INTERVAL.findall(block)]
    return tiers


def _strip_tone(word: str) -> str:
    """Vần không dấu thanh (chỉ để gom nhóm thống kê)."""
    import unicodedata

    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from vifasttts.frontend.syllable import parse_syllable

    files = sorted(Path(args.dir).glob("*.TextGrid"))
    if args.limit:
        files = files[:args.limit]
    tally: dict[str, Counter] = defaultdict(Counter)
    by_rime: dict[str, Counter] = defaultdict(Counter)
    by_nucleus: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    examples: dict[str, list[str]] = defaultdict(list)
    words_total = phones_unmatched = 0
    for index, path in enumerate(files, start=1):
        tiers = _parse(path)
        words = [(a, b, c) for a, b, c in tiers.get("words", []) if c.strip()]
        phones = [(a, b, c) for a, b, c in tiers.get("phones", []) if c.strip()]
        for w_start, w_end, word in words:
            words_total += 1
            try:
                abstract = parse_syllable(word.strip()).abstract
            except (ValueError, KeyError, TypeError):
                continue
            coda = abstract.coda
            if coda not in ("NH", "N", "NG", "CH", "T"):
                continue
            inside = [(b, c) for a, b, c in phones
                      if min(b, w_end) - max(a, w_start) > 0.5 * max(1e-6, b - a)]
            if not inside:
                phones_unmatched += 1
                continue
            last_phone = inside[-1][1]
            expected = _EXPECTED.get(coda, "")
            verdict = "ok" if last_phone == expected else ("n_for_nh" if last_phone == "n" else last_phone)
            tally[coda][verdict] += 1
            rime = _strip_tone(word.strip())
            by_rime[rime][verdict] += 1
            by_nucleus[coda][abstract.nucleus][verdict] += 1
            if verdict != "ok" and len(examples[f"{coda}:{verdict}"]) < 8:
                examples[f"{coda}:{verdict}"].append(f"{word}({rime})→{last_phone}")
        if index % 2000 == 0:
            print(f"  {index}/{len(files)} · từ {words_total}", flush=True)

    summary = {}
    for coda, counter in sorted(tally.items()):
        total = sum(counter.values())
        summary[coda] = {"total": total, **{k: v for k, v in counter.most_common()},
                         "n_for_expected": counter.get("n_for_nh", 0) if coda == "NH" else None}
    result = {"files": len(files), "words": words_total, "phones_unmatched": phones_unmatched,
              "summary": summary,
              "by_rime": {rime: dict(counter) for rime, counter in sorted(by_rime.items())},
              "by_nucleus": {coda: {nuc: dict(counter) for nuc, counter in sorted(nucs.items())}
                             for coda, nucs in sorted(by_nucleus.items())},
              "examples": dict(examples)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    nh = tally.get("NH", Counter())
    nh_total = sum(nh.values())
    if nh_total:
        print(f"NH→n: {nh.get('n_for_nh', 0)}/{nh_total} = {nh.get('n_for_nh', 0) / nh_total * 100:.1f}%")
    print(f"→ {out}")


if __name__ == "__main__":
    main()

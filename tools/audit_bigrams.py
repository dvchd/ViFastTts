"""tools/audit_bigrams.py — soát lỗi chính tả transcript bằng **thống kê bigram của chính corpus**.

Vì sao (V11S3/P0.2): text dạy máy có lỗi nguồn (đo được "thư **dục**" thay vì "thư **mục**";
`dục` xuất hiện 2 lần / `mục` 119 lần trong 258k từ). Từ-hiếm đơn thuần **không đủ** (nhiều từ
hiếm vẫn hợp lệ). Tín hiệu mạnh hơn: **bigram hiếm** mà một **biến thể 1 âm tiết** của nó lại
**phổ biến** — đúng dạng lỗi gõ/nghe nhầm.

Thuật toán:
1. Đếm unigram + bigram (chữ thường) trên toàn manifest.
2. Với mỗi bigram **hiếm** (≤ `--max-count`): sinh biến thể của từng từ (giữ vần khác đầu ·
   giữ đầu+vần khác thanh), chỉ nhận biến thể CÓ trong từ vựng corpus.
3. Nếu bigram biến thể **phổ biến** (≥ `--min-support`) và bigram gốc **hiếm hơn nhiều**
   (tỉ lệ ≥ `--ratio`) ⇒ gắn cờ "nghi lỗi" kèm ngữ cảnh.

Chạy:
    .venv/bin/python tools/audit_bigrams.py --manifest data/codelung/train_clean_v2.jsonl \
        --out artifacts/dataset_v2/audit_bigrams.json

Kết quả JSON: `{stats, suspects: [{bigram, count, fix, fix_count, ratio, contexts}]}` —
`suspects` xếp theo tỉ lệ giảm; KHÔNG tự sửa gì (sửa tay/duyệt trước — đây là công cụ soi).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_TONE_KEYS = ("sắc", "huyền", "hỏi", "ngã", "nặng")


def _decompose(word: str) -> tuple[str, str, str] | None:
    """Tách âm tiết Việt thành (đầu, vần, thanh) bằng chính bộ chuẩn hoá của repo.

    Dùng `frontend.orthography` + `frontend.syllable` (không tự chế bảng) — None nếu không parse.
    """
    from vifasttts.frontend.orthography import compose_canonical
    from vifasttts.frontend.syllable import safe_parse_syllable

    item = safe_parse_syllable(word)
    if item is None:
        return None
    o, a = item.orthography, item.abstract
    onset = o.onset or ""
    rime = compose_canonical(None, o.medial, o.nucleus, o.coda, "NGANG")
    return onset, rime, a.tone.name


def _variants(word: str, vocabulary: set[str], by_rime_tone: dict[str, set[str]],
              by_onset_rime: dict[str, set[str]]) -> set[str]:
    """Biến thể **đúng MỘT khác biệt** (lỗi gõ thường vậy — tránh nhiễu 2 sửa):

    (a) cùng vần+thanh, khác đầu ("dục"→"mục"); (b) cùng đầu+vần, khác thanh ("sẽ"↔"sẻ").
    """
    parts = _decompose(word)
    if parts is None:
        return set()
    onset, rime, tone = parts
    out = {cand for cand in by_rime_tone.get(f"{rime}|{tone}", set()) if cand != word}
    out |= {cand for cand in by_onset_rime.get(f"{onset}|{rime}", set()) if cand != word}
    return {cand for cand in out if cand in vocabulary}


def audit(manifest: str | Path, *, max_count: int = 2, min_support: int = 5, ratio: float = 3.0,
          limit_context: int = 3) -> dict:
    """Soát bigram hiếm có biến thể phổ biến — trả báo cáo (không tự sửa)."""
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line]
    unigram: Counter[str] = Counter()
    bigram: Counter[tuple[str, str]] = Counter()
    contexts: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        words = [w.lower() for w in _WORD_RE.findall(row["text"])]
        unigram.update(words)
        bigram.update(itertools.pairwise(words))
        for left, right in itertools.pairwise(words):
            pair = (left, right)
            if len(contexts[pair]) < limit_context:
                index = words.index(left)
                contexts[pair].append(" ".join(words[max(0, index - 3):index + 5]))

    vocabulary = {w for w in unigram if len(w) > 1}
    by_rime_tone: dict[str, set[str]] = defaultdict(set)
    by_onset_rime: dict[str, set[str]] = defaultdict(set)
    for word in vocabulary:
        parts = _decompose(word)
        if parts is None:
            continue
        onset, rime, tone = parts
        by_rime_tone[f"{rime}|{tone}"].add(word)
        by_onset_rime[f"{onset}|{rime}"].add(word)

    suspects = []
    for (left, right), count in bigram.items():
        if count > max_count:
            continue
        candidates: list[tuple[str, int, tuple[str, str], str]] = []
        for position, word in ((0, left), (1, right)):
            # Từ GỐC phải hiếm (lỗi gõ là hiếm) — tránh "sửa" từ phổ biến thành từ khác.
            if unigram.get(word, 0) > 3:
                continue
            for variant in _variants(word, vocabulary, by_rime_tone, by_onset_rime):
                if unigram.get(variant, 0) < 20:  # từ "đúng" phải phổ biến
                    continue
                fixed = (variant, right) if position == 0 else (left, variant)
                support = bigram.get(fixed, 0)
                if support >= min_support and support >= count * ratio:
                    candidates.append((variant, support, fixed, word))
        if not candidates:
            continue
        variant, support, fixed, original = max(candidates, key=lambda item: item[1])
        suspects.append({
            "bigram": f"{left} {right}", "count": count,
            "fix": f"{fixed[0]} {fixed[1]}", "fix_count": support,
            "ratio": round(support / max(count, 1), 2), "changed": variant,
            "original_word": original, "original_freq": unigram.get(original, 0),
            "contexts": contexts[(left, right)],
        })
    suspects.sort(key=lambda item: (-item["fix_count"], -item["ratio"]))
    return {
        "manifest": str(manifest),
        "stats": {"clips": len(rows), "distinct_words": len(unigram), "tokens": sum(unigram.values()),
                  "distinct_bigrams": len(bigram),
                  "rare_bigrams": sum(1 for c in bigram.values() if c <= max_count),
                  "suspects": len(suspects)},
        "params": {"max_count": max_count, "min_support": min_support, "ratio": ratio},
        "suspects": suspects,
    }


def main() -> None:
    """CLI: soát bigram manifest, ghi JSON + in top nghi vấn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-count", type=int, default=2)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--ratio", type=float, default=3.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(args.manifest, max_count=args.max_count, min_support=args.min_support,
                   ratio=args.ratio)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = report["stats"]
    print(f"clip {stats['clips']} · từ khác nhau {stats['distinct_words']} · bigram hiếm "
          f"{stats['rare_bigrams']} · NGHI {stats['suspects']}")
    for item in report["suspects"][:15]:
        print(f"  {item['bigram']!r} ({item['count']}) → {item['fix']!r} ({item['fix_count']}) "
              f"· {item['contexts'][0][:70] if item['contexts'] else ''}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()

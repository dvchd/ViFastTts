"""Audit cặp âm tiết của frontend — 4 câu hỏi đo được:

1. **Fallback**: âm tiết nào trong ngữ liệu không parse được?
2. **Gộp** (cùng abstract, khác chữ): có phải toàn biến thể chính tả/đồng âm hợp lệ?
   (bẫy cần tránh: hai âm KHÁC NHAU bị gộp oan).
3. **Tách-thừa** (cùng phiên âm Bắc, khác abstract): các họ chủ đích như S/X, CH/TR,
   D/GI/R (trung lập phương ngữ) — hay còn họ nào khác?
4. **Cảnh báo strict** theo mã (chính tả lạ trong dữ liệu, không phải lỗi parser).

Chạy:
    python tools/audit_syllable_pairs.py data/codelung/train_clean_v2_dur_seg_slot4.jsonl ...
    python tools/audit_syllable_pairs.py data/codelung/*.jsonl --report artifacts/frontend_pairs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for extra in (_ROOT, _ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from vifasttts.frontend.g2p import syllable_to_phonemes, to_phoneme_string
from vifasttts.frontend.lexicon import strict_warnings
from vifasttts.frontend.syllable import safe_parse_syllable


def audit_syllables(syllables: set[str]) -> dict:
    """Chạy 4 phép kiểm trên tập âm tiết duy nhất; trả dict để in/ghi JSON."""
    fallback: list[str] = []
    abs2spell: dict[tuple, set[str]] = defaultdict(set)
    ipa2abs: dict[str, set[tuple]] = defaultdict(set)
    warnings: Counter[str] = Counter()
    warning_examples: dict[str, list[str]] = defaultdict(list)

    for word in sorted(syllables):
        parsed = safe_parse_syllable(word)
        if parsed is None:
            fallback.append(word)
            continue
        abstract = parsed.abstract
        key = (
            abstract.onset, abstract.medial, abstract.nucleus,
            abstract.coda, abstract.tone.value,
        )
        abs2spell[key].add(word)
        ipa = to_phoneme_string(syllable_to_phonemes(parsed))
        ipa2abs[ipa].add(key)
        for issue in strict_warnings(parsed):
            warnings[issue] += 1
            examples = warning_examples[issue]
            if len(examples) < 5:
                examples.append(word)

    merges = [
        {"abstract": "+".join(map(str, key)), "spellings": sorted(words)}
        for key, words in sorted(abs2spell.items()) if len(words) > 1
    ]
    splits = []
    for ipa, keys in sorted(ipa2abs.items()):
        if len(keys) < 2:
            continue
        spellings = sorted({w for key in keys for w in abs2spell[key]})
        splits.append({
            "ipa": ipa,
            "onsets": sorted({str(key[0]) for key in keys}),
            "spellings": spellings[:10],
        })
    split_families = Counter(
        "/".join(item["onsets"]) for item in splits
    )
    return {
        "syllables": len(syllables),
        "fallback": fallback,
        "merges": merges,
        "splits": splits,
        "split_families": dict(split_families),
        "strict_warnings": dict(warnings),
        "strict_examples": {code: words for code, words in sorted(warning_examples.items())},
    }


def load_unique_syllables(paths: list[str]) -> set[str]:
    """Đọc các manifest JSONL, gom `syllables[].normalized` của mọi row."""
    syllables: set[str] = set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for syllable in row.get("syllables") or []:
                    syllables.add(syllable["normalized"])
    return syllables


def main() -> None:
    """Đọc tham số, chạy audit, in tóm tắt và (tuỳ chọn) ghi báo cáo JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", help="Manifest JSONL có 'syllables'")
    parser.add_argument("--report", help="Ghi báo cáo JSON (vd artifacts/frontend_pairs.json)")
    args = parser.parse_args()

    report = audit_syllables(load_unique_syllables(args.manifests))
    print(f"Âm tiết duy nhất: {report['syllables']} | fallback: {len(report['fallback'])}")
    if report["fallback"]:
        print("  fallback:", report["fallback"][:20])
    print(f"Nhóm GỘP (cùng abstract, khác chữ): {len(report['merges'])}")
    for item in report["merges"][:40]:
        print(f"   {item['abstract']} <- {item['spellings']}")
    if len(report["merges"]) > 40:
        print(f"   ... (+{len(report['merges']) - 40} nhóm nữa trong báo cáo)")
    print(f"Nhóm TÁCH-THỪA (cùng IPA Bắc, khác abstract): {len(report['splits'])} "
          f"| họ: {report['split_families']}")
    if report["strict_warnings"]:
        print("Cảnh báo strict theo mã:", report["strict_warnings"])
        print("  ví dụ:", report["strict_examples"])

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

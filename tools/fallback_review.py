"""Rà soát từ fallback (B1) + merge cách đọc vào overrides.

Chạy:
    # 1) Dựng gói rà soát: top từ fallback (>= min-count) + clip mẫu + gợi ý auto
    python tools/fallback_review.py --manifest data/codelung/train.jsonl \
        --manifest data/codelung/val.jsonl --out data/codelung/review --top 400

    # 2) Nghe clip mẫu, điền cách đọc vào data/codelung/review/filled.txt
    #    (mỗi dòng:  <từ>|<cách đọc tiếng Việt>     — dòng nào không điền thì bỏ trống)
    # 3) Merge vào overrides (kiểm tra mọi cách đọc phải parse được)
    python tools/fallback_review.py --merge data/codelung/review/filled.txt \
        --overrides configs/overrides_tech_vi.json

Gợi ý auto (cần nghe xác nhận): dạng "-ing"/"-ed"/"-s" nếu từ gốc có trong bảng
`english_fallback`; từ viết tắt toàn hoa kiểm tra ACRONYM_READINGS.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from vifasttts.frontend.english_fallback import (
    ACRONYM_READINGS,
    CASE_INSENSITIVE_ENGLISH,
)
from vifasttts.frontend.pipeline import parse_text


def collect_fallback_words(manifests: list[str]) -> tuple[Counter[str], dict[str, list[str]]]:
    """Đếm từ fallback + giữ tối đa 3 id clip mẫu cho mỗi từ."""
    counter: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    for manifest in manifests:
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for syllable in row["syllables"]:
                if str(syllable.get("status", "VALID")) == "VALID":
                    continue
                token = str(syllable["normalized"])
                counter[token] += 1
                bucket = samples.setdefault(token, [])
                if len(bucket) < 3 and str(row["id"]) not in bucket:
                    bucket.append(str(row["id"]))
    return counter, samples


def suggest_reading(word: str) -> str | None:
    """Gợi ý cách đọc (best-effort) — None nếu không chắc."""
    if word.isupper() and word in ACRONYM_READINGS:
        return ACRONYM_READINGS[word]
    lowered = word.lower()
    if lowered in CASE_INSENSITIVE_ENGLISH:
        return CASE_INSENSITIVE_ENGLISH[lowered]
    for suffix, tail in (("ing", " ring"), ("ings", " ring"), ("ed", " đờ")):
        if lowered.endswith(suffix) and len(lowered) > len(suffix) + 1:
            base = lowered[: -len(suffix)].removesuffix("e")
            if base in CASE_INSENSITIVE_ENGLISH:
                return CASE_INSENSITIVE_ENGLISH[base] + tail
    return None


def _safe_name(word: str) -> str:
    return re.sub(r"[^\w.-]+", "_", word)[:40] or "word"


def build_review(manifests: list[str], out: str | Path, *, top: int = 400,
                 min_count: int = 2) -> dict[str, Any]:
    """Dựng gói rà soát: candidates.jsonl + filled.txt + clip mẫu (symlink)."""
    out_path = Path(out)
    clips_dir = out_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    counter, samples = collect_fallback_words(manifests)
    ranked = [(word, count) for word, count in counter.most_common() if count >= min_count][:top]

    # Bảng id → audio_path để tạo symlink clip mẫu.
    audio_by_id: dict[str, str] = {}
    for manifest in manifests:
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            audio_by_id[str(row["id"])] = str(row["audio_path"])

    rows: list[dict[str, Any]] = []
    for word, count in ranked:
        sample_ids = samples.get(word, [])
        for sample_id in sample_ids:
            source = audio_by_id.get(sample_id)
            if not source or not Path(source).exists():
                continue
            target = clips_dir / f"{_safe_name(word)}__{sample_id}.wav"
            if not target.exists():
                try:
                    os.symlink(Path(source).resolve(), target)
                except OSError:
                    shutil.copy2(source, target)
        rows.append({"word": word, "count": count, "clips": sample_ids,
                     "suggestion": suggest_reading(word)})
    (out_path / "candidates.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    if not (out_path / "filled.txt").exists():
        lines = []
        for row in rows:
            suggestion = row["suggestion"] or ""
            marker = "  # gợi ý (cần nghe xác nhận)" if suggestion else ""
            lines.append(f"{row['word']}|{suggestion}{marker}")
        (out_path / "filled.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"words": len(rows), "occurrences": sum(count for _, count in ranked),
            "suggested": sum(1 for row in rows if row["suggestion"])}


def _validate_reading(reading: str) -> bool:
    """Cách đọc phải parse thành âm tiết hợp lệ (giống test của english_fallback)."""
    items = parse_text(reading)
    return bool(items) and all(str(item.status) == "VALID" for item in items)


def merge_overrides(fill_path: str | Path, overrides_path: str | Path) -> dict[str, Any]:
    """Đọc `filled.txt` (từ|cách đọc), kiểm tra rồi merge vào overrides JSON."""
    overrides_file = Path(overrides_path)
    payload: dict[str, str] = {}
    if overrides_file.exists():
        payload = json.loads(overrides_file.read_text(encoding="utf-8"))
    added: dict[str, str] = {}
    invalid: list[tuple[str, str]] = []
    for line in Path(fill_path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "|" not in line:
            continue
        word, reading = (part.strip() for part in line.split("|", 1))
        if not word or not reading:
            continue
        if not _validate_reading(reading):
            invalid.append((word, reading))
            continue
        added[word] = reading
    for word, reading in added.items():
        payload[word] = reading
    overrides_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"added": len(added), "invalid": invalid, "total": len(payload)}


def main() -> None:
    """Subcommand đơn giản: build (mặc định) hoặc --merge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", help="Manifest JSONL (lặp lại được)")
    parser.add_argument("--out", default="data/codelung/review", help="Thư mục gói rà soát")
    parser.add_argument("--top", type=int, default=400, help="Số từ nhiều nhất giữ lại")
    parser.add_argument("--min-count", type=int, default=2, help="Bỏ qua từ xuất hiện ít hơn")
    parser.add_argument("--merge", help="Đường dẫn filled.txt cần merge")
    parser.add_argument("--overrides", default="configs/overrides_tech_vi.json",
                        help="File overrides JSON để merge vào")
    args = parser.parse_args()

    if args.merge:
        result = merge_overrides(args.merge, args.overrides)
        print(f"Đã thêm {result['added']} cách đọc (tổng {result['total']} mục) → {args.overrides}")
        for word, reading in result["invalid"]:
            print(f"  BỎ QUA (không parse được): {word} = {reading!r}")
        return
    if not args.manifest:
        raise SystemExit("Cần --manifest (hoặc --merge)")
    stats = build_review(args.manifest, args.out, top=args.top, min_count=args.min_count)
    print(
        f"Gói rà soát tại {args.out}: {stats['words']} từ (>= {args.min_count} lần, top {args.top}) "
        f"· {stats['occurrences']} lượt fallback · {stats['suggested']} gợi ý sẵn "
        f"— nghe clip mẫu rồi điền {args.out}/filled.txt, sau đó chạy --merge"
    )


if __name__ == "__main__":
    main()

"""tools/r2_weak_labels.py — R2: chọn nguồn weak label O/M/N/C (B4, Plan §4).

Ba phương án trên cùng dữ liệu:
  1. `ipa` — biên segment lấy trọn từ phone tier MFA (onset | medial | nucleus | coda).
  2. `rule` — weight rule thuần (`frontend/segments.py`; luôn chạy được).
  3. `lai` — biên onset|rhyme từ IPA + chia trong rhyme bằng weight rule
     (mặc định nếu coverage biên onset|rhyme ≥ 85% — B4).

Đo:
- coverage từng phương án + lý do fail (không thấy nguyên âm · coda thiếu · cấu trúc lạ);
- bảng phone→role thực nghiệm;
- thời lượng segment `lai` vs `ipa` trên âm tiết IPA đầy đủ — lượng hoá trước việc
  "weight rule lệch IPA bao nhiêu" (đúng kỳ vọng B5: timing thật cần biên IPA/aligner).

Chạy:
    python tools/r2_weak_labels.py --textgrid-dir data/codelung/mfa_v2c_vn_aligned \
        --dict data/codelung/vn_models/vietnamese_mfa.dict \
        --clips 200 --seed 42 --out artifacts/dataset_v2/r2_report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from mfa_to_durations import parse_long_textgrid

from vifasttts.frontend.segments import (
    _CODA_WEIGHTS,
    Segment,
    distribute_frames,
)
from vifasttts.frontend.syllable import safe_parse_syllable
from vifasttts.types import ParseStatus

TONE_CHARS = frozenset("˥˦˧˨˩")
SILENCE = {"", "sil", "sp", "spn", "silence", "<eps>"}
SR = 24000
HOP = 256


def is_vowel(label: str) -> bool:
    """Nguyên âm = phone có ký hiệu thanh (Chao) trong nhãn."""
    return any(char in TONE_CHARS for char in label)


def analyze(phones: list[tuple[float, float, str]], word: str) -> dict:
    """Phân tích một âm tiết: biên O|M|N|C theo IPA + cấu trúc so với parse.

    Trả về dict gồm: cờ coverage từng phương án, chỉ số biên, lý do fail.
    """
    parsed = safe_parse_syllable(word)
    if parsed is None or parsed.status != ParseStatus.VALID:
        return {"parse_ok": False}
    abstract = parsed.abstract
    labels = [label for _, _, label in phones]
    result: dict = {
        "parse_ok": True,
        "has_onset": abstract.onset != "NONE",
        "has_medial": abstract.medial != "NONE",
        "has_coda": abstract.coda != "NONE",
        "coda": abstract.coda,
        "nucleus": abstract.nucleus,
        "labels": labels,
    }
    vowels = [index for index, label in enumerate(labels) if is_vowel(label)]
    if not vowels:
        result["fail"] = "khong_nguyen_am"
        return result
    boundary = vowels[0]
    while boundary > 0 and labels[boundary - 1] == "w":
        boundary -= 1
    if abstract.onset == "NONE" and all(label == "ʔ" for label in labels[:boundary]):
        boundary = 0
    medial_end = boundary
    while medial_end < len(labels) and labels[medial_end] == "w":
        medial_end += 1
    last_vowel = vowels[-1]
    onset_labels = labels[:boundary]
    medial_labels = labels[boundary:medial_end]
    nucleus_labels = labels[medial_end:last_vowel + 1]
    coda_labels = labels[last_vowel + 1:]
    result.update({
        "boundary": boundary, "medial_end": medial_end, "last_vowel": last_vowel,
        "onset_labels": onset_labels, "medial_labels": medial_labels,
        "nucleus_labels": nucleus_labels, "coda_labels": coda_labels,
        "hybrid_ok": True,
    })
    onset_match = bool(onset_labels) == (abstract.onset != "NONE")
    medial_match = bool(medial_labels) == (abstract.medial != "NONE")
    coda_match = bool(coda_labels) == (abstract.coda != "NONE")
    result["onset_match"] = onset_match
    result["medial_match"] = medial_match
    result["coda_match"] = coda_match
    result["ipa_full_ok"] = onset_match and medial_match and coda_match
    if not result["ipa_full_ok"]:
        if not coda_match and abstract.coda != "NONE":
            result["fail"] = "coda_thieu"
        elif not onset_match:
            result["fail"] = "onset_lech"
        elif not medial_match:
            result["fail"] = "medial_lech"
        else:
            result["fail"] = "khac"
    return result


def _frame_of(seconds: float) -> int:
    """Giây → frame (làm tròn, cùng quy ước mfa_to_durations)."""
    return round(seconds * SR / HOP)


def hybrid_segments(result: dict, phones: list[tuple[float, float, str]]) -> dict:
    """Chia frame theo phương án `lai`: biên onset từ IPA, trong rhyme chia theo weight."""
    boundary = result["boundary"]
    total_start, total_end = phones[0][0], phones[-1][1]
    rhyme_time = phones[boundary][0] if boundary < len(phones) else total_end
    onset_frames = max(0, _frame_of(rhyme_time) - _frame_of(total_start))
    total_frames = max(1, _frame_of(total_end) - _frame_of(total_start))
    rhyme_frames = max(0, total_frames - onset_frames)
    segments: list[Segment] = []
    # Slot theo cấu trúc abstract (model chỉ active slot có thành phần) — không theo phone lạ.
    if result["has_medial"]:
        segments.append(Segment(unit="M", role="medial", weight=0.3))
    nucleus_weight = 1.2 if result.get("nucleus") == "OO" else 1.0
    segments.append(Segment(unit="N", role="nucleus", weight=nucleus_weight))
    if result["has_coda"]:
        segments.append(Segment(unit="C", role="coda", weight=_CODA_WEIGHTS.get(result["coda"], 0.35)))
    frames = distribute_frames(rhyme_frames, segments) if segments else []
    out = {"onset": onset_frames, "medial": 0, "nucleus": 0, "coda": 0}
    for segment, value in zip(segments, frames):
        out[segment.role] += value
    return out


def ipa_segments(result: dict, phones: list[tuple[float, float, str]]) -> dict:
    """Chia frame theo phương án `ipa` (biên từ phone tier)."""
    boundary, medial_end, last_vowel = result["boundary"], result["medial_end"], result["last_vowel"]
    total_start, total_end = phones[0][0], phones[-1][1]
    out = {"onset": 0, "medial": 0, "nucleus": 0, "coda": 0}
    if boundary > 0:
        out["onset"] = max(0, _frame_of(phones[boundary - 1][1]) - _frame_of(total_start))
    if medial_end > boundary:
        out["medial"] = max(0, _frame_of(phones[medial_end - 1][1]) - _frame_of(phones[boundary][0]))
    out["nucleus"] = max(0, _frame_of(phones[last_vowel][1]) - _frame_of(phones[medial_end][0]))
    if last_vowel + 1 < len(phones):
        out["coda"] = max(0, _frame_of(total_end) - _frame_of(phones[last_vowel + 1][0]))
    return out


def analyze_corpus(items: list[tuple[str, list[tuple[float, float, str]]]], *,
                   measure_durations: bool = True) -> dict:
    """Chạy analyzer + thống kê coverage/phone-role/lệch thời lượng trên danh sách âm tiết.

    `measure_durations=False` cho dict (thời gian trong dict là giả — chỉ đo coverage).
    """
    total = len(items)
    parse_fail = 0
    ipa_ok = hybrid_ok = 0
    fail_modes: Counter = Counter()
    fail_examples: dict[str, list[str]] = defaultdict(list)
    coda_missing: Counter = Counter()
    phone_role: dict[str, Counter] = {role: Counter() for role in ("onset", "medial", "nucleus", "coda")}
    delta: dict[str, list[float]] = defaultdict(list)

    def note(key: str, word: str, labels: list[str]) -> None:
        if len(fail_examples[key]) < 10:
            fail_examples[key].append(f"{word} ⟦{' '.join(labels)}⟧")

    for word, phones in items:
        result = analyze(phones, word)
        if not result.get("parse_ok"):
            parse_fail += 1
            continue
        if result.get("hybrid_ok"):
            hybrid_ok += 1
        else:
            key = result.get("fail", "khong_nguyen_am")
            fail_modes[key] += 1
            note(key, word, result.get("labels", []))
            continue
        if result["ipa_full_ok"]:
            ipa_ok += 1
            for role in ("onset", "medial", "nucleus", "coda"):
                for label in result[f"{role}_labels"]:
                    phone_role[role][label] += 1
            if measure_durations:
                ipa_durations = ipa_segments(result, phones)
                lai_durations = hybrid_segments(result, phones)
                for role in ("onset", "medial", "nucleus", "coda"):
                    delta[role].append((lai_durations[role] - ipa_durations[role]) * HOP / SR * 1000)
        else:
            key = result.get("fail", "khac")
            fail_modes[key] += 1
            note(key, word, result.get("labels", []))
            if not result["coda_match"] and result["has_coda"]:
                coda_missing[result["coda"]] += 1
    denominator = max(1, total - parse_fail)
    deltas = {
        role: {"mean_abs_ms": round(float(np.mean(np.abs(values))), 1),
               "mean_signed_ms": round(float(np.mean(values)), 1),
               "n": len(values)}
        for role, values in delta.items() if values
    }
    return {
        "syllables": total,
        "parse_fail": parse_fail,
        "coverage_ipa_pct": round(100.0 * ipa_ok / denominator, 1),
        "coverage_lai_pct": round(100.0 * hybrid_ok / denominator, 1),
        "coverage_rule_pct": 100.0,
        "fail_modes": dict(fail_modes),
        "fail_examples": {key: values for key, values in sorted(fail_examples.items())},
        "coda_thieu_theo_loai": dict(coda_missing),
        "phone_role_top": {role: dict(counter.most_common(12)) for role, counter in phone_role.items()},
        "lai_vs_ipa_ms": deltas,
    }


def _phones_in(intervals: list[tuple[float, float, str]], low: float, high: float):
    """Phone nằm trong khoảng [low, high] (bỏ im lặng)."""
    return [(start, end, label) for start, end, label in intervals
            if start >= low - 1e-6 and end <= high + 1e-6 and label.strip().lower() not in SILENCE]


def clip_items(textgrid_dir: Path, clips: int, seed: int) -> list[tuple[str, list]]:
    """Lấy mẫu clip, trả danh sách (word, phones) từ word+phones tier."""
    paths = sorted(textgrid_dir.glob("*.TextGrid"))
    picked = random.Random(seed).sample(paths, min(clips, len(paths)))
    items: list[tuple[str, list]] = []
    for path in picked:
        tiers = parse_long_textgrid(path)
        words = [(a, b, label) for a, b, label in tiers.get("words", []) if label.strip()]
        phones = tiers.get("phones", [])
        for start, end, label in words:
            subset = _phones_in(phones, start, end)
            if subset:
                items.append((label.strip(), subset))
    return items


def dict_items(dict_path: Path) -> list[tuple[str, list]]:
    """Mỗi mục dict → (word, [(0,1,phone), ...]) — thời gian giả, chỉ dùng cho coverage/table."""
    items: list[tuple[str, list]] = []
    for line in dict_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        word = parts[0].strip()
        phones = [(float(i), float(i + 1), label) for i, label in enumerate(parts[5].split())]
        if word and phones:
            items.append((word, phones))
    return items


def main() -> None:
    """Chạy R2 trên TextGrid + dict, ghi JSON + in bảng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrid-dir", required=True)
    parser.add_argument("--dict", dest="dict_path", required=True)
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    corpus = analyze_corpus(clip_items(Path(args.textgrid_dir), args.clips, args.seed))
    dictionary = analyze_corpus(dict_items(Path(args.dict_path)), measure_durations=False)
    report = {"corpus": corpus, "dict": dictionary}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    for name, summary in (("corpus (TextGrid)", corpus), ("dict", dictionary)):
        print(f"\n== R2 · {name} · {summary['syllables']} âm tiết "
              f"(parse fail {summary['parse_fail']}) ==")
        print(f"  coverage: ipa {summary['coverage_ipa_pct']}% · "
              f"lai (onset|rhyme) {summary['coverage_lai_pct']}% · rule 100%")
        print(f"  fail modes: {summary['fail_modes']}")
        for key, examples in summary.get("fail_examples", {}).items():
            print(f"    ví dụ {key}: {' · '.join(examples[:5])}")
        if summary["coda_thieu_theo_loai"]:
            print(f"  coda thiếu theo loại: {summary['coda_thieu_theo_loai']}")
        if summary["lai_vs_ipa_ms"]:
            print("  lệch thời lượng lai vs ipa (ms):", summary["lai_vs_ipa_ms"])
        print("  phone→role (top):")
        for role, table in summary["phone_role_top"].items():
            top = ", ".join(f"{phone}:{count}" for phone, count in list(table.items())[:8])
            print(f"    {role:8s} {top}")
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()

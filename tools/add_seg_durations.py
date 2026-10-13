"""tools/add_seg_durations.py — T2.1: sinh `seg_durations [N,4]` (weak label hybrid, R2).

Nguồn nhãn (docs/ACOUSTIC_V2.md §17): **hybrid** = biên onset|rhyme từ phone tier MFA
+ weight rule trong rhyme; fallback weight rule khi không tìm được nguyên âm / không có
onset thực. Slot theo cấu trúc abstract (inactive = 0 frame); min-1 frame cho slot active
khi đủ frame; `d_syl < n_act` → ưu tiên N, O, C, M (B2).

Bất biến: `Σ seg == d_syl` tuyệt đối (largest remainder) — verify 100%.
Ghi kèm `seg_method` (hybrid / hybrid+rule / rule), `seg_method_version`, `seg_weights_hash` (B9).

Chạy:
    python tools/add_seg_durations.py \
        --manifest data/codelung/train_clean_v2_dur_seg.jsonl \
        --manifest data/codelung/val_clean_v2_dur_seg.jsonl \
        --manifest data/codelung/holdout_clean_v2_dur.jsonl \
        --textgrid-dir data/codelung/mfa_v2c_vn_aligned \
        --out-dir data/codelung --report artifacts/dataset_v2/seg_durations_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for extra in (_ROOT, _ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from mfa_to_durations import parse_long_textgrid

from vifasttts.frontend.segments import (
    _CODA_WEIGHTS,
    WEIGHT_MEDIAL,
    WEIGHT_NUCLEUS,
    WEIGHT_ONSET,
)
from vifasttts.frontend.syllable import safe_parse_syllable
from vifasttts.types import ParseStatus

SR = 24000
HOP = 256
TONE_CHARS = frozenset("˥˦˧˨˩")
SILENCE = {"", "sil", "sp", "spn", "silence", "<eps>"}
ROLE_ORDER = ("O", "M", "N", "C")
PRIORITY = ("N", "O", "C", "M")
SEG_METHOD_VERSION = "t2.1-hybrid-v1"


def seg_weights_hash() -> str:
    """Hash bảng trọng số + version — ghi vào manifest/checkpoint để truy vết (B9)."""
    payload = {
        "onset": WEIGHT_ONSET, "medial": WEIGHT_MEDIAL, "nucleus": WEIGHT_NUCLEUS,
        "nucleus_oo": 1.2, "coda": {str(key): float(value) for key, value in sorted(_CODA_WEIGHTS.items())},
        "priority": PRIORITY, "version": SEG_METHOD_VERSION,
    }
    return sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def role_weights(abstract) -> dict[str, float]:
    """Trọng số 4 slot theo cấu trúc abstract (chỉ slot active)."""
    weights: dict[str, float] = {}
    if abstract.onset != "NONE":
        weights["O"] = WEIGHT_ONSET
    if abstract.medial != "NONE":
        weights["M"] = WEIGHT_MEDIAL
    weights["N"] = 1.2 if abstract.nucleus == "OO" else WEIGHT_NUCLEUS
    if abstract.coda != "NONE":
        weights["C"] = float(_CODA_WEIGHTS[abstract.coda])
    return weights


def distribute(frames: int, weights: dict[str, float]) -> dict[str, int]:
    """Chia `frames` theo trọng số: min-1 + largest remainder; thiếu frame → ưu tiên N,O,C,M."""
    result = {role: 0 for role in ROLE_ORDER}
    if frames <= 0 or not weights:
        return result
    roles = [role for role in ROLE_ORDER if role in weights]
    if frames < len(roles):
        for role in [role for role in PRIORITY if role in weights][:frames]:
            result[role] = 1
        return result
    remaining = frames - len(roles)
    total = sum(weights[role] for role in roles)
    raw = {role: weights[role] / total * remaining for role in roles}
    floors = {role: int(raw[role]) for role in roles}
    leftover = remaining - sum(floors.values())
    order = sorted(roles, key=lambda role: (-(raw[role] - floors[role]), PRIORITY.index(role)))
    for role in order[:leftover]:
        floors[role] += 1
    for role in roles:
        result[role] = 1 + floors[role]
    return result


def split_syllable(d_syl: int, weights: dict[str, float], onset_frames: int | None) -> dict[str, int]:
    """Chia một âm tiết hybrid: onset cố định từ IPA, rhyme chia theo trọng số.

    `onset_frames=None` → weight rule thuần (fallback). Bất biến: Σ == d_syl,
    inactive = 0, active ≥ 1 khi `d_syl ≥ n_act`.
    """
    if d_syl <= 0 or not weights:
        return {role: 0 for role in ROLE_ORDER}
    if d_syl < len(weights):
        return distribute(d_syl, weights)
    if onset_frames is None or "O" not in weights:
        return distribute(d_syl, weights)
    rhyme_roles = [role for role in ROLE_ORDER if role in weights and role != "O"]
    onset = max(0, min(int(onset_frames), d_syl - len(rhyme_roles)))
    result = distribute(d_syl - onset, {role: weights[role] for role in rhyme_roles})
    if onset == 0:
        donor = max(rhyme_roles, key=lambda role: (result[role], -PRIORITY.index(role)))
        if result[donor] >= 2:
            result[donor] -= 1
            onset = 1
    result["O"] = onset
    return result


def boundary_time(phones: list[tuple[float, float, str]], abstract) -> float | None:
    """Thời điểm bắt đầu rhyme (onset|rhyme) từ phone tier; None nếu không xác định được."""
    if not phones:
        return None
    if abstract.onset == "NONE":
        return phones[0][0]  # O inactive — mọi frame thuộc rhyme
    labels = [label for _, _, label in phones]
    vowels = [index for index, label in enumerate(labels) if any(char in TONE_CHARS for char in label)]
    if not vowels:
        return None
    boundary = vowels[0]
    while boundary > 0 and labels[boundary - 1] == "w":
        boundary -= 1
    return phones[boundary][0]


def starts_of(row: dict) -> list[int]:
    """Frame bắt đầu (tuyệt đối) của từng âm tiết trong row."""
    count = len(row["durations"])
    gaps = list(row.get("gaps") or [])
    if len(gaps) == count - 1:
        gaps = gaps + [0]
    if "frame_range" in row:
        cursor = int(row["frame_range"][0])
    else:
        cursor = int(row.get("lead_frames") or 0)
    starts: list[int] = []
    for index in range(count):
        starts.append(cursor)
        cursor += int(row["durations"][index]) + int(gaps[index])
    return starts


def load_textgrid_words(path: Path):
    """(words, phones) từ TextGrid — words đã bỏ khoảng trống, khớp thứ tự syllable."""
    tiers = parse_long_textgrid(path)
    words = [(float(a), float(b), label) for a, b, label in tiers.get("words", []) if label.strip()]
    phones = [(float(a), float(b), label) for a, b, label in tiers.get("phones", [])
              if label.strip().lower() not in SILENCE]
    return words, phones


def phones_within(phones: list, low: float, high: float) -> list:
    """Phone nằm trong [low, high] (dung sai nhỏ)."""
    return [(a, b, label) for a, b, label in phones if a >= low - 1e-4 and b <= high + 1e-4]


def _abstract_of(label: str):
    """AbstractSyllable của một từ TextGrid (None nếu parse fail)."""
    parsed = safe_parse_syllable(label)
    if parsed is None or parsed.status != ParseStatus.VALID:
        return None
    return parsed.abstract


def find_slice(parent: dict, row: dict) -> int | None:
    """Tìm chỉ số âm tiết đầu của row trong parent (khớp nhãn + thời gian ±6 frame)."""
    labels_row = [str(syllable.get("raw", "")) for syllable in row["syllables"]]
    labels_parent = [str(syllable.get("raw", "")) for syllable in parent["syllables"]]
    count = len(labels_row)
    starts_parent = starts_of(parent)
    starts_row = starts_of(row)
    best, best_delta = None, 99
    for index in range(len(labels_parent) - count + 1):
        if labels_parent[index:index + count] != labels_row:
            continue
        delta = abs(starts_parent[index] - starts_row[0])
        if delta < best_delta:
            best, best_delta = index, delta
    return best if best_delta <= 6 else None


def process_row(row: dict, parent: dict | None, parent_entries: list[dict]) -> tuple[dict, dict]:
    """Sinh seg_durations cho một row; trả (row mới, thống kê)."""
    count = len(row["syllables"])
    starts = starts_of(row)
    stats = {"syllables": count, "fallback": 0, "onset_missing": 0, "borrow": 0,
             "priority": 0, "invariant_fail": 0, "tier_fail": 0}
    seg_durations: list[list[int]] = []
    entries = parent_entries if parent is not None else []
    offset = 0
    if parent is not None:
        found = find_slice(parent, row)
        offset = found if found is not None else -1
        if offset < 0:
            entries = []
    for index in range(count):
        syllable = row["syllables"][index]
        parsed = safe_parse_syllable(str(syllable.get("raw", "")))
        d_syl = int(row["durations"][index])
        if parsed is None or parsed.status != ParseStatus.VALID:
            weights = {"N": WEIGHT_NUCLEUS}
            stats["fallback"] += 1
            seg = split_syllable(d_syl, weights, None)
        else:
            weights = role_weights(parsed.abstract)
            entry = (entries[offset + index]
                     if (parent is not None and 0 <= offset + index < len(entries)) else None)
            onset_frames = None
            if entry is not None:
                abstract, phones = entry
                if abstract is not None:
                    t_boundary = boundary_time(phones, abstract)
                    if t_boundary is not None:
                        onset_frames = max(0, round(t_boundary * SR / HOP) - starts[index])
                    else:
                        stats["fallback"] += 1
                        if abstract.onset != "NONE":
                            stats["onset_missing"] += 1
                else:
                    stats["fallback"] += 1
            else:
                stats["fallback"] += 1
            if d_syl < len(weights):
                stats["priority"] += 1
            seg = split_syllable(d_syl, weights, onset_frames)
            if (onset_frames is not None and "O" in weights and onset_frames == 0
                    and d_syl >= len(weights)):
                stats["borrow"] += 1
        if sum(seg.values()) != d_syl:
            stats["invariant_fail"] += 1
        seg_durations.append([seg[role] for role in ROLE_ORDER])
    if stats["fallback"] == 0:
        method = "hybrid"
    elif stats["fallback"] == count:
        method = "rule"
    else:
        method = "hybrid+rule"
    new_row = {
        **row,
        "seg_durations": seg_durations,
        "seg_method": method,
        "seg_method_version": SEG_METHOD_VERSION,
        "seg_weights_hash": seg_weights_hash(),
    }
    return new_row, stats


def main() -> None:
    """Chạy trên các manifest, ghi `<stem>_slot4.jsonl` + report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--textgrid-dir", required=True)
    parser.add_argument("--parent-dir", default=None,
                        help="Thư mục chứa `*_dur.jsonl` (mặc định: cùng thư mục manifest)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    textgrid_dir = Path(args.textgrid_dir)
    parent_dir = Path(args.parent_dir) if args.parent_dir else Path(args.manifest[0]).parent
    parents: dict[str, dict] = {}
    for path in sorted(parent_dir.glob("*_dur.jsonl")):
        if path.stem.endswith("_seg"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parent_row = json.loads(line)
                parents[str(parent_row["id"])] = parent_row
    print(f"Parent manifest: {len(parents)} row")

    grid_cache: dict[str, list[tuple]] = {}
    report: dict = {"version": SEG_METHOD_VERSION, "weights_hash": seg_weights_hash(), "manifests": {}}
    for manifest in args.manifest:
        source = Path(manifest)
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        out_rows: list[dict] = []
        totals: Counter = Counter()
        shares: dict[str, list[float]] = defaultdict(list)
        nucleus_share_by_tone: dict[str, list[float]] = defaultdict(list)
        active_counts: Counter = Counter()
        method_counts: Counter = Counter()
        for row in rows:
            entries: list[dict] = []
            parent = None
            if "frame_range" in row:
                parent = parents.get(str(row.get("parent", "")))
            else:
                parent = row  # clip đầy đủ — tự nó là parent
            if parent is not None:
                parent_id = str(parent["id"])
                if parent_id not in grid_cache:
                    grid_path = textgrid_dir / f"{parent_id}.TextGrid"
                    if grid_path.exists():
                        words, phones = load_textgrid_words(grid_path)
                        grid_cache[parent_id] = [
                            (_abstract_of(label), phones_within(phones, low, high))
                            for low, high, label in words
                        ]
                    else:
                        grid_cache[parent_id] = []
                entries = grid_cache[parent_id]
            new_row, stats = process_row(row, parent, entries)
            out_rows.append(new_row)
            for key, value in stats.items():
                totals[key] += value
            method_counts[new_row["seg_method"]] += 1
            for index, seg in enumerate(new_row["seg_durations"]):
                d_syl = max(1, int(row["durations"][index]))
                active = sum(1 for value in seg if value > 0)
                active_counts[active] += 1
                for role_index, role in enumerate(ROLE_ORDER):
                    shares[role].append(seg[role_index] / d_syl)
                tone = str(row["syllables"][index]["abstract"]["tone"])
                nucleus_share_by_tone[tone].append(seg[2] / d_syl)
        out_path = Path(args.out_dir) / f"{source.stem}_slot4.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for out_row in out_rows:
                handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
        summary = {
            "rows": len(out_rows), "syllables": totals["syllables"],
            "fallback_syllables": totals["fallback"],
            "onset_missing": totals["onset_missing"],
            "borrow_fixups": totals["borrow"],
            "priority_cases": totals["priority"],
            "invariant_fail": totals["invariant_fail"],
            "methods": dict(method_counts),
            "active_slot_counts": {str(key): value for key, value in sorted(active_counts.items())},
            "role_share": {role: {"mean": round(float(np.mean(values)), 3),
                                  "median": round(float(np.median(values)), 3)}
                           for role, values in shares.items() if values},
            "nucleus_share_by_tone": {tone: round(float(np.median(values)), 3)
                                      for tone, values in sorted(nucleus_share_by_tone.items())},
            "out": str(out_path),
        }
        report["manifests"][source.stem] = summary
        print(f"\n{source.stem}: {summary['rows']} row · {summary['syllables']} âm tiết · "
              f"fail bất biến {summary['invariant_fail']} · fallback {summary['fallback_syllables']} "
              f"({100 * summary['fallback_syllables'] / max(1, summary['syllables']):.1f}%) · "
              f"methods {summary['methods']}")
        print(f"  role share (mean): { {k: v['mean'] for k, v in summary['role_share'].items()} }")
        print(f"  nucleus share theo thanh: {summary['nucleus_share_by_tone']}")
        print(f"  → {out_path}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()

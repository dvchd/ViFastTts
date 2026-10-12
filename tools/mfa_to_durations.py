"""TextGrid MFA → `durations` cấp âm tiết (frames) + QC + thông tin pause (cho cắt segment).

Quy tắc (docs/AI_REVIEW_ANSWERS_R3 §3.3, REVISION_PLAN_V2 §9.2, REBUILD_PLAN_M6 §3(6)):
- Map word interval → âm tiết theo đúng thứ tự trong manifest (số từ phải khớp, lệch → loại clip).
- `durations[i] = frame(biên i+1) − frame(biên i)` với biên 0 = đầu clip, biên cuối = T frame
  (T = 1 + len(wav)//hop) ⇒ **âm tiết đầu gánh khoảng lặng đầu clip, âm tiết cuối gánh đuôi clip**,
  khoảng lặng giữa các từ nằm trong âm tiết đứng trước (đúng tinh thần "gán silence vào âm tiết
  mang dấu câu" khi chưa có SIL token).
- **Pause ≥150 ms giữa hai từ**: nếu âm tiết trước kết thúc câu/đoạn (`boundaries.sentence_end|phrase_end`)
  → giữ và ghi vào `pauses` (để `cut_segments.py` cắt segment tại đó); nếu nằm giữa câu
  (ngừng thở) → **loại clip** (D5).
- Vi phạm cứng (C4): phone > 500 ms · âm tiết hữu thanh > 1,0 s → loại clip.
- **Assert** `sum(durations) == T`, mọi âm tiết ≥ 1 frame; không xếp được → loại clip + log.

Chạy (sau `mfa align`):
    python tools/mfa_to_durations.py --textgrid-dir data/codelung/mfa_v2b_aligned \
        --manifest data/codelung/train_clean_v2.jsonl --manifest data/codelung/val_clean_v2.jsonl \
        --manifest data/codelung/holdout_clean_v2.jsonl --out-dir data/codelung
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

MERGE_PAUSE_SECONDS = 0.150   # pause ≥ ngưỡng: tách SIL/loại (D5)
HARD_PHONE_SECONDS = 0.500    # phone quá dài (C4)
HARD_SYLLABLE_SECONDS = 1.000  # âm tiết hữu thanh quá dài (C4)


def parse_long_textgrid(path: str | Path) -> dict[str, list[tuple[float, float, str]]]:
    """Đọc TextGrid long (Praat) → {tier_name: [(xmin, xmax, text), ...]}."""
    text = Path(path).read_text(encoding="utf-8")
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    for item in re.split(r"item\s*\[\d+\]\s*:", text)[1:]:
        match = re.search(r'name\s*=\s*"([^"]*)"', item)
        if not match:
            continue
        name = match.group(1)
        intervals: list[tuple[float, float, str]] = []
        for xmin, xmax, label in re.findall(
            r'xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*"((?:[^"\\]|\\.)*)"',
            item,
        ):
            intervals.append((float(xmin), float(xmax), label.replace('\\"', '"')))
        tiers[name] = intervals
    return tiers


def compute_durations(
    word_intervals: list[tuple[float, float, str]],
    *,
    frames_total: int,
    sample_rate: int,
    hop_length: int,
    syllable_boundaries: list[dict[str, Any]],
    phrase_boundary_ok: bool = True,
) -> dict[str, Any]:
    """Tính durations (frames) cho một clip từ word intervals của TextGrid.

    Trả về dict: `{"durations": [...], "pauses": [...], "dropped": None|reason}`.
    `syllable_boundaries[i]` là `boundaries` (dict) của âm tiết i trong manifest.
    """
    count = len(word_intervals)
    if count == 0:
        return {"dropped": "không có từ trong TextGrid"}
    if count != len(syllable_boundaries):
        return {"dropped": f"số từ lệch manifest ({count} vs {len(syllable_boundaries)})"}

    # Pause giữa hai từ: kiểm tra ngưỡng + vị trí ranh giới câu.
    pauses: list[dict[str, Any]] = []
    for i in range(count - 1):
        gap = word_intervals[i + 1][0] - word_intervals[i][1]
        if gap < MERGE_PAUSE_SECONDS:
            continue
        bounds = syllable_boundaries[i] or {}
        at_boundary = bool(bounds.get("sentence_end") or (phrase_boundary_ok and bounds.get("phrase_end")))
        if not at_boundary:
            return {"dropped": f"ngừng thở {gap * 1000:.0f} ms giữa câu (sau âm tiết {i})"}
        pauses.append({"after": i, "seconds": round(gap, 3)})

    starts = [interval[0] for interval in word_intervals]
    frames_for: list[int] = [0]
    for i in range(1, count + 1):
        target = frames_total if i == count else round(starts[i] * sample_rate / hop_length)
        least = frames_for[-1] + 1
        most = frames_total - (count - i)
        if least > most:
            return {"dropped": "không thể xếp boundary (audio quá ngắn so với số âm tiết)"}
        frames_for.append(min(max(target, least), most))
    if frames_for[-1] != frames_total:
        return {"dropped": "boundary cuối không khớp T"}

    durations = [frames_for[i + 1] - frames_for[i] for i in range(count)]
    if any(d <= 0 for d in durations):  # phòng thủ; nhánh trên đã đảm bảo ≥1
        return {"dropped": "có âm tiết 0 frame"}
    return {"durations": durations, "pauses": pauses, "dropped": None}


def _duration_seconds(word_intervals: list[tuple[float, float, str]]) -> list[float]:
    """Độ dài hữu thanh (giây) của từng từ theo word tier."""
    return [max(0.0, xmax - xmin) for xmin, xmax, _ in word_intervals]


def load_mfa_scores(db_path: str | Path) -> dict[str, float]:
    """Đọc điểm align từ DB MFA: `{clip_id: alignment_log_likelihood / num_frames}`.

    Bảng: `utterance.file_id → sound_file.file_id → sound_file_path` (lấy stem làm id).
    Chỉ trả các utterance đã có điểm (sau `mfa align`); bảng thiếu/cột trống → trả {}.
    """
    import sqlite3

    try:
        connection = sqlite3.connect(str(db_path))
        rows = connection.execute(
            "select sf.sound_file_path, u.alignment_log_likelihood, u.num_frames "
            "from utterance u join sound_file sf on u.file_id = sf.file_id "
            "where u.alignment_log_likelihood is not null and u.num_frames > 0"
        ).fetchall()
        connection.close()
    except sqlite3.Error:
        return {}
    scores: dict[str, float] = {}
    for path, likelihood, num_frames in rows:
        scores[Path(str(path)).stem] = float(likelihood) / float(num_frames)
    return scores


def score_drop_ids(scores: dict[str, float], fraction: float) -> set[str]:
    """Tập id cần loại = `fraction` thấp nhất theo điểm log-likelihood/frame (C4)."""
    if not scores or fraction <= 0:
        return set()
    ordered = sorted(scores.items(), key=lambda item: item[1])
    cut = int(len(ordered) * fraction)
    return {sample_id for sample_id, _ in ordered[:cut]}


def convert_manifest(manifest: str | Path, textgrid_dir: str | Path, out_path: str | Path, *,
                     sample_rate: int = 24000, hop_length: int = 256,
                     drop_score_ids: set[str] | None = None) -> dict[str, Any]:
    """Chuyển một manifest: thêm `durations` (+`pauses`), loại clip lỗi; ghi manifest mới + trả thống kê."""
    from vifasttts.data.features import load_wav

    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    textgrid_dir = Path(textgrid_dir)
    kept: list[dict[str, Any]] = []
    dropped: dict[str, list[str]] = {}
    stats = {"clips": len(rows), "kept": 0, "one_frame": 0, "pauses": 0,
             "syllables": 0, "median_ms": 0.0, "p95_ms": 0.0}

    def drop(sample_id: str, reason: str) -> None:
        dropped.setdefault(reason, []).append(sample_id)

    durations_all: list[float] = []
    for row in rows:
        sample_id = str(row["id"])
        if drop_score_ids and sample_id in drop_score_ids:
            drop(sample_id, "score thấp (C4)")
            continue
        grid_path = textgrid_dir / f"{sample_id}.TextGrid"
        if not grid_path.exists():
            drop(sample_id, "thiếu TextGrid")
            continue
        tiers = parse_long_textgrid(grid_path)
        words = [(a, b, t) for a, b, t in tiers.get("words", []) if t.strip()]
        phones = tiers.get("phones", [])
        if any(0.0 < (xmax - xmin) > HARD_PHONE_SECONDS for xmin, xmax, _ in phones if xmax > xmin):
            drop(sample_id, f"phone > {HARD_PHONE_SECONDS:.1f}s")
            continue
        spoken = _duration_seconds(words)
        if any(value > HARD_SYLLABLE_SECONDS for value in spoken):
            drop(sample_id, f"âm tiết > {HARD_SYLLABLE_SECONDS:.0f}s")
            continue
        frames_total = 1 + load_wav(str(row["audio_path"]), sample_rate).numel() // hop_length
        result = compute_durations(
            words, frames_total=frames_total, sample_rate=sample_rate, hop_length=hop_length,
            syllable_boundaries=[s.get("boundaries", {}) for s in row["syllables"]],
        )
        if result.get("dropped"):
            drop(sample_id, str(result["dropped"]))
            continue
        durations = result["durations"]
        if len(durations) != len(row["syllables"]) or sum(durations) != frames_total:
            drop(sample_id, "assert durations lệch")
            continue
        new_row = {**row, "durations": durations}
        if result["pauses"]:
            new_row["pauses"] = result["pauses"]
            stats["pauses"] += len(result["pauses"])
        kept.append(new_row)
        stats["one_frame"] += sum(1 for d in durations if d == 1)
        stats["syllables"] += len(durations)
        durations_all.extend(d * hop_length / sample_rate * 1000.0 for d in durations)

    durations_all.sort()
    if durations_all:
        stats["median_ms"] = round(durations_all[len(durations_all) // 2], 1)
        stats["p95_ms"] = round(durations_all[int(len(durations_all) * 0.95)], 1)
    stats["kept"] = len(kept)
    stats["dropped"] = {reason: len(items) for reason, items in sorted(dropped.items())}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    """Chuyển danh sách manifest; in thống kê + ghi report JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrid-dir", required=True, help="Thư mục TextGrid của MFA align")
    parser.add_argument("--manifest", action="append", required=True, help="Manifest JSONL (lặp lại được)")
    parser.add_argument("--out-dir", required=True, help="Thư mục ghi <tên>_dur.jsonl")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--report", default=None, help="File JSON report tổng (mặc định <out-dir>/mfa_durations_report.json)")
    parser.add_argument("--mfa-db", default=None, help="DB MFA (sau align) để lọc 10%% score thấp nhất (C4)")
    parser.add_argument("--score-drop-fraction", type=float, default=0.10)
    args = parser.parse_args()

    drop_score_ids: set[str] = set()
    if args.mfa_db:
        scores = load_mfa_scores(args.mfa_db)
        drop_score_ids = score_drop_ids(scores, args.score_drop_fraction)
        print(f"Score C4: {len(scores)} clip có điểm · loại {len(drop_score_ids)} "
              f"({args.score_drop_fraction:.0%} thấp nhất)")

    report: dict[str, Any] = {}
    for manifest in args.manifest:
        name = Path(manifest).stem
        out_path = Path(args.out_dir) / f"{name}_dur.jsonl"
        stats = convert_manifest(manifest, args.textgrid_dir, out_path,
                                 sample_rate=args.sample_rate, hop_length=args.hop_length,
                                 drop_score_ids=drop_score_ids or None)
        report[name] = stats
        print(f"{name}: giữ {stats['kept']}/{stats['clips']} clip · 1-frame {stats['one_frame']} "
              f"· median {stats['median_ms']} ms · p95 {stats['p95_ms']} ms · loại {stats['dropped']}")
    report_path = Path(args.report) if args.report else Path(args.out_dir) / "mfa_durations_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

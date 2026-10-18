"""Lọc manifest theo danh sách clip bị loại (QC chất lượng) — sinh bản `_filt`.

- Khoá clip chuẩn hoá = `Path(audio_path).stem` cho MỌI row (manifest thiếu `parent`
  vẫn khớp được với danh sách loại dạng id hoặc dạng đường dẫn).
- Ghi manifest mới cạnh bản gốc: `<tên>.jsonl` → `<tên>_filt.jsonl` (KHÔNG sửa bản gốc).
- Báo cáo: số row trước/sau, delta âm tiết, thanh, coda tắc — cảnh báo nếu ô hiếm đổi mạnh.

Chạy:
    python tools/filter_manifests.py --drop-list artifacts/clip_qc/drop_list.json \
        data/codelung/train_clean_v2_dur_seg_slot4.jsonl ... \
        --report artifacts/clip_qc/filter_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def clip_id(row: dict) -> str:
    """Khoá clip thống nhất: đuôi tệp của `audio_path` (bỏ thư mục và .wav)."""
    return Path(str(row["audio_path"])).stem


def normalize_drop(drop_keys: list[str]) -> set[str]:
    """Chuẩn hoá danh sách loại: nhận cả `id` trần lẫn đường dẫn tệp."""
    return {Path(key).stem for key in drop_keys}


def filter_manifest(manifest: str | Path, drop: set[str]) -> tuple[list[dict], dict]:
    """Đọc manifest, bỏ row có clip nằm trong `drop`; trả (rows mới, thống kê)."""
    kept: list[dict] = []
    stats = {"rows_before": 0, "rows_after": 0, "clips_dropped": 0, "syllables": 0}
    clips_dropped: set[str] = set()
    tones = Counter()
    stop_coda = 0
    with Path(manifest).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows_before"] += 1
            cid = clip_id(row)
            if cid in drop:
                clips_dropped.add(cid)
                continue
            kept.append(row)
            stats["rows_after"] += 1
            for syllable in row.get("syllables") or []:
                stats["syllables"] += 1
                tones[syllable["abstract"]["tone"]] += 1
                if syllable["abstract"]["coda"] in ("P", "T", "C", "CH"):
                    stop_coda += 1
    stats["clips_dropped"] = len(clips_dropped)
    stats["tones"] = dict(sorted(tones.items()))
    stats["stop_coda"] = stop_coda
    return kept, stats


def main() -> None:
    """Đọc tham số, lọc từng manifest, ghi `_filt`, in và lưu báo cáo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--drop-list", required=True, help="JSON danh sách clip loại")
    parser.add_argument("--report", help="Ghi báo cáo JSON")
    args = parser.parse_args()

    payload = json.loads(Path(args.drop_list).read_text(encoding="utf-8"))
    drop = normalize_drop(payload["drop"] if isinstance(payload, dict) else payload)
    report: dict = {"drop_keys": sorted(drop), "manifests": {}}

    for manifest in args.manifests:
        path = Path(manifest)
        kept, stats = filter_manifest(path, drop)
        out = path.with_name(path.stem + "_filt.jsonl")
        with out.open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in kept)
        before = stats["rows_before"] - stats["rows_after"]
        report["manifests"][str(path)] = {**stats, "rows_dropped": before, "output": str(out)}
        print(f"{path.name}: {stats['rows_before']} → {stats['rows_after']} row "
              f"(−{before}, {stats['clips_dropped']} clip) → {out.name}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Báo cáo: {args.report}")


if __name__ == "__main__":
    main()

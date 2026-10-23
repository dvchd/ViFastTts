"""v18.37 — thống kê độc lập train/val/holdout theo video / clip cha / segment.

Mục đích: giải quyết mâu thuẫn "val∩holdout = 0" với "9 video holdout là tập con
của val". Câu trả lời đúng phụ thuộc **mức**:
- video: val∩holdout = 9 (holdout dùng lại video của val);
- clip cha: val∩holdout = 0;
- segment: val∩holdout = 0.
Vì vậy holdout là **unseen-clip trong video đã thấy ở val**, không phải unseen-video.

Ghi `artifacts/bench/v1837_split_stats.json`; không sửa dữ liệu.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vifasttts.data.split import video_id_of_row

SPLITS = {
    "train": "data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl",
    "val": "data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl",
    "holdout": "data/codelung/v32/holdout_clean_v2_dur_seg_slot4.jsonl",
}


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    rows = {name: load(path) for name, path in SPLITS.items()}
    sets = {}
    for name, subset in rows.items():
        sets[name] = {
            "rows": len(subset),
            "videos": {video_id_of_row(row) for row in subset},
            "parents": {str(row["parent"]) for row in subset if row.get("parent")},
            "segments": {str(row["id"]) for row in subset},
            "hours": round(sum(int(row["frame_range"][1]) - int(row["frame_range"][0])
                               for row in subset) * 256 / 24000 / 3600, 3),
        }

    def inter(a: str, b: str, level: str) -> dict:
        left, right = sets[a][level], sets[b][level]
        return {"n": len(left & right), "examples": sorted(left & right)[:8]}

    pairs = [("train", "val"), ("train", "holdout"), ("val", "holdout")]
    result = {
        "splits": {name: {key: (sorted(value) if isinstance(value, set) else value)
                          for key, value in data.items()}
                   for name, data in sets.items()},
        "intersections": {
            f"{a}__{b}": {level: inter(a, b, level) for level in ("videos", "parents", "segments")}
            for a, b in pairs
        },
        "explanation": {
            "val_holdout_video_overlap": len(sets["val"]["videos"] & sets["holdout"]["videos"]),
            "val_holdout_parent_overlap": len(sets["val"]["parents"] & sets["holdout"]["parents"]),
            "val_holdout_segment_overlap": len(sets["val"]["segments"] & sets["holdout"]["segments"]),
            "holdout_is_subset_of_val_videos": sets["holdout"]["videos"] <= sets["val"]["videos"],
            "holdout_is_unseen_video": len(sets["holdout"]["videos"] & sets["train"]["videos"]) == 0
            and len(sets["holdout"]["videos"] & sets["val"]["videos"]) == 0,
            "conclusion": (
                "holdout KHÔNG phải unseen-video: 9/12 video đã có trong val; "
                "holdout là unseen-clip (parent/segment = 0 giao) trong video đã thấy ở val. "
                "Train vẫn tách video hoàn toàn khỏi val/holdout."
            ),
        },
    }
    out = Path("artifacts/bench/v1837_split_stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("intersections", "explanation")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

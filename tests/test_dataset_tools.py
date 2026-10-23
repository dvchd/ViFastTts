import json
from pathlib import Path

import pytest

from vifasttts.data.report import dataset_summary, format_report
from vifasttts.data.split import (
    pick_val_videos,
    split_rows,
    video_id_of,
    video_id_of_row,
    write_split,
)


def test_video_id_of():
    assert video_id_of("lRojc8KDJ6k_0001") == "lRojc8KDJ6k"
    assert video_id_of("abc") == "abc"
    assert video_id_of("a_b_0002") == "a_b"


def test_video_id_of_row_giu_video_co_gach_duoi():
    """Video id có `_` (4ITcvEa_Acc) không bị cắt; segment id không làm sai."""
    assert video_id_of_row({"parent": "4ITcvEa_Acc_0001",
                            "id": "4ITcvEa_Acc_0001__s00"}) == "4ITcvEa_Acc"
    assert video_id_of_row({"parent": "EM_ObHWvd94_0007",
                            "id": "EM_ObHWvd94_0007__s02"}) == "EM_ObHWvd94"
    assert video_id_of_row({"id": "a_b_0002"}) == "a_b"
    assert video_id_of_row({"id": "a_b_0002__s01"}) == "a_b"


def test_pick_val_videos_exact_and_deterministic():
    videos = [f"vid{i}" for i in range(10)]
    picked = pick_val_videos(videos, val_videos=3, seed=7)
    assert len(picked) == 3
    assert set(picked) <= set(videos)
    assert pick_val_videos(videos, val_videos=3, seed=7) == picked
    assert pick_val_videos(videos, val_videos=3, seed=8) != picked


def test_pick_val_videos_ratio_and_small_data():
    videos = [f"vid{i}" for i in range(20)]
    picked = pick_val_videos(videos, val_ratio=0.1, seed=1)
    assert len(picked) == 2
    assert pick_val_videos(["only_one"], val_videos=1) == []
    # Yêu cầu val nhiều hơn số video cũng chỉ lấy n-1 để còn ít nhất 1 video train.
    almost_all = pick_val_videos(videos, val_videos=99)
    assert len(almost_all) == len(videos) - 1


def test_split_rows_no_leakage():
    rows = [
        {"id": "v1_0001"}, {"id": "v1_0002"}, {"id": "v2_0001"},
        {"id": "v3_0001"}, {"id": "v4_0001"},
    ]
    train, val, val_videos = split_rows(rows, val_videos=2, seed=5)
    train_videos = {video_id_of(str(row["id"])) for row in train}
    val_set = {video_id_of(str(row["id"])) for row in val}
    assert val_set == set(val_videos)
    assert train_videos.isdisjoint(val_set)
    assert len(train) + len(val) == len(rows)


def test_write_split_creates_files(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [{"id": f"{video}_{index:04d}", "text": "chào"} for video in ("vA", "vB", "vC") for index in (1, 2)]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    clips = []
    for row in rows:
        clips.append({"id": row["id"], "duration": 2.0})
    (tmp_path / "clips.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in clips), encoding="utf-8"
    )

    stats = write_split(manifest, tmp_path, val_videos=1, seed=3)
    assert stats["train"] + stats["val"] == 6
    assert len(stats["val_videos"]) == 1
    train_rows = (tmp_path / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    val_rows = (tmp_path / "val.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train_rows) + len(val_rows) == 6
    assert stats["train_hours"] + stats["val_hours"] == pytest.approx(6 * 2.0 / 3600.0)


def test_dataset_summary_and_format(tmp_path: Path):
    clips = [
        {"id": "vA_0001", "video": "vA", "duration": 5.0, "fallback_ratio": 0.0},
        {"id": "vA_0002", "video": "vA", "duration": 21.0, "fallback_ratio": 0.3},
        {"id": "vB_0001", "video": "vB", "duration": 10.0, "fallback_ratio": 0.1},
    ]
    (tmp_path / "clips.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in clips), encoding="utf-8"
    )
    (tmp_path / "stats.json").write_text(
        json.dumps({"license": "CC BY 4.0", "oov_top": [["zibble", 2]], "manifest_issues": []}),
        encoding="utf-8",
    )
    summary = dataset_summary(tmp_path)
    assert summary["videos"] == 2 and summary["clips"] == 3
    assert summary["hours"] == 36.0 / 3600.0
    assert summary["zero_fallback_clips"] == 1
    assert summary["per_video"]["vA"]["high_fallback"] == 1
    assert {item["why"] for item in summary["flagged"]} == {"fallback cao", "độ dài bất thường"}
    text = format_report(summary)
    assert "vA" in text and "zibble" in text and "CC BY 4.0" in text

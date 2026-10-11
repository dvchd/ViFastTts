import csv
import json
from pathlib import Path

from vifasttts.data.manifest import build_manifest, validate_manifest


def test_build_manifest_skips_bad_rows(tmp_path: Path):
    meta = tmp_path / "meta.csv"
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    out = tmp_path / "out.jsonl"
    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["s1", "Hôm nay trời đẹp."])
        w.writerow(["badrow"])
        w.writerow(["s2", "Chào bạn!"])
    build_manifest(str(meta), str(wav_dir), str(out))
    lines = [json.loads(x) for x in Path(out).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 2
    assert lines[0]["id"] == "s1"
    assert len(lines[0]["syllables"]) == 4


def test_build_manifest_english_no_crash(tmp_path: Path):
    meta = tmp_path / "meta.csv"
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    out = tmp_path / "out.jsonl"
    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["s1", "Hello world"])
    build_manifest(str(meta), str(wav_dir), str(out))
    lines = [json.loads(x) for x in Path(out).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 1
    assert len(lines[0]["syllables"]) == 2


def test_validate_manifest_reports_missing_wav(tmp_path: Path):
    meta = tmp_path / "meta.csv"
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    out = tmp_path / "out.jsonl"
    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["s1", "Chào bạn"])
        w.writerow(["s2", "Hôm nay"])
    build_manifest(str(meta), str(wav_dir), str(out))
    (wav_dir / "s2.wav").write_bytes(b"RIFF")  # chỉ s1 thiếu

    report = validate_manifest(str(out), str(wav_dir))
    assert report["total"] == 2
    assert [(issue["id"], issue["issue"]) for issue in report["issues"]] == [("s1", "missing_wav")]


def test_validate_manifest_uses_audio_path(tmp_path: Path):
    meta = tmp_path / "meta.csv"
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    out = tmp_path / "out.jsonl"
    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["s1", "Chào bạn"])
    build_manifest(str(meta), str(wav_dir), str(out))
    (wav_dir / "s1.wav").write_bytes(b"RIFF")
    report = validate_manifest(str(out))
    assert report["issues"] == []

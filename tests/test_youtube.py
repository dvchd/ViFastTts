import csv
import json
import re
import sys
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vifasttts.cli import app
from vifasttts.data.subtitles import load_subtitles
from vifasttts.data.youtube import (
    FetchOptions,
    _prepare_dataset_text,
    fetch_dataset,
    read_wav_mono,
)
from vifasttts.frontend.dialect import Dialect
from vifasttts.frontend.normalizer import NormalizerOptions
from vifasttts.frontend.syllable import safe_parse_syllable

runner = CliRunner()

FAKE_YTDLP = textwrap.dedent(
    """
    import json, math, os, sys, wave
    from pathlib import Path

    args = sys.argv[1:]

    def opt(name, default=None):
        if name in args:
            return args[args.index(name) + 1]
        return default

    if "-J" in args:
        print(os.environ["FAKE_ENTRIES"])
        sys.exit(0)

    url = args[-1]
    vid = url.split("v=")[-1].split("&")[0]
    fail = os.environ.get("FAKE_FAIL_IDS", "").split(",")
    if vid in fail:
        print("lỗi giả lập khi tải", file=sys.stderr)
        sys.exit(3)

    out = (opt("-o") or "").replace("%(id)s", vid).replace("%(ext)s", "wav")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    if "--write-subs" in args:
        once = os.environ.get("FAKE_429_ONCE", "")
        marker = Path(__file__).with_suffix(".429")
        if vid == once and not marker.exists():
            marker.write_text("1", encoding="utf-8")
            print("ERROR: HTTP Error 429: Too Many Requests", file=sys.stderr)
            sys.exit(1)
        subs = json.loads(os.environ.get("FAKE_SUBS", "{}"))
        if vid in subs:
            Path(out + ".vi.json3").write_text(subs[vid], encoding="utf-8")
        sys.exit(0)

    seconds = float(os.environ.get("FAKE_SECONDS", "8"))
    amplitude = float(os.environ.get("FAKE_AMPLITUDE", "8000"))
    rate = 24000
    frames = bytearray()
    for i in range(int(seconds * rate)):
        value = int(amplitude * math.sin(2 * math.pi * 440 * i / rate))
        frames += int(value).to_bytes(2, "little", signed=True)
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    sys.exit(0)
    """
).strip()


def _json3(events: list[tuple[float, float, str]]) -> str:
    return json.dumps({
        "events": [
            {"tStartMs": int(start * 1000), "dDurationMs": int(duration * 1000),
             "segs": [{"utf8": text}]}
            for start, duration, text in events
        ]
    })


ENTRIES = [
    {"id": "vidA", "title": "Video A", "duration": 8.0,
     "url": "https://www.youtube.com/watch?v=vidA"},
    {"id": "vidB", "title": "Video B", "duration": 9.0,
     "url": "https://www.youtube.com/watch?v=vidB"},
    {"id": "vidC", "title": "Video C", "duration": 7.0,
     "url": "https://www.youtube.com/watch?v=vidC"},
]
SUBS = {
    "vidA": _json3([
        (0.5, 1.0, "xin chào các bạn"),
        (1.7, 1.0, "hôm nay trời đẹp"),
        (4.0, 1.5, "chúng ta cùng học lập trình"),
    ]),
    # Trùng nguyên văn text của clip đã gộp từ vidA → phải bị lọc trùng.
    "vidC": _json3([(0.5, 1.9, "xin chào các bạn hôm nay trời đẹp")]),
}


@pytest.fixture()
def fake_ytdlp(tmp_path: Path) -> Path:
    path = tmp_path / "fake_ytdlp.py"
    path.write_text(FAKE_YTDLP, encoding="utf-8")
    return path


def _options(tmp_path: Path, fake_ytdlp: Path, **overrides) -> FetchOptions:
    defaults: dict = {
        "channel": "https://www.youtube.com/channel/UCtest/videos",
        "out": tmp_path / "dataset",
        "ytdlp": [sys.executable, str(fake_ytdlp)],
        "ffmpeg": "ffmpeg-giả",
        "sleep": 0.0,
        "retry_delay": 0.0,
    }
    defaults.update(overrides)
    return FetchOptions(**defaults)


def _set_fake_env(monkeypatch, entries=ENTRIES, subs=SUBS, seconds=8.0, amplitude=8000.0, fail=""):
    monkeypatch.setenv("FAKE_ENTRIES", json.dumps({"entries": entries}))
    monkeypatch.setenv("FAKE_SUBS", json.dumps(subs))
    monkeypatch.setenv("FAKE_SECONDS", str(seconds))
    monkeypatch.setenv("FAKE_AMPLITUDE", str(amplitude))
    monkeypatch.setenv("FAKE_FAIL_IDS", fail)


def test_fetch_dataset_end_to_end(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    _set_fake_env(monkeypatch)
    options = _options(tmp_path, fake_ytdlp)
    stats = fetch_dataset(options, log=lambda _: None)

    assert stats["videos_listed"] == 3
    assert stats["videos_processed"] == 2
    assert stats["videos_no_subs"] == 1
    assert stats["clips"] == 2
    assert stats["skipped_duplicate"] == 1
    assert stats["videos_error"] == 0

    records = [json.loads(line) for line in (options.out / "clips.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert sorted(record["id"] for record in records) == ["vidA_0001", "vidA_0002"]
    first, second = sorted(records, key=lambda r: r["start"])
    assert first["text"] == "xin chào các bạn hôm nay trời đẹp"
    assert second["text"] == "chúng ta cùng học lập trình"
    for record in records:
        assert options.min_duration <= record["duration"] <= options.max_duration

    clip = options.out / "audio" / "vidA_0001.wav"
    samples, rate = read_wav_mono(clip)
    assert rate == 24000
    assert samples.size > 0 and float(abs(samples).max()) > 0.01

    metadata = list(csv.reader((options.out / "metadata.csv").open(encoding="utf-8"), delimiter="|"))
    assert len(metadata) == 2 and metadata[0][0] == "vidA_0001"

    manifest = [json.loads(line) for line in (options.out / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(manifest) == 2
    assert all(row["syllables"] for row in manifest)
    assert all(Path(row["audio_path"]).exists() for row in manifest)

    cues = load_subtitles(options.out / "subs" / "vidA.srt")
    assert [cue.text for cue in cues] == [
        "xin chào các bạn hôm nay trời đẹp",
        "chúng ta cùng học lập trình",
    ]
    assert not (options.out / "subs" / "vidB.srt").exists()

    card = (options.out / "dataset_card.md").read_text(encoding="utf-8")
    assert "CC BY 4.0" in card and "vidA" in card and "Tái tạo" in card
    assert (options.out / "LICENSE-DATASET.txt").exists()
    stats_json = json.loads((options.out / "stats.json").read_text(encoding="utf-8"))
    assert stats_json["clips"] == 2 and stats_json["coverage"]["syllables"] > 0
    assert stats_json["manifest_issues"] == []


def test_fetch_dataset_resume_is_idempotent(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    _set_fake_env(monkeypatch)
    options = _options(tmp_path, fake_ytdlp)
    fetch_dataset(options, log=lambda _: None)
    stats = fetch_dataset(options, log=lambda _: None)
    assert stats["videos_skipped_done"] == 2
    assert stats["clips"] == 2
    lines = (options.out / "clips.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_fetch_dataset_dry_run(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    _set_fake_env(monkeypatch)
    options = _options(tmp_path, fake_ytdlp, dry_run=True)
    stats = fetch_dataset(options, log=lambda _: None)
    assert stats["videos_listed"] == 3
    assert not (options.out / "manifest.jsonl").exists()
    assert not (options.out / "audio").exists()


def test_fetch_dataset_filters_silence(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    _set_fake_env(monkeypatch, entries=ENTRIES[:1], subs={"vidA": SUBS["vidA"]}, amplitude=0.0)
    options = _options(tmp_path, fake_ytdlp)
    stats = fetch_dataset(options, log=lambda _: None)
    assert stats["clips"] == 0
    assert stats["skipped_silence"] >= 1


def test_fetch_dataset_survives_single_video_error(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    entry = [{"id": "vidE", "title": "Video lỗi", "duration": 8.0,
              "url": "https://www.youtube.com/watch?v=vidE"}]
    _set_fake_env(monkeypatch, entries=entry, subs={"vidE": SUBS["vidA"]}, fail="vidE")
    options = _options(tmp_path, fake_ytdlp)
    stats = fetch_dataset(options, log=lambda _: None)
    assert stats["videos_error"] == 1
    assert stats["clips"] == 0
    assert (options.out / "manifest.jsonl").exists()


def test_prepare_dataset_text_respells_codes():
    options = NormalizerOptions(dialect=Dialect.NORTH)
    assert _prepare_dataset_text("in 3D", options, True) == "in ba dê"
    assert _prepare_dataset_text("xem Chương2 nhé", options, True) == "xem Chương hai nhé"
    assert _prepare_dataset_text("file MP3", options, True) == "phai em pê ba"
    assert _prepare_dataset_text("bản v2", options, True) == "bản vê hai"
    assert _prepare_dataset_text("nhận 500k", options, True) == "nhận năm trăm ca"
    assert _prepare_dataset_text("dùng n8n", options, True) == "dùng en tám en"
    assert _prepare_dataset_text("bản 1.13.1 nhé", options, True) == "bản một mười ba một nhé"
    assert _prepare_dataset_text("đọc 3.14 nhé", options, True) == "đọc ba phẩy một bốn nhé"
    # Tắt respell: giữ hành vi thận trọng của frontend với mã phiên bản.
    assert _prepare_dataset_text("file MP3", options, False) == "phai MP3"
    assert _prepare_dataset_text("bản v2", options, False) == "bản v2"


def test_fetch_dataset_respells_codes_end_to_end(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    entry = [{"id": "vidCode", "title": "Mã", "duration": 6.0,
              "url": "https://www.youtube.com/watch?v=vidCode"}]
    _set_fake_env(monkeypatch, entries=entry,
                  subs={"vidCode": _json3([(0.5, 2.0, "nhận file v2 và MP3 nhé")])})
    options = _options(tmp_path, fake_ytdlp)
    fetch_dataset(options, log=lambda _: None)
    records = [json.loads(line) for line in (options.out / "clips.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert "phai" in records[0]["text"]
    assert "vê hai" in records[0]["text"]
    assert "em pê ba" in records[0]["text"]


def test_fetch_dataset_retries_transient_error(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    _set_fake_env(monkeypatch, entries=ENTRIES[:1], subs={"vidA": SUBS["vidA"]})
    monkeypatch.setenv("FAKE_429_ONCE", "vidA")  # lần đầu trả 429, lần sau thành công
    options = _options(tmp_path, fake_ytdlp)
    stats = fetch_dataset(options, log=lambda _: None)
    assert stats["videos_error"] == 0
    assert stats["clips"] == 2


def test_overrides_tech_config_is_parseable():
    payload = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "overrides_tech_vi.json")
        .read_text(encoding="utf-8")
    )
    assert payload, "file overrides mẫu không được rỗng"
    for token, reading in payload.items():
        assert token.strip() and reading.strip()
        for word in re.split(r"\s+", reading):
            assert safe_parse_syllable(word) is not None, f"{word!r} trong {reading!r} ({token!r})"


def test_fetch_dataset_reports_oov(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    entry = [{"id": "vidOov", "title": "Có từ lạ", "duration": 6.0,
              "url": "https://www.youtube.com/watch?v=vidOov"}]
    sentence = "xin chào các bạn hôm nay zibble florp rất hay"
    _set_fake_env(monkeypatch, entries=entry, subs={"vidOov": _json3([(0.5, 2.0, sentence)])})
    options = _options(tmp_path, fake_ytdlp)
    stats = fetch_dataset(options, log=lambda _: None)
    assert stats["coverage"]["fallback"] == 2
    tokens = dict(stats["oov_top"])
    assert tokens == {"zibble": 1, "florp": 1}
    report = json.loads((options.out / "oov_report.json").read_text(encoding="utf-8"))
    assert {"token": "zibble", "count": 1} in report["tokens"]
    card = (options.out / "dataset_card.md").read_text(encoding="utf-8")
    assert "Từ lạ (OOV)" in card


def test_fetch_dataset_overrides_reduce_oov(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    entry = [{"id": "vidOov", "title": "Có từ lạ", "duration": 6.0,
              "url": "https://www.youtube.com/watch?v=vidOov"}]
    sentence = "xin chào các bạn hôm nay zibble florp rất hay"
    _set_fake_env(monkeypatch, entries=entry, subs={"vidOov": _json3([(0.5, 2.0, sentence)])})
    options = _options(tmp_path, fake_ytdlp,
                       overrides={"zibble": "xíp bồ", "florp": "phờ lốp"},
                       overrides_source="overrides.json")
    stats = fetch_dataset(options, log=lambda _: None)
    assert stats["coverage"]["fallback"] == 0
    assert stats["oov_top"] == []
    records = [json.loads(line) for line in (options.out / "clips.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert records[0]["text"] == "xin chào các bạn hôm nay xíp bồ phờ lốp rất hay"
    card = (options.out / "dataset_card.md").read_text(encoding="utf-8")
    assert "--overrides overrides.json" in card


def test_fetch_dataset_demo_picks_short_video(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    entries = [
        {"id": "vidLong", "title": "Dài", "duration": 3600.0,
         "url": "https://www.youtube.com/watch?v=vidLong"},
        {"id": "vidA", "title": "Ngắn", "duration": 500.0,
         "url": "https://www.youtube.com/watch?v=vidA"},
    ]
    _set_fake_env(monkeypatch, entries=entries)
    options = _options(tmp_path, fake_ytdlp, demo=True, min_caption_density=0.0)
    stats = fetch_dataset(options, log=lambda _: None)
    assert stats["videos_listed"] == 2  # demo đưa cả danh sách vào candidates
    records = [json.loads(line) for line in (options.out / "clips.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert records and all(record["video"] == "vidA" for record in records)


def test_fetch_dataset_skips_sparse_captions(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    entry = [{"id": "vidSparse", "title": "Nhiều nhạc", "duration": 300.0,
              "url": "https://www.youtube.com/watch?v=vidSparse"}]
    sparse = _json3([(0.5, 1.0, "chào bạn")])  # 8 ký tự / 300 giây = quá thưa
    _set_fake_env(monkeypatch, entries=entry, subs={"vidSparse": sparse})
    options = _options(tmp_path, fake_ytdlp)
    stats = fetch_dataset(options, log=lambda _: None)
    assert stats["videos_sparse_subs"] == 1
    assert stats["videos_processed"] == 0
    assert stats["clips"] == 0
    assert not (options.out / "raw").exists()  # chưa tải audio vì bị lọc trước


def test_demo_tries_candidates_until_good(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    entries = [
        {"id": "vidSparse", "title": "Ngắn nhưng nhiều nhạc", "duration": 10.0,
         "url": "https://www.youtube.com/watch?v=vidSparse"},
        {"id": "vidGood", "title": "Nói đều", "duration": 20.0,
         "url": "https://www.youtube.com/watch?v=vidGood"},
    ]
    good = _json3([
        (0.3, 0.8, "xin chào mọi người hôm nay"),
        (1.2, 0.8, "chúng ta cùng học lập trình"),
        (2.1, 0.8, "đây là ví dụ dễ hiểu"),
        (3.0, 0.8, "cảm ơn các bạn đã xem"),
    ])
    _set_fake_env(monkeypatch, entries=entries,
                  subs={"vidSparse": _json3([(0.5, 1.0, "chào")]), "vidGood": good})
    options = _options(tmp_path, fake_ytdlp, demo=True)
    stats = fetch_dataset(options, log=lambda _: None)
    assert stats["videos_sparse_subs"] == 1
    assert stats["videos_processed"] == 1
    records = [json.loads(line) for line in (options.out / "clips.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert records and all(record["video"] == "vidGood" for record in records)


def test_youtube_fetch_cli(tmp_path: Path, fake_ytdlp: Path, monkeypatch):
    _set_fake_env(monkeypatch)
    out = tmp_path / "cli-dataset"
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"code": "cốt"}), encoding="utf-8")
    result = runner.invoke(app, [
        "youtube-fetch", "https://www.youtube.com/channel/UCtest/videos",
        "--out", str(out), "--demo", "--sleep", "0", "--overrides", str(overrides),
        "--ytdlp-bin", f'{sys.executable} "{fake_ytdlp}"', "--ffmpeg-bin", "ffmpeg-giả",
    ])
    assert result.exit_code == 0, result.stdout
    assert "Tổng kết" in result.stdout
    assert (out / "manifest.jsonl").exists()
    assert (out / "dataset_card.md").exists()

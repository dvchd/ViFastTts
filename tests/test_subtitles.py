import json

import pytest

from vifasttts.data.subtitles import (
    Cue,
    clean_caption,
    format_srt,
    load_subtitles,
    merge_cues,
    normalize_cues,
    parse_json3,
    parse_srt,
    parse_timestamp,
    parse_vtt,
)


def _json3(events: list[tuple[float, float, str]]) -> str:
    return json.dumps({
        "events": [
            {"tStartMs": int(start * 1000), "dDurationMs": int(duration * 1000),
             "segs": [{"utf8": text}]}
            for start, duration, text in events
        ]
    })


def test_parse_timestamp_variants():
    assert parse_timestamp("00:00:01,500") == pytest.approx(1.5)
    assert parse_timestamp("01:02.250") == pytest.approx(62.25)
    assert parse_timestamp("1:02:03.004") == pytest.approx(3723.004)
    assert parse_timestamp("00:00:02.240 align:start position:0%") == pytest.approx(2.24)
    assert parse_timestamp("không phải thời gian") == 0.0


def test_clean_caption_removes_noise_and_tags():
    assert clean_caption("<c>xin chào</c> <00:00:01.000>các bạn") == "xin chào các bạn"
    assert clean_caption("[Music] hôm nay trời đẹp [Applause]") == "hôm nay trời đẹp"
    assert clean_caption("[Âm nhạc]") == ""
    assert clean_caption("(music playing)") == ""
    assert clean_caption("♪ ♪") == ""
    assert clean_caption(">> lưu ý quan trọng") == "lưu ý quan trọng"
    assert clean_caption("giữ lại (phần 1) nhé") == "giữ lại (phần 1) nhé"
    assert clean_caption("&nbsp;dòng&nbsp;mới  ") == "dòng mới"
    assert clean_caption("{\\an8}karaoke") == "karaoke"


def test_parse_srt_basic_and_growth():
    text = (
        "1\n00:00:00,500 --> 00:00:01,500\nXin chào <i>các bạn</i>\n\n"
        "2\n00:00:01,200 --> 00:00:02,000\nXin chào các bạn ơi\n\n"
        "3\n\n00:00:03,000 --> 00:00:04,000\nHôm nay trời đẹp\n"
    )
    cues = parse_srt(text)
    assert [cue.text for cue in cues] == ["Xin chào các bạn", "ơi", "Hôm nay trời đẹp"]
    assert cues[0].start == pytest.approx(0.5) and cues[0].end == pytest.approx(1.5)
    assert cues[1].start == pytest.approx(1.2)


def test_parse_srt_dedupes_repeats():
    text = (
        "1\n00:00:00,000 --> 00:00:01,000\nchào bạn\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nchào bạn\n"
    )
    cues = parse_srt(text)
    assert len(cues) == 1
    assert cues[0].end == pytest.approx(2.0)


def test_parse_vtt_skips_headers_notes_and_settings():
    text = (
        "WEBVTT\nKind: captions\nLanguage: vi\n\n"
        "NOTE đoạn ghi chú bị bỏ\n\n"
        "cue-1\n00:00:01.000 --> 00:00:02.000 align:start position:0%\nxin chào\n\n"
        "00:00:02.000 --> 00:00:03.000\ncác bạn\n"
    )
    cues = parse_vtt(text)
    assert [(cue.start, cue.text) for cue in cues] == [
        (pytest.approx(1.0), "xin chào"),
        (pytest.approx(2.0), "các bạn"),
    ]


def test_parse_json3_splits_rolling_captions():
    cues = parse_json3(_json3([
        (0.5, 1.0, "xin chào"),
        (1.2, 0.8, "xin chào các bạn"),
        (2.2, 0.6, "\n"),
        (2.5, 1.0, "chúng ta bắt đầu"),
    ]))
    assert [cue.text for cue in cues] == ["xin chào", "các bạn", "chúng ta bắt đầu"]
    assert cues[1].start == pytest.approx(1.2)


def test_parse_json3_closes_open_ends_with_next_start():
    cues = parse_json3(_json3([
        (1.0, 0.0, "một hai"),
        (2.0, 0.5, "ba bốn"),
    ]))
    assert cues[0].end == pytest.approx(2.0)


def test_normalize_cues_drops_nested_and_empty():
    cues = normalize_cues([
        Cue(0.0, 5.0, "xin chào các bạn ơi"),
        Cue(1.0, 2.0, "chào"),
        Cue(6.0, 7.0, "   "),
        Cue(7.0, 8.0, "hẹn gặp lại"),
    ])
    assert [cue.text for cue in cues] == ["xin chào các bạn ơi", "hẹn gặp lại"]


def test_merge_cues_gap_and_punctuation_break():
    cues = [
        Cue(0.0, 1.0, "câu thứ nhất"),
        Cue(1.2, 2.0, "vẫn cùng câu"),      # gap 0.2 → gộp
        Cue(3.0, 4.0, "câu mới."),           # gap 1.0 → tách
        Cue(4.2, 5.0, "nối tiếp"),           # hết câu + gap đủ lớn → vẫn tách
    ]
    merged = merge_cues(cues, max_gap=0.35)
    assert [cue.text for cue in merged] == ["câu thứ nhất vẫn cùng câu", "câu mới.", "nối tiếp"]


def test_merge_cues_stops_at_sentence_end_with_gap():
    cues = [Cue(0.0, 1.0, "hết câu."), Cue(1.3, 2.0, "câu khác")]
    merged = merge_cues(cues, max_gap=0.35, break_gap=0.18)
    assert len(merged) == 2


def test_merge_cues_respects_max_chars_and_duration():
    cues = [Cue(0.0, 1.0, "a" * 50), Cue(1.1, 2.0, "b" * 50)]
    assert len(merge_cues(cues, max_chars=60)) == 2
    long_cue = [Cue(0.0, 30.0, "một hai ba bốn năm sáu")]
    parts = merge_cues(long_cue, max_duration=10.0)
    assert len(parts) == 3
    assert parts[0].text == "một hai" and parts[0].end == pytest.approx(10.0)
    assert parts[-1].end == pytest.approx(30.0)


def test_merge_cues_merges_overlap():
    cues = [Cue(0.0, 2.0, "phần một"), Cue(1.5, 3.0, "phần hai")]
    merged = merge_cues(cues)
    assert len(merged) == 1
    assert merged[0].end == pytest.approx(3.0)


def test_format_srt_roundtrip():
    cues = [Cue(0.5, 1.5, "xin chào"), Cue(2.0, 3.25, "hẹn gặp lại")]
    text = format_srt(cues)
    assert "00:00:00,500 --> 00:00:01,500" in text
    parsed = parse_srt(text)
    assert [(cue.start, cue.end, cue.text) for cue in parsed] == [
        (pytest.approx(0.5), pytest.approx(1.5), "xin chào"),
        (pytest.approx(2.0), pytest.approx(3.25), "hẹn gặp lại"),
    ]


def test_load_subtitles_by_extension(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nchào\n", encoding="utf-8")
    vtt = tmp_path / "b.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nchào\n", encoding="utf-8")
    json3 = tmp_path / "c.json3"
    json3.write_text(_json3([(0.0, 1.0, "chào")]), encoding="utf-8")
    assert load_subtitles(srt)[0].text == "chào"
    assert load_subtitles(vtt)[0].text == "chào"
    assert load_subtitles(json3)[0].text == "chào"
    with pytest.raises(ValueError):
        load_subtitles(tmp_path / "d.txt")

"""Test `tools/align_quality_report.py`: metric chất lượng alignment từ TextGrid giả."""
from __future__ import annotations

from pathlib import Path

from tools.align_quality_report import clip_metrics


def _textgrid(path: Path, words: list[tuple[float, float, str]], phones: list[tuple[float, float, str]]) -> None:
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0 ",
        "xmax = 2.0 ",
        "tiers? <exists> ",
        "size = 2 ",
        "item []: ",
        "    item [1]:",
        '        class = "IntervalTier" ',
        '        name = "words" ',
        "        xmin = 0 ",
        "        xmax = 2.0 ",
        f"        intervals: size = {len(words)} ",
    ]
    for index, (xmin, xmax, text) in enumerate(words, start=1):
        lines += [f"        intervals [{index}]:", f"            xmin = {xmin} ",
                  f"            xmax = {xmax} ", f'            text = "{text}" ']
    lines += [
        "    item [2]:",
        '        class = "IntervalTier" ',
        '        name = "phones" ',
        "        xmin = 0 ",
        "        xmax = 2.0 ",
        f"        intervals: size = {len(phones)} ",
    ]
    for index, (xmin, xmax, text) in enumerate(phones, start=1):
        lines += [f"        intervals [{index}]:", f"            xmin = {xmin} ",
                  f"            xmax = {xmax} ", f'            text = "{text}" ']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_clip_metrics_floor_rate_va_long_phone(tmp_path: Path):
    grid = tmp_path / "s1.TextGrid"
    _textgrid(
        grid,
        words=[(0.0, 0.04, "chào"), (0.5, 0.8, "bạn")],
        phones=[(0.0, 0.04, "Och"), (0.04, 0.7, "NaHuyen"), (0.7, 0.8, "Cn"), (0.8, 1.0, "sil")],
    )
    metrics = clip_metrics(grid)
    assert metrics is not None
    assert metrics["words"] == 2
    assert metrics["floor_rate"] == 0.5               # 1/2 từ ≤ 55 ms
    assert metrics["coverage"] == round(0.34 / 0.8, 3)
    assert metrics["word_dur_median_s"] == round(0.17, 3)
    # phone hữu thanh > 0,5 s: NaHuyen (0,66 s) / 3 phone hữu thanh (sil bị bỏ)
    assert metrics["long_phone_rate"] == round(1 / 3, 4)


def test_clip_metrics_khong_word_tra_none(tmp_path: Path):
    grid = tmp_path / "empty.TextGrid"
    _textgrid(grid, words=[(0.0, 1.0, "")], phones=[])
    assert clip_metrics(grid) is None

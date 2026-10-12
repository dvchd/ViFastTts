"""Test `tools/mfa_to_durations.py`: TextGrid giả + ca biên (pause, làm tròn, loại clip)."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from tools.mfa_to_durations import compute_durations, convert_manifest, parse_long_textgrid

SR, HOP = 24000, 256


def _wav(path: Path, seconds: float = 1.0) -> None:
    time_axis = np.arange(int(seconds * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 200.0 * time_axis)
    pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes(pcm.tobytes())


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


def _row(tmp_path: Path, boundaries: list[dict]) -> dict:
    wav = tmp_path / "s1.wav"
    _wav(wav)
    return {
        "id": "s1", "audio_path": str(wav), "text": "chào bạn nhé",
        "syllables": [
            {"normalized": token, "boundaries": bounds}
            for token, bounds in zip(("chào", "bạn", "nhé"), boundaries)
        ],
    }


def _manifest(tmp_path: Path, row: dict) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_parse_long_textgrid():
    sample = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = 1.0 
tiers? <exists> 
size = 1 
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "words" 
        xmin = 0 
        xmax = 1.0 
        intervals: size = 2 
        intervals [1]:
            xmin = 0.0 
            xmax = 0.5 
            text = "chào" 
        intervals [2]:
            xmin = 0.5 
            xmax = 1.0 
            text = "" 
"""
    tmp = Path("/private/tmp/opencode/_test_textgrid.TextGrid")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(sample, encoding="utf-8")
    tiers = parse_long_textgrid(tmp)
    assert tiers["words"] == [(0.0, 0.5, "chào"), (0.5, 1.0, "")]
    tmp.unlink()


def test_compute_durations_round_trip():
    words = [(0.0, 0.3, "chào"), (0.35, 0.6, "bạn"), (0.8, 1.0, "nhé")]
    result = compute_durations(
        words, frames_total=94, sample_rate=SR, hop_length=HOP,
        syllable_boundaries=[{}, {"sentence_end": True}, {}],
    )
    assert result["dropped"] is None
    assert result["durations"] == [33, 42, 19] and sum(result["durations"]) == 94
    assert result["pauses"] == [{"after": 1, "seconds": 0.2}]


def test_convert_manifest_giu_pause_o_ranh_gioi_cau(tmp_path: Path):
    row = _row(tmp_path, [{}, {"sentence_end": True}, {}])
    manifest = _manifest(tmp_path, row)
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    _textgrid(grid_dir / "s1.TextGrid",
              [(0.0, 0.3, "chào"), (0.35, 0.6, "bạn"), (0.8, 1.0, "nhé")],
              [(0.0, 0.3, "o_ch"), (0.3, 0.35, "sil")])
    stats = convert_manifest(manifest, grid_dir, tmp_path / "out.jsonl")
    assert stats["kept"] == 1 and stats["pauses"] == 1
    kept = json.loads((tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert sum(kept["durations"]) == 94 and kept["pauses"] == [{"after": 1, "seconds": 0.2}]


def test_convert_manifest_loai_ngung_tho_giua_cau(tmp_path: Path):
    row = _row(tmp_path, [{}, {"sentence_end": False, "phrase_end": False}, {}])
    manifest = _manifest(tmp_path, row)
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    _textgrid(grid_dir / "s1.TextGrid",
              [(0.0, 0.3, "chào"), (0.35, 0.6, "bạn"), (0.8, 1.0, "nhé")],
              [(0.0, 0.3, "o_ch")])
    stats = convert_manifest(manifest, grid_dir, tmp_path / "out.jsonl")
    assert stats["kept"] == 0
    assert any("ngừng thở" in reason for reason in stats["dropped"])


def test_convert_manifest_loai_phone_qua_dai_va_thieu_grid(tmp_path: Path):
    row = _row(tmp_path, [{}, {}, {}])
    manifest = _manifest(tmp_path, row)
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    _textgrid(grid_dir / "s1.TextGrid",
              [(0.0, 0.3, "chào"), (0.35, 0.6, "bạn"), (0.8, 1.0, "nhé")],
              [(0.0, 0.7, "o_ch")])  # phone 0,7s > 0,5s
    stats = convert_manifest(manifest, grid_dir, tmp_path / "out.jsonl")
    assert stats["kept"] == 0 and any("phone" in reason for reason in stats["dropped"])

    grid_dir2 = tmp_path / "grid2"
    grid_dir2.mkdir()
    stats2 = convert_manifest(manifest, grid_dir2, tmp_path / "out2.jsonl")
    assert stats2["kept"] == 0 and "thiếu TextGrid" in stats2["dropped"]


def test_compute_durations_infeasible_thi_loai():
    words = [(i * 0.01, i * 0.01 + 0.005, f"w{i}") for i in range(200)]
    result = compute_durations(words, frames_total=94, sample_rate=SR, hop_length=HOP,
                               syllable_boundaries=[{}] * 200)
    assert result["dropped"] is not None


def test_load_mfa_scores_va_loc_10_phan_tram(tmp_path: Path):
    import sqlite3

    from tools.mfa_to_durations import load_mfa_scores, score_drop_ids

    db = tmp_path / "mfa.db"
    connection = sqlite3.connect(str(db))
    connection.executescript(
        "create table sound_file (file_id integer primary key, sound_file_path text);"
        "create table utterance (id integer primary key, file_id integer, num_frames integer,"
        " alignment_log_likelihood real);"
    )
    connection.execute("insert into sound_file values (1, '/x/s1.wav')")
    connection.execute("insert into sound_file values (2, '/x/s2.wav')")
    connection.execute("insert into sound_file values (3, '/x/s3.wav')")
    connection.execute("insert into utterance values (1, 1, 100, -1000.0)")  # -10.0/frame (tệ nhất)
    connection.execute("insert into utterance values (2, 2, 100, -200.0)")   # -2.0/frame
    connection.execute("insert into utterance values (3, 3, 100, -500.0)")   # -5.0/frame
    connection.commit()
    connection.close()

    scores = load_mfa_scores(db)
    assert scores == {"s1": -10.0, "s2": -2.0, "s3": -5.0}
    dropped = score_drop_ids(scores, 0.34)  # 1/3 thấp nhất → s1
    assert dropped == {"s1"}
    assert score_drop_ids({}, 0.1) == set()

import csv
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vifasttts.cli import app

runner = CliRunner()


def test_parse_cli():
    result = runner.invoke(app, ["parse", "Chào bạn"])
    assert result.exit_code == 0
    assert "Chào" in result.stdout and "bạn" in result.stdout


def test_parse_syllable_cli():
    result = runner.invoke(app, ["parse-syllable", "ngoài"])
    assert result.exit_code == 0
    assert "ngoài" in result.stdout


def test_build_and_validate_manifest_cli(tmp_path: Path):
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    with wave.open(str(wav_dir / "s1.wav"), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24000)
        fh.writeframes(b"\x00\x00" * 2400)
    metadata = tmp_path / "meta.csv"
    with metadata.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["s1", "Chào bạn"])
    manifest = tmp_path / "manifest.jsonl"

    result = runner.invoke(app, [
        "build-manifest", "--metadata", str(metadata),
        "--wav-dir", str(wav_dir), "--output", str(manifest),
    ])
    assert result.exit_code == 0 and manifest.exists()

    result = runner.invoke(app, ["validate-manifest", "--manifest", str(manifest), "--wav-dir", str(wav_dir)])
    assert result.exit_code == 0
    assert "0 vấn đề" in result.stdout


def test_check_profile_cli():
    result = runner.invoke(app, ["check-profile", "voan", "--profile", "strict"])
    assert result.exit_code == 0
    assert "labial-w" in result.stdout

    clean = runner.invoke(app, ["check-profile", "quốc", "--profile", "strict"])
    assert clean.exit_code == 0
    assert "0 cảnh báo" in clean.stdout


def test_build_lexicon_cli(tmp_path: Path):
    wordlist = tmp_path / "words.txt"
    wordlist.write_text("ba mẹ khoai\n", encoding="utf-8")
    output = tmp_path / "lexicon.json"
    result = runner.invoke(app, ["build-lexicon", "--wordlist", str(wordlist), "--output", str(output)])
    assert result.exit_code == 0
    assert output.exists() and "shape" in result.stdout


def test_cache_features_cli(tmp_path: Path):
    pytest.importorskip("torch")

    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    with wave.open(str(wav_dir / "s1.wav"), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24000)
        fh.writeframes(b"\x00\x00" * 4800)
    metadata = tmp_path / "meta.csv"
    with metadata.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["s1", "Chào"])
    manifest = tmp_path / "manifest.jsonl"
    runner.invoke(app, ["build-manifest", "--metadata", str(metadata),
                        "--wav-dir", str(wav_dir), "--output", str(manifest)])
    config = tmp_path / "tiny.yaml"
    config.write_text(
        "sample_rate: 24000\nmel:\n  n_fft: 256\n  win_length: 256\n"
        "  hop_length: 64\n  n_mels: 20\n  f_min: 0\n  f_max: 12000\n",
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    result = runner.invoke(app, ["cache-features", "--manifest", str(manifest),
                                 "--config", str(config), "--cache-dir", str(cache_dir)])
    assert result.exit_code == 0
    assert list(cache_dir.glob("*.pt"))

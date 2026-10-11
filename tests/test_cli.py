import csv
import wave
from pathlib import Path

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

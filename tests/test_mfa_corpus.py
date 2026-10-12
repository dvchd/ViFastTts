"""Test dựng corpus + lexicon MFA (`tools/mfa_build_corpus.py`)."""
from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np

from vifasttts.data.manifest import build_manifest


def _write_wav(path: Path, seconds: float = 0.3, sample_rate: int = 24000) -> None:
    time_axis = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 220.0 * time_axis)
    pcm = (np.clip(tone, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def _manifest(tmp_path: Path) -> Path:
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir(exist_ok=True)
    _write_wav(wav_dir / "s1.wav")
    metadata = tmp_path / "meta.csv"
    with metadata.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|")
        writer.writerow(["s1", "Chào bạn"])
    manifest = tmp_path / "manifest.jsonl"
    build_manifest(str(metadata), str(wav_dir), str(manifest))
    return manifest


def test_build_corpus(tmp_path: Path):
    from tools.mfa_build_corpus import build_corpus, phones_for_syllable

    manifest = _manifest(tmp_path)
    out = tmp_path / "mfa"
    stats = build_corpus([str(manifest)], out)
    assert stats["clips"] == 1 and stats["words"] == 2

    lab = (out / "s1.lab").read_text(encoding="utf-8").strip()
    assert lab == "chào bạn"
    assert (out / "s1.wav").exists()

    lexicon = dict(
        line.split("\t") for line in (out / "lexicon.txt").read_text(encoding="utf-8").splitlines() if line
    )
    assert lexicon["chào"] == "o_ch n_a_huyen c_w"
    assert lexicon["bạn"] == "o_b n_a_nang c_n"

    phones = set((out / "phones.txt").read_text(encoding="utf-8").split())
    assert {"o_ch", "n_a_huyen", "n_a_nang", "c_n"} <= phones

    index = json.loads((out / "syllable_index.json").read_text(encoding="utf-8"))
    assert index["bạn"]["components"]["tone"] == "NANG"
    assert phones_for_syllable({"onset": "NONE", "medial": "W", "nucleus": "A", "coda": "NONE", "tone": "SAC"}) == [
        "m_w", "n_a_sac",
    ]


def test_build_corpus_train_val_combined(tmp_path: Path):
    from tools.mfa_build_corpus import build_corpus

    manifest = _manifest(tmp_path)
    out = tmp_path / "mfa"
    stats = build_corpus([str(manifest), str(manifest)], out)
    assert stats["clips"] == 2  # hai manifest cùng file vẫn đếm clip; lexicon dedup theo từ
    assert stats["words"] == 2

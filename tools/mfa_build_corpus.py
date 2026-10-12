"""Dựng corpus + lexicon cho MFA từ manifest ViFastTts (REVISION_PLAN_V2 §5).

Sinh trong `--out`:
- `wavs/<id>.wav`   — symlink tới audio gốc (copy nếu không symlink được)
- `labs/<id>.lab`   — câu đã respell, token = trường `normalized` của từng âm tiết
- `lexicon.txt`     — `<âm tiết>\t<phone ...>` (gốc để `mfa train` bằng lexicon riêng)
- `phones.txt`      — danh sách phone (tiền tố vị trí: o_/m_/n_/c_)
- `syllable_index.json` — normalized → {components, phones} để map ngược TextGrid

Phone set: `o_<onset>`, `m_<medial>`, `c_<coda>`, `n_<nucleus>_<tone>` (thanh gắn liền
nguyên âm chính — không align thanh thành đoạn riêng). Tiền tố bắt buộc vì cùng ký hiệu
nhưng khác vị trí là khác âm học (NG đầu vs NG cuối, W đệm vs W cuối).

Chạy:
    python tools/mfa_build_corpus.py --manifest data/codelung/train.jsonl \
        --manifest data/codelung/val.jsonl --out data/codelung/mfa
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def phones_for_syllable(abstract: dict[str, str]) -> list[str]:
    """5 thành phần abstract → danh sách phone có tiền tố vị trí (thứ tự thời gian)."""
    phones: list[str] = []
    onset = str(abstract.get("onset", "NONE"))
    medial = str(abstract.get("medial", "NONE"))
    nucleus = str(abstract.get("nucleus", "NONE"))
    coda = str(abstract.get("coda", "NONE"))
    tone = str(abstract.get("tone", "NGANG"))
    if onset != "NONE":
        phones.append(f"o_{onset.lower()}")
    if medial != "NONE":
        phones.append(f"m_{medial.lower()}")
    phones.append(f"n_{nucleus.lower()}_{tone.lower()}")
    if coda != "NONE":
        phones.append(f"c_{coda.lower()}")
    return phones


def build_corpus(manifests: list[str], out: str | Path) -> dict[str, Any]:
    """Dựng corpus; trả về dict thống kê (clips, words, phones, fallback...)."""
    out_path = Path(out)
    wav_dir = out_path / "wavs"
    lab_dir = out_path / "labs"
    for directory in (wav_dir, lab_dir):
        directory.mkdir(parents=True, exist_ok=True)

    lexicon: dict[str, list[str]] = {}
    index: dict[str, dict[str, Any]] = {}
    fallback_samples: list[str] = []
    fallback_words: set[str] = set()
    clips = 0
    fallback = 0
    missing_audio = 0

    for manifest in manifests:
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            tokens = [str(syllable["normalized"]) for syllable in row["syllables"]]
            if not tokens:
                continue
            (lab_dir / f"{sample_id}.lab").write_text(" ".join(tokens) + "\n", encoding="utf-8")
            audio = Path(str(row["audio_path"])).resolve()
            target = wav_dir / f"{sample_id}.wav"
            if not target.exists():
                if not audio.exists():
                    missing_audio += 1
                    continue
                try:
                    os.symlink(audio, target)
                except OSError:
                    shutil.copy2(audio, target)
            has_fallback = False
            for syllable in row["syllables"]:
                token = str(syllable["normalized"])
                abstract = syllable.get("abstract", {})
                phones = phones_for_syllable(abstract)
                if token not in lexicon:
                    lexicon[token] = phones
                    index[token] = {
                        "components": {
                            "onset": abstract.get("onset"),
                            "medial": abstract.get("medial"),
                            "nucleus": abstract.get("nucleus"),
                            "coda": abstract.get("coda"),
                            "tone": abstract.get("tone"),
                        },
                        "phones": phones,
                    }
                if str(syllable.get("status", "VALID")) != "VALID":
                    fallback += 1
                    has_fallback = True
                    fallback_words.add(token)
            if has_fallback:
                fallback_samples.append(sample_id)
            clips += 1

    lexicon_sorted = {word: lexicon[word] for word in sorted(lexicon)}
    (out_path / "lexicon.txt").write_text(
        "\n".join(f"{word}\t{' '.join(phones)}" for word, phones in lexicon_sorted.items()) + "\n",
        encoding="utf-8",
    )
    all_phones = sorted({phone for phones in lexicon.values() for phone in phones})
    (out_path / "phones.txt").write_text("\n".join(all_phones) + "\n", encoding="utf-8")
    (out_path / "syllable_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    # Clip/từ có fallback (nhãn đọc không chuẩn) — dùng để loại khỏi MFA/acoustic sau QC.
    (out_path / "fallback_samples.txt").write_text(
        "\n".join(sorted(fallback_samples)) + ("\n" if fallback_samples else ""), encoding="utf-8"
    )
    (out_path / "fallback_words.txt").write_text(
        "\n".join(sorted(fallback_words)) + ("\n" if fallback_words else ""), encoding="utf-8"
    )
    return {
        "clips": clips,
        "words": len(lexicon),
        "phones": len(all_phones),
        "fallback_syllables": fallback,
        "fallback_clips": len(fallback_samples),
        "fallback_words": len(fallback_words),
        "missing_audio": missing_audio,
    }


def main() -> None:
    """Đọc tham số, dựng corpus, in thống kê."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True,
                        help="Manifest JSONL (lặp lại được, ví dụ train + val)")
    parser.add_argument("--out", required=True, help="Thư mục corpus cho MFA")
    args = parser.parse_args()

    stats = build_corpus(args.manifest, args.out)
    print(
        f"Corpus tại {args.out}: {stats['clips']} clip · {stats['words']} từ "
        f"· {stats['phones']} phone · fallback {stats['fallback_syllables']} âm tiết "
        f"({stats['fallback_clips']} clip, {stats['fallback_words']} từ) "
        f"· thiếu audio {stats['missing_audio']}"
    )


if __name__ == "__main__":
    main()

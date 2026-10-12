"""Dựng corpus + lexicon cho MFA từ manifest ViFastTts (REVISION_PLAN_V2 §5).

Sinh trong `--out`:
- `wavs/<id>.wav`   — symlink tới audio gốc (copy nếu không symlink được)
- `labs/<id>.lab`   — câu đã respell, token = trường `normalized` của từng âm tiết
- `lexicon.txt`     — `<âm tiết>\t<phone ...>` (gốc để `mfa train` bằng lexicon riêng)
- `phones.txt`      — danh sách phone (tiền tố vị trí: O/M/N/C)
- `syllable_index.json` — normalized → {components, phones} để map ngược TextGrid

Phone set: `O<onset>`, `M<medial>`, `C<coda>`, `N<nucleus><Tone>` (thanh gắn liền
nguyên âm chính — không align thanh thành đoạn riêng). Tiền tố bắt buộc vì cùng ký hiệu
nhưng khác vị trí là khác âm học (NG đầu vs NG cuối, W đệm vs W cuối).

⚠️ **Không dùng `_` trong tên phone** (bài học 13/10/2026): MFA/Kaldi coi hậu tố sau `_`
cuối cùng là "vị trí" (`split_phone_position` = `rsplit("_", 1)`), nên tên kiểu
`n_a_huyen` bị hiểu thành base `n_a` + position `huyen` → TextGrid export mất tone,
thống kê base phone gộp sai. Tên CamelCase ở đây an toàn với mọi regex phone-set của MFA
(không khớp ARPA/PINYIN/IPA) và ≤10 ký tự.

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


def phones_with_sources(abstract: dict[str, str]) -> list[tuple[str, str]]:
    """5 thành phần → [(tên phone, chữ ký nguồn)] theo thứ tự thời gian.

    Chữ ký chỉ gồm thành phần tạo ra phone đó (để phát hiện hai thành phần khác nhau
    sinh cùng một tên). Tên phone **không chứa `_`** (MFA dùng `_` cuối làm dấu vị trí).
    Quy ước: `O<onset>`, `M<medial>`, `N<nucleus><Tone>`, `C<coda>` (Tone viết hoa để tách).
    """
    onset = str(abstract.get("onset", "NONE"))
    medial = str(abstract.get("medial", "NONE"))
    nucleus = str(abstract.get("nucleus", "NONE"))
    coda = str(abstract.get("coda", "NONE"))
    tone = str(abstract.get("tone", "NGANG"))
    pairs: list[tuple[str, str]] = []
    if onset != "NONE":
        pairs.append((f"O{onset.lower()}", f"onset:{onset.lower()}"))
    if medial != "NONE":
        pairs.append((f"M{medial.lower()}", f"medial:{medial.lower()}"))
    pairs.append((f"N{nucleus.lower()}{tone.lower().title()}", f"nucleus:{nucleus.lower()}/{tone.lower()}"))
    if coda != "NONE":
        pairs.append((f"C{coda.lower()}", f"coda:{coda.lower()}"))
    return pairs


def phones_for_syllable(abstract: dict[str, str]) -> list[str]:
    """5 thành phần abstract → danh sách phone có tiền tố vị trí (thứ tự thời gian)."""
    return [name for name, _ in phones_with_sources(abstract)]


def build_corpus(manifests: list[str], out: str | Path) -> dict[str, Any]:
    """Dựng corpus; trả về dict thống kê (clips, words, phones, fallback...).

    Layout **phẳng**: `<out>/<id>.wav` + `<out>/<id>.lab` nằm CẠNH nhau — MFA ghép
    transcription với audio theo tên file **trong cùng thư mục** (tách `labs/`+`wavs/`
    sẽ khiến MFA đọc text rỗng — lỗi đã gặp ở corpus v2b, 12/10/2026).
    """
    out_path = Path(out)
    wav_dir = out_path
    lab_dir = out_path
    out_path.mkdir(parents=True, exist_ok=True)

    lexicon: dict[str, list[str]] = {}
    index: dict[str, dict[str, Any]] = {}
    phone_sources: dict[str, str] = {}  # tên phone → chữ ký thành phần (chống trùng tên giữa 2 thành phần khác nhau)
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
                pairs = phones_with_sources(abstract)
                phones = [phone for phone, _ in pairs]
                for phone, signature in pairs:
                    known = phone_sources.get(phone)
                    if known is None:
                        phone_sources[phone] = signature
                    elif known != signature:
                        raise ValueError(
                            f"xung đột tên phone {phone!r}: {known!r} vs {signature!r}"
                        )
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
    too_long = [phone for phone in all_phones if len(phone) > 10]
    if too_long:
        raise ValueError(f"tên phone > 10 ký tự (giới hạn kaldi_label String(10) của MFA): {too_long}")
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

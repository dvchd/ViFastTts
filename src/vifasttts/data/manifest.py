"""Manifest JSONL: dựng từ metadata CSV, kiểm tra và báo coverage trước khi train."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from vifasttts.frontend.pipeline import parse_text


def build_manifest(metadata: str | Path, wav_dir: str | Path, output: str | Path) -> None:
    """CSV `id|text` + thư mục WAV → manifest JSONL (kèm âm tiết đã parse)."""
    wav_dir = Path(wav_dir)
    rows: list[dict[str, Any]] = []
    with Path(metadata).open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) < 2:
                continue
            sample_id, text = row[0].strip(), row[1].strip()
            wav = wav_dir / f"{sample_id}.wav"
            parsed = [item.to_dict() for item in parse_text(text)]
            rows.append({"id": sample_id, "audio_path": str(wav), "text": text, "syllables": parsed})
    with Path(output).open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(entry, ensure_ascii=False) + "\n" for entry in rows)


def validate_manifest(manifest: str | Path, wav_dir: str | Path | None = None) -> dict[str, Any]:
    """Kiểm tra manifest: WAV tồn tại và mỗi dòng có âm tiết.

    Trả về {"total": n, "issues": [...]}; mỗi issue gồm dòng, id, loại lỗi
    và đường dẫn liên quan. Dùng trước khi train để tránh lỗi giữa chừng.
    """
    root = Path(wav_dir) if wav_dir is not None else None
    issues: list[dict[str, Any]] = []
    total = 0
    with Path(manifest).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            audio = root / f"{row['id']}.wav" if root is not None else Path(str(row.get("audio_path", "")))
            if not audio.exists():
                issues.append({"line": lineno, "id": row.get("id"), "issue": "missing_wav", "path": str(audio)})
            if not row.get("syllables"):
                issues.append({"line": lineno, "id": row.get("id"), "issue": "empty_syllables"})
    return {"total": total, "issues": issues}


def coverage_report(manifest: str | Path) -> dict[str, Any]:
    """Đếm coverage âm vị học của manifest để kiểm tra trước khi train.

    Trả về số âm tiết, số fallback, phân bố onset/nucleus/coda/tone và các
    thanh còn thiếu so với 6 thanh (dùng cho bước "coverage report" trong
    docs/DATA_PIPELINE.md).
    """
    from collections import Counter

    onsets: Counter[str] = Counter()
    nuclei: Counter[str] = Counter()
    codas: Counter[str] = Counter()
    tones: Counter[str] = Counter()
    total = 0
    fallback = 0
    with Path(manifest).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for syllable in row.get("syllables", []):
                total += 1
                if syllable.get("status") == "FALLBACK_REQUIRED":
                    fallback += 1
                abstract = syllable.get("abstract", {})
                onsets[abstract.get("onset")] += 1
                nuclei[abstract.get("nucleus")] += 1
                codas[abstract.get("coda")] += 1
                tones[abstract.get("tone")] += 1
    all_tones = ["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]
    return {
        "syllables": total,
        "fallback": fallback,
        "onset": dict(onsets),
        "nucleus": dict(nuclei),
        "coda": dict(codas),
        "tone": dict(tones),
        "missing_tones": [tone for tone in all_tones if tone not in tones],
    }

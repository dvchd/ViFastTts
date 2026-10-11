from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from vifasttts.frontend.pipeline import parse_text


def build_manifest(metadata: str | Path, wav_dir: str | Path, output: str | Path) -> None:
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

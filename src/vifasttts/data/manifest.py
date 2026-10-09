from __future__ import annotations

import csv
import json
from pathlib import Path
from vifasttts.frontend.pipeline import parse_text


def build_manifest(metadata: str | Path, wav_dir: str | Path, output: str | Path) -> None:
    wav_dir = Path(wav_dir)
    rows = []
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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

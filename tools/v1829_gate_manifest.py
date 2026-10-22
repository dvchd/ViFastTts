"""tools/v1829_gate_manifest.py — gắn cờ gate theo VIDEO vào manifest (nhánh Fc).

Cờ = 1 nếu tỉ lệ frame im kỹ thuật số (RMS < −90 dBFS) của prefix video > 1 %.
Ghi manifest mới (giữ nguyên mọi trường khác) + JSON tỉ lệ theo prefix.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

HOP = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--threshold-pct", type=float, default=1.0)
    args = parser.parse_args()

    rows = [json.loads(x) for x in Path(args.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        start, end = row["frame_range"]
        frames = int(end) - int(start)
        with sf.SoundFile(row["audio_path"]) as handle:
            handle.seek(start * HOP)
            audio = handle.read(frames * HOP, dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < frames * HOP:
            continue
        blocks = audio[: frames * HOP].reshape(frames, HOP).astype(np.float64)
        db = 20 * np.log10(np.sqrt(np.maximum((blocks**2).mean(axis=1), 1e-24)))
        prefix = str(row["id"]).split("_")[0]
        stats[prefix][0] += int((db < -90.0).sum())
        stats[prefix][1] += frames

    shares = {key: round(100.0 * value[0] / max(1, value[1]), 3) for key, value in stats.items()}
    gate = {key: 1 if share > args.threshold_pct else 0 for key, share in shares.items()}
    out_rows = 0
    with Path(args.out).open("w", encoding="utf-8") as handle:
        for row in rows:
            prefix = str(row["id"]).split("_")[0]
            if prefix not in gate:
                continue
            updated = dict(row)
            updated["gate"] = gate[prefix]
            handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
            out_rows += 1

    report = {
        "rows": out_rows,
        "prefixes": len(shares),
        "gate_prefixes": sorted(key for key, value in gate.items() if value == 1),
        "shares_pct": shares,
        "threshold_pct": args.threshold_pct,
        "out": args.out,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "prefixes", "gate_prefixes", "threshold_pct")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

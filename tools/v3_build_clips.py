#!/usr/bin/env python
"""tools/v3_build_clips.py — Data v3 · B2: chuẩn hoá lại + CAP pause + chuyển `seg4` → clip v3.

Cho từng clip (đã DFN denoise ở B1):
 1. **re-normalize** −26 dBFS (đúng luật `normalize_audio_v2`: speech p90−30 / floor −55, boost ≤14,
    peak ≤0.891);
 2. **CAP mọi khoảng lặng** (đầu clip / giữa các âm tiết / đuôi) ở **0,7 s = 66 frame** (93,75 fps) —
    cắt bớt frame lặng thừa (nguyên nhân pause "suy nghĩ" 1–7 s lọt vào train, xem ACOUSTIC_V2 §49.1);
    lặng đầu giữ phần SÁT âm tiết đầu, lặng giữa/đuôi giữ phần ĐẦU (sát âm tiết trước);
 3. **chuyển `seg4`** (O/M/N/C mỗi âm tiết, nhãn hybrid MFA cũ) theo đúng thứ tự âm tiết — cap chỉ
    đổi `gaps`, lòng âm tiết không đổi nên nhãn cũ chuyển được nguyên vẹn;
 4. ghi audio `v3/audio/<clip>.wav` + manifest clip-level v3 (durations giữ nguyên; gaps/lead/pauses
    đã cap; kèm `seg4`).

Clip có số âm tiết KHÔNG khớp giữa manifest clip-level và các segment cũ (đuôi vụn bị bỏ khi cắt cũ)
sẽ bị **bỏ** (log lại) — quy mô nhỏ (~2%), đổi lấy nhãn seg4 nhất quán.

Chạy: .venv/bin/python tools/v3_build_clips.py \
    --manifests data/codelung/train_clean_v2_dur.jsonl,data/codelung/val_clean_v2_dur.jsonl,data/codelung/holdout_clean_v2_dur.jsonl \
    --old-seg data/codelung/train_clean_v2_dur_seg_slot4.jsonl,data/codelung/val_clean_v2_dur_seg_slot4.jsonl,data/codelung/holdout_clean_v2_dur_seg_slot4.jsonl \
    --denoised-dir data/codelung/v3/denoised --out-dir data/codelung/v3
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_RATE = 24000
HOP = 256
FPS = SAMPLE_RATE / HOP  # 93.75
CAP_FRAMES = round(0.7 * FPS)  # 66
TARGET_DB = -26.0
MAX_BOOST_DB = 14.0
PEAK_CEIL = 0.891
SPEECH_P90_OFFSET = 30.0
SPEECH_FLOOR_DB = -55.0


def normalize(wav: np.ndarray) -> np.ndarray:
    """−26 dBFS trên frame có tiếng (luật `normalize_audio_v2._worker`)."""
    step = max(1, int(SAMPLE_RATE * 0.02))
    count = int(wav.size // step)
    if count == 0:
        return wav.astype(np.float32)
    frames = wav[: count * step].reshape(count, step)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12
    db = 20.0 * np.log10(rms)
    threshold = max(float(np.percentile(db, 90)) - SPEECH_P90_OFFSET, SPEECH_FLOOR_DB)
    speech = db[db >= threshold]
    if speech.size == 0:
        speech = db
    speech_rms_db = 10.0 * math.log10(float(np.mean(10.0 ** (speech / 10.0))))
    gain_db = min(TARGET_DB - speech_rms_db, MAX_BOOST_DB)
    peak = float(np.max(np.abs(wav))) * (10.0 ** (gain_db / 20.0))
    if peak > PEAK_CEIL:
        gain_db += 20.0 * math.log10(PEAK_CEIL / peak)
    return np.clip(wav * (10.0 ** (gain_db / 20.0)), -1.0, 1.0).astype(np.float32)


def load_old_seg(path: str) -> dict[str, dict[str, Any]]:
    """Gom segment cũ theo `parent`: danh sách (thứ tự âm tiết, seg4, method...)."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[str(row["parent"])].append(row)
    merged: dict[str, dict[str, Any]] = {}
    for parent, segs in grouped.items():
        segs.sort(key=lambda r: int(r["frame_range"][0]))
        merged[parent] = {
            "syllables": [s for seg in segs for s in seg["syllables"]],
            "seg4": [d for seg in segs for d in seg.get("seg_durations", [])],
            "seg_method": segs[0].get("seg_method"),
            "seg_method_version": segs[0].get("seg_method_version"),
            "seg_weights_hash": segs[0].get("seg_weights_hash"),
        }
    return merged


def build_clip(row: dict[str, Any], old: dict[str, Any] | None, denoised_dir: Path,
               audio_dir: Path) -> dict[str, Any] | None:
    durations = [int(d) for d in row["durations"]]
    gaps = [int(g) for g in row["gaps"]]
    lead = int(row.get("lead_frames") or 0)
    name = Path(str(row["audio_path"])).name
    src = denoised_dir / name
    if not src.exists():
        return None
    if old is None or len(old["syllables"]) != len(row["syllables"]) or len(old["seg4"]) != len(durations):
        return None
    for a, b in zip(old["syllables"], row["syllables"]):
        if str(a.get("raw")) != str(b.get("raw")):
            return None

    # keep-spans theo frame (lặng: đầu giữ phần sát âm tiết đầu; giữa/đuôi giữ phần đầu)
    spans: list[tuple[int, int]] = []
    cursor = lead - min(lead, CAP_FRAMES)
    spans.append((cursor, lead))
    cursor = lead
    new_gaps: list[int] = []
    for i, duration in enumerate(durations):
        spans.append((cursor, cursor + duration))
        cursor += duration
        gap = gaps[i] if i < len(gaps) else 0
        keep = min(gap, CAP_FRAMES)
        spans.append((cursor, cursor + keep))
        cursor += keep
        new_gaps.append(keep)

    import soundfile as sf

    wav, _sr = sf.read(str(src), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    total = len(wav)
    pieces = [wav[max(0, a * HOP): min(total, b * HOP)] for a, b in spans if b > a]
    cut = np.concatenate(pieces) if pieces else wav[:1]
    out = normalize(cut)
    audio_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_dir / name), out, SAMPLE_RATE, subtype="PCM_16")

    pauses = []
    for pause in row.get("pauses", []):
        entry = {"after": int(pause["after"]), "seconds": round(min(float(pause["seconds"]), 0.7), 3)}
        if pause.get("mid_sentence"):
            entry["mid_sentence"] = True
        pauses.append(entry)
    return {
        "id": row["id"],
        "audio_path": f"data/codelung/v3/audio/{name}",
        "text": row["text"],
        "syllables": row["syllables"],
        "durations": durations,
        "gaps": new_gaps,
        "lead_frames": min(lead, CAP_FRAMES),
        "pauses": pauses,
        "seg4": old["seg4"],
        "seg_method": old.get("seg_method"),
        "seg_method_version": old.get("seg_method_version"),
        "seg_weights_hash": old.get("seg_weights_hash"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", required=True)
    parser.add_argument("--old-seg", required=True)
    parser.add_argument("--denoised-dir", default="data/codelung/v3/denoised")
    parser.add_argument("--out-dir", default="data/codelung/v3")
    args = parser.parse_args()

    denoised_dir = ROOT / args.denoised_dir
    out_dir = ROOT / args.out_dir
    audio_dir = out_dir / "audio"
    old_by_clip: dict[str, dict[str, Any]] = {}
    for path in args.old_seg.split(","):
        old_by_clip.update(load_old_seg(path))

    for manifest in args.manifests.split(","):
        rows = [json.loads(line) for line in (ROOT / manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        kept, skipped_missing, skipped_mismatch = [], 0, 0
        for row in rows:
            old = old_by_clip.get(str(row["id"]))
            built = build_clip(row, old, denoised_dir, audio_dir)
            if built is None:
                if old is None:
                    skipped_missing += 1
                else:
                    skipped_mismatch += 1
                continue
            kept.append(built)
        name = Path(manifest).stem.replace("_clean_v2_dur", "")
        out_path = out_dir / f"{name}_clean_v2_dur.jsonl"
        out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n", encoding="utf-8")
        print(f"{manifest}: {len(rows)} → {len(kept)} clip (bỏ {skipped_mismatch} lệch âm tiết, "
              f"{skipped_missing} thiếu seg cũ) → {out_path}", flush=True)


if __name__ == "__main__":
    main()

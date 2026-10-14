"""tools/synth_to_manifest.py — biến dữ liệu TỔNG HỢP thành manifest train được.

Dữ liệu synth (VieNeu-TTS) chỉ có `{id, text, audio_path}` — train cần `durations` (frame/âm tiết)
+ `syllables` (cấu trúc abstract) + `gaps`. Tool này:
1. Transcribe bằng faster-whisper **kèm word timestamps** (audio synth SẠCH ⇒ mốc từ khá tốt).
2. Khớp âm tiết kỳ vọng (`parse_text`) với từ nghe được — dùng khoá **gộp phương ngữ Bắc**
   (r/d/gi · s/x · tr/ch — xem `tools/check_synth_quality.py`) để không lệch vì ASR.
3. Quy ra `durations` (frame @ hop 256/24k), `gaps` (lặng giữa âm tiết), `lead_frames`.
4. Ghi manifest `<out>` — bước sau: `tools/add_seg_durations.py` (rule fallback) rồi `cache-features`.

⚠️ `durations` từ ASR ±30–50 ms (≈2× nhiễu nhãn MFA ~18 ms) — đủ cho ABLATION, KHÔNG dùng cho run
sản phẩm (muốn chuẩn phải MFA align như dữ liệu thật).

Chạy:
    .venv/bin/python tools/synth_to_manifest.py --synth artifacts/synth_data/full.jsonl \
        --out artifacts/synth_data/full_dur.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # cho phép chạy `python tools/synth_to_manifest.py`
    sys.path.insert(0, str(_ROOT))

from tools.check_synth_quality import _syllable_key

HOP = 256
SR = 24000


_ENGINE = None


def _engine():
    """Nạp faster-whisper MỘT LẦN (nạp lại mỗi clip rất tốn: ~1–2 s/clip)."""
    global _ENGINE
    if _ENGINE is None:
        from faster_whisper import WhisperModel

        _ENGINE = WhisperModel("small", device="cpu", compute_type="int8")
    return _ENGINE


def _words_with_times(path: str) -> list[tuple[str, float, float]]:
    """Word timestamps của 1 wav (16 kHz, faster-whisper small)."""
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != 16000:
        count = round(audio.size * 16000 / rate)
        audio = np.interp(np.linspace(0, audio.size - 1, count), np.arange(audio.size), audio)
    segments, _ = _engine().transcribe(audio.astype("float32"), language="vi", beam_size=5,
                                    word_timestamps=True)
    out = []
    for segment in segments:
        for word in getattr(segment, "words", None) or []:
            out.append((word.word.strip(), float(word.start), float(word.end)))
    return out


def build_row(entry: dict, engine_words: list[tuple[str, float, float]]) -> dict | None:
    """Ghép âm tiết kỳ vọng với mốc thời gian nghe được → row có `durations`/`gaps`."""
    import difflib

    from vifasttts.frontend.pipeline import parse_text

    items = parse_text(entry["text"])
    if not items:
        return None
    expected = [_syllable_key(item.normalized) for item in items]
    heard = [_syllable_key(word) for word, _s, _e in engine_words]
    matcher = difflib.SequenceMatcher(a=expected, b=heard, autojunk=False)
    times: list[tuple[float, float] | None] = [None] * len(items)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                _word, start, end = engine_words[j1 + offset]
                times[i1 + offset] = (start, end)
    known = [index for index, value in enumerate(times) if value is not None]
    if len(known) < max(2, len(items) // 2):
        return None
    # Nội suy tuyến tính cho âm tiết không khớp (giữa 2 mốc đã biết).
    for index in range(len(times)):
        if times[index] is not None:
            continue
        left = max([k for k in known if k < index], default=None)
        right = min([k for k in known if k > index], default=None)
        if left is None:
            times[index] = times[right]
        elif right is None:
            times[index] = times[left]
        else:
            ratio = (index - left) / (right - left)
            times[index] = (times[left][0] + ratio * (times[right][0] - times[left][0]),
                            times[left][1] + ratio * (times[right][1] - times[left][1]))

    scale = SR / HOP
    durations, gaps = [], []
    for index, (start, end) in enumerate(times):
        frames = max(1, round((end - start) * scale))
        durations.append(frames)
        if index < len(times) - 1:
            gaps.append(max(0, round((times[index + 1][0] - end) * scale)))
    gaps.append(0)
    lead = max(0, round(times[0][0] * scale))
    return {
        "id": entry["id"], "text": entry["text"], "audio_path": entry["audio_path"],
        "source": "synth", "durations": durations, "gaps": gaps, "lead_frames": lead,
        "syllables": [_item_to_dict(item) for item in items],
    }


def _item_to_dict(item) -> dict:
    """ParsedSyllable → dict khớp định dạng manifest thật (raw/normalized/orthography/abstract)."""
    return {
        "raw": item.raw, "normalized": item.normalized,
        "orthography": {"onset": item.orthography.onset, "medial": item.orthography.medial,
                        "nucleus": item.orthography.nucleus, "coda": item.orthography.coda,
                        "tone_location": item.orthography.tone_location},
        "abstract": {"onset": item.abstract.onset, "medial": item.abstract.medial,
                     "nucleus": item.abstract.nucleus, "coda": item.abstract.coda,
                     "tone": item.abstract.tone.name},
        "boundaries": {"word_end": item.boundaries.word_end, "phrase_end": item.boundaries.phrase_end,
                       "sentence_end": item.boundaries.sentence_end},
    }


def main() -> None:
    """CLI: synth manifest → manifest có durations (+ syllables) cho train."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    entries = [json.loads(l) for l in Path(args.synth).read_text(encoding="utf-8").splitlines() if l]
    if args.limit:
        entries = entries[: args.limit]
    rows, skipped = [], 0
    for index, entry in enumerate(entries, 1):
        try:
            words = _words_with_times(entry["audio_path"])
            row = build_row(entry, words)
        except Exception:  # noqa: BLE001 — 1 clip lỗi không làm sập
            row = None
        if row is None:
            skipped += 1
        else:
            rows.append(row)
        if index % 50 == 0:
            print(f"  … {index}/{len(entries)} (bỏ {skipped})", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"xong {len(rows)}/{len(entries)} row (bỏ {skipped}) → {args.out}")


if __name__ == "__main__":
    main()

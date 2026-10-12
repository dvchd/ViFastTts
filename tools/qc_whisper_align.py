"""QC chéo alignment MFA với Whisper (C3): so word-timing + tỉ lệ phủ tiếng nói.

Với mỗi clip: đọc word tier từ TextGrid MFA, transcribe Whisper (word timestamps),
khớp chuỗi từ (difflib), đo:
- `median_abs_delta_s` / `p90`: |start_whisper − start_mfa| trên các từ khớp (giây).
- `frac_within_0p3s`: tỉ lệ từ lệch < 0,3 s (Whisper cũng có sai số ~±0,2 s).
- `duration_ratio_median`: trung vị độ dài word MFA / Whisper trên từ khớp (≈1 là khớp;
  <<1 là MFA "co từ" — silence chiếm mất đuôi từ).
- `coverage`: tổng thời lượng word / khoảng span của MFA (kỳ vọng ~0,7–0,95 với
  clip nói liên tục; thấp là dấu hiệu alignment co từ / nhồi silence).
- `match_rate`: tỉ lệ từ trong lab khớp được với Whisper (QC nội dung).

Chạy (sau `mfa align`):
    python tools/qc_whisper_align.py --textgrid-dir data/codelung/mfa_v2c_test \
        --corpus data/codelung/mfa_v2c_test --sample 10 \
        --out artifacts/dataset_v2/qc_whisper_test.json

Cần `faster-whisper` (pip). Audio đọc bằng soundfile rồi tự resample 16 kHz (tránh
lỗi PyAV `metadata_errors` ở một số phiên bản av).
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from mfa_to_durations import parse_long_textgrid

WORD_STRIP = re.compile(r"^[^\wàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+|[^\wàáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+$")


def normalize_word(word: str) -> str:
    """Chuẩn hoá từ để so khớp: thường hoá + bỏ dấu câu hai đầu."""
    return WORD_STRIP.sub("", word.strip().lower())


def mfa_words(grid_path: Path) -> list[tuple[float, float, str]]:
    """Word tier của TextGrid MFA → [(start, end, từ đã chuẩn hoá)]."""
    tiers = parse_long_textgrid(grid_path)
    words: list[tuple[float, float, str]] = []
    for xmin, xmax, text in tiers.get("words", []):
        word = normalize_word(text)
        if word:
            words.append((xmin, xmax, word))
    return words


def resample_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample tuyến tính về 16 kHz (đủ tốt cho QC ASR)."""
    if sample_rate == 16000:
        return audio.astype(np.float32)
    target = round(len(audio) * 16000 / sample_rate)
    return np.interp(
        np.linspace(0.0, len(audio), target, endpoint=False),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def compare_clip(
    grid_path: Path, audio_path: Path, whisper_model: Any
) -> dict[str, Any] | None:
    """So khớp một clip; trả dict metric hoặc None nếu thiếu dữ liệu."""
    import soundfile as sf

    words_mfa = mfa_words(grid_path)
    if len(words_mfa) < 3:
        return None
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    segments, _ = whisper_model.transcribe(
        resample_16k(audio, sample_rate), language="vi", word_timestamps=True, beam_size=5
    )
    words_wh = [
        (w.start, w.end, normalize_word(w.word))
        for seg in segments
        for w in seg.words
        if normalize_word(w.word)
    ]
    if not words_wh:
        return None

    a = [w for _, _, w in words_mfa]
    b = [w for _, _, w in words_wh]
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    deltas: list[float] = []
    duration_ratios: list[float] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for i, j in zip(range(i1, i2), range(j1, j2)):
            deltas.append(words_wh[j][0] - words_mfa[i][0])
            dur_wh = words_wh[j][1] - words_wh[j][0]
            dur_mfa = words_mfa[i][1] - words_mfa[i][0]
            if dur_wh > 0.05:
                duration_ratios.append(dur_mfa / dur_wh)

    span = words_mfa[-1][1] - words_mfa[0][0]
    coverage = sum(b - a for a, b, _ in words_mfa) / span if span > 0 else 0.0
    matched = len(deltas)
    return {
        "clip": grid_path.stem,
        "words_mfa": len(words_mfa),
        "words_whisper": len(words_wh),
        "matched": matched,
        "match_rate": round(matched / len(words_mfa), 3),
        "coverage": round(coverage, 3),
        "median_abs_delta_s": round(float(np.median(np.abs(deltas))), 3) if deltas else None,
        "p90_abs_delta_s": round(float(np.percentile(np.abs(deltas), 90)), 3) if deltas else None,
        "frac_within_0p3s": round(float(np.mean(np.abs(deltas) < 0.3)), 3) if deltas else None,
        "duration_ratio_median": round(float(np.median(duration_ratios)), 3) if duration_ratios else None,
    }


def main() -> None:
    """Chạy QC trên một thư mục TextGrid + corpus tương ứng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrid-dir", required=True, help="Thư mục TextGrid (mfa align)")
    parser.add_argument("--corpus", required=True, help="Thư mục corpus chứa `<id>.wav`")
    parser.add_argument("--sample", type=int, default=0, help="Chỉ lấy N clip đầu (0 = hết)")
    parser.add_argument("--model", default="small", help="Model faster-whisper (tiny/base/small...)")
    parser.add_argument("--out", default=None, help="File JSON report")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    grid_dir = Path(args.textgrid_dir)
    corpus = Path(args.corpus)
    grids = sorted(grid_dir.glob("*.TextGrid"))
    if args.sample:
        grids = grids[: args.sample]
    print(f"QC {len(grids)} clip · Whisper {args.model}")

    whisper_model = WhisperModel(args.model, device="cpu", compute_type="int8")
    clips: list[dict[str, Any]] = []
    for grid in grids:
        audio_path = corpus / f"{grid.stem}.wav"
        if not audio_path.exists():
            continue
        metrics = compare_clip(grid, audio_path, whisper_model)
        if metrics:
            clips.append(metrics)
            print(f"  {metrics['clip']}: coverage {metrics['coverage']} · "
                  f"median|Δ| {metrics['median_abs_delta_s']}s · <0,3s {metrics['frac_within_0p3s']}")

    if clips:
        report = {
            "clips": len(clips),
            "coverage_median": round(float(np.median([c["coverage"] for c in clips])), 3),
            "median_abs_delta_s": round(
                float(np.median([c["median_abs_delta_s"] for c in clips if c["median_abs_delta_s"] is not None])), 3
            ),
            "frac_within_0p3s_median": round(
                float(np.median([c["frac_within_0p3s"] for c in clips if c["frac_within_0p3s"] is not None])), 3
            ),
            "duration_ratio_median": round(
                float(
                    np.median(
                        [c["duration_ratio_median"] for c in clips if c["duration_ratio_median"] is not None]
                    )
                ),
                3,
            ),
            "match_rate_median": round(float(np.median([c["match_rate"] for c in clips])), 3),
            "per_clip": clips,
        }
        print(
            f"Tổng: {report['clips']} clip · coverage median {report['coverage_median']} · "
            f"median|Δ| {report['median_abs_delta_s']}s · <0,3s {report['frac_within_0p3s_median']} · "
            f"dur ratio {report['duration_ratio_median']} · match {report['match_rate_median']}"
        )
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Report: {out_path}")


if __name__ == "__main__":
    main()

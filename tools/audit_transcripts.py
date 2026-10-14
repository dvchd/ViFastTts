"""tools/audit_transcripts.py — soát chéo transcript của manifest với ASR (Whisper) từng clip.

Vì sao (P0.2/V11S3): text dạy máy lấy từ phụ đề nguồn và **chứa lỗi chính tả** (đo được:
"thư dục" thay vì "thư mục") ⇒ model đọc đúng cái sai, và **mọi checkpoint sai giống nhau**
(vì lỗi nằm ở dữ liệu). Không có cách sửa checkpoint — phải tìm lỗi trong dữ liệu.

Cách làm: transcribe lại bằng faster-whisper (mạnh hơn phụ đề tự động), chuẩn hoá hai bên
về chữ thường bỏ dấu câu, so từng từ (difflib). Cặp lệch xuất hiện **lặp lại nhiều clip**
= nghi lỗi hệ thống (vd `dục↔mục`), khác với lỗi rải rác của ASR.

Chạy:
    .venv/bin/python tools/audit_transcripts.py --manifest data/codelung/train_clean_v2.jsonl \
        --sample 200 --model small --out artifacts/dataset_v2/audit_transcripts.json

Kết quả JSON: `{model, sample, clips, stats, top_pairs, per_clip}` — `top_pairs` xếp theo
số clip xuất hiện (cặp `manifest↔whisper`), `per_clip` giữ ví dụ để soi tay.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _words(text: str) -> list[str]:
    """Chữ thường, bỏ dấu câu/số — đơn vị so là TỪ (âm tiết tiếng Việt)."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def _load_audio(path: str, target_rate: int = 16000):
    """Đọc wav → mono 16 kHz (soundfile + resample thủ công, tránh lỗi PyAV)."""
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != target_rate:
        # Nội suy tuyến tính đủ tốt cho ASR (không cần chất lượng cao).
        count = round(audio.size * target_rate / rate)
        audio = np.interp(np.linspace(0, audio.size - 1, count), np.arange(audio.size), audio)
    return audio.astype("float32")


def audit(manifest: str | Path, *, sample: int = 200, model: str = "small", seed: int = 42,
          min_words: int = 3) -> dict:
    """Soát `sample` clip đầu (ngẫu nhiên, seed cố định) — trả báo cáo tổng hợp + per-clip."""
    from faster_whisper import WhisperModel

    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line]
    rows = [row for row in rows if len(_words(row.get("text", ""))) >= min_words]
    rng = random.Random(seed)
    picked = rng.sample(rows, min(sample, len(rows)))
    engine = WhisperModel(model, device="cpu", compute_type="int8")

    per_clip: list[dict] = []
    pair_clips: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_count: Counter[tuple[str, str]] = Counter()
    total_words = total_bad = 0
    started = time.time()
    for index, row in enumerate(picked, 1):
        try:
            audio = _load_audio(row["audio_path"])
            segments, _info = engine.transcribe(audio, language="vi", beam_size=5, vad_filter=False)
            heard = _words(" ".join(segment.text for segment in segments))
        except Exception as error:  # noqa: BLE001 — clip lỗi: ghi lại rồi đi tiếp
            per_clip.append({"id": row.get("id"), "error": f"{type(error).__name__}: {error}"})
            continue
        written = _words(row["text"])
        matcher = difflib.SequenceMatcher(a=written, b=heard, autojunk=False)
        bad: list[dict] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            written_part = written[i1:i2]
            heard_part = heard[j1:j2]
            # Ghép cặp theo vị trí trong đoạn lệch (dài ngắn khác nhau vẫn lấy được cặp đầu).
            for offset in range(max(len(written_part), len(heard_part))):
                left = written_part[offset] if offset < len(written_part) else "∅"
                right = heard_part[offset] if offset < len(heard_part) else "∅"
                pair = (left, right)
                pair_count[pair] += 1
                pair_clips[pair].add(str(row.get("id")))
                bad.append({"manifest": left, "whisper": right})
        total_words += len(written)
        total_bad += len(bad)
        per_clip.append({"id": row.get("id"), "n_words": len(written), "n_heard": len(heard),
                         "n_mismatch": len(bad), "mismatch": bad[:12],
                         "manifest_text": row["text"][:200],
                         "whisper_text": " ".join(heard)[:200]})
        if index % 25 == 0:
            print(f"  … {index}/{len(picked)} clip ({time.time() - started:.0f}s)", flush=True)

    top_pairs = [{"manifest": left, "whisper": right, "n_clips": len(pair_clips[(left, right)]),
                  "n_total": pair_count[(left, right)], "clips": sorted(pair_clips[(left, right)])[:5]}
                 for (left, right) in pair_count]
    top_pairs.sort(key=lambda item: (-item["n_clips"], -item["n_total"]))
    return {
        "manifest": str(manifest),
        "model": model,
        "sample": len(picked),
        "clips_ok": sum(1 for item in per_clip if "error" not in item),
        "stats": {
            "total_words": total_words,
            "total_mismatch": total_bad,
            "mismatch_rate": total_bad / max(total_words, 1),
            "clips_with_mismatch": sum(1 for item in per_clip if item.get("n_mismatch", 0) > 0),
        },
        "top_pairs": top_pairs[:120],
        "per_clip": per_clip,
    }


def main() -> None:
    """CLI: soát transcript một manifest, ghi JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--model", default="small", help="model faster-whisper (tiny/base/small/medium)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = audit(args.manifest, sample=args.sample, model=args.model, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = report["stats"]
    print(f"clip {report['clips_ok']}/{report['sample']} · lệch từ "
          f"{stats['total_mismatch']}/{stats['total_words']} = {stats['mismatch_rate'] * 100:.2f}% · "
          f"clip có lệch {stats['clips_with_mismatch']}")
    print("cặp nghi lỗi hệ thống (≥3 clip):")
    for pair in report["top_pairs"][:12]:
        if pair["n_clips"] >= 3:
            print(f"  {pair['manifest']!r} ↔ {pair['whisper']!r}: {pair['n_clips']} clip")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()

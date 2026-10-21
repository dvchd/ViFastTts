"""tools/audit_real_cer.py — CER của audio THẬT (Whisper) trên manifest segment.

Vì sao (A0 lượt 4, Việc 1b.2.1): CER synth (vd 0,37 trên CER-100) chỉ có nghĩa khi so với CER
của **audio thật** cùng công thức. Tool cắt đúng `frame_range` (hop 256 @ 24 kHz), transcribe
faster-whisper small int8, tính CER ký tự (cùng chuẩn hoá với `eval_acoustic`), rồi:
- thống kê mean/median/p90 + top 3% lệch nhất;
- phân loại lỗi **coda** (nh→n, ch→t, ng→n, …), **thanh**, **onset**, **nucleus** bằng
  `parse_syllable` (so 5 thành phần âm vị; giữ dấu vì so thanh);
- xuất **N segment tệ nhất** (wav 24 kHz + HTML) cho người dùng nghe.

Chạy:
  .venv/bin/python tools/audit_real_cer.py --manifest data/codelung/v32/holdout_clean_v2_dur_seg_slot4.jsonl \
      --ids-from artifacts/eval_acoustic/v32_cer100.json --out artifacts/dataset_v2/real_cer_holdout100.json \
      --html-dir artifacts/listen_small_filt/audit_real
  .venv/bin/python tools/audit_real_cer.py --manifest data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl \
      --sample 300 --out artifacts/dataset_v2/real_cer_val300.json
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_VI_CHARS = "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"


def _words(text: str) -> list[str]:
    """Chữ thường, bỏ dấu câu/số — đơn vị so là TỪ (âm tiết tiếng Việt)."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def _cer(reference: str, hypothesis: str) -> float:
    """CER cấp ký tự sau chuẩn hoá (bỏ dấu câu, thường hoá, gộp khoảng trắng)."""

    def clean(text: str) -> str:
        text = text.lower()
        text = re.sub(rf"[^\w\s{_VI_CHARS}]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    ref, hyp = clean(reference), clean(hypothesis)
    if not ref:
        return 0.0
    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start=1):
        current = [i]
        for j, hyp_char in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1] / len(ref)


def _classify(ref_word: str, hyp_word: str) -> str:
    """So 5 thành phần âm vị → nhãn lỗi (coda/thanh/onset/nucleus/other)."""
    from vifasttts.frontend.syllable import parse_syllable

    try:
        a = parse_syllable(ref_word).abstract
        b = parse_syllable(hyp_word).abstract
    except (ValueError, KeyError, TypeError, AttributeError):
        return "other"
    same = [a.onset == b.onset, a.medial == b.medial, a.nucleus == b.nucleus, a.coda == b.coda]
    if all(same) and a.tone != b.tone:
        return "tone"
    if not same[3] and all(same[:3]):
        return f"coda:{a.coda}->{b.coda}"
    if not same[0] and all(same[1:]):
        return f"onset:{a.onset}->{b.onset}"
    if same[0] and same[1] and same[3] and not same[2]:
        return f"nucleus:{a.nucleus}->{b.nucleus}"
    return "other"


def _alignment_errors(reference: str, hypothesis: str) -> list[dict]:
    """Ghép từ theo difflib, trả danh sách cặp thay thế kèm nhãn lỗi."""
    ref_words, hyp_words = _words(reference), _words(hypothesis)
    matcher = difflib.SequenceMatcher(a=ref_words, b=hyp_words, autojunk=False)
    errors: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        span = max(i2 - i1, j2 - j1)
        for offset in range(span):
            ref_word = ref_words[i1 + offset] if i1 + offset < i2 else ""
            hyp_word = hyp_words[j1 + offset] if j1 + offset < j2 else ""
            if ref_word and hyp_word:
                label = _classify(ref_word, hyp_word)
            elif ref_word:
                label = "deletion"
            else:
                label = "insertion"
            errors.append({"ref": ref_word, "hyp": hyp_word, "label": label})
    return errors


def _load_whisper(model_name: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device="cpu", compute_type="int8")


def _transcribe(whisper, wav: np.ndarray, sample_rate: int) -> str:
    """16 kHz numpy (đúng độ dài thời gian) → text; cùng kiểu `_transcribe` của eval_acoustic."""
    target = round(len(wav) * 16000 / sample_rate)
    resampled = np.interp(np.linspace(0, len(wav) - 1, target), np.arange(len(wav)), wav).astype(np.float32)
    segments, _ = whisper.transcribe(resampled, language="vi", beam_size=1)
    return " ".join(segment.text for segment in segments).strip()


def _slice_wav(path: str, frame_range, hop: int, sample_rate: int) -> np.ndarray:
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != sample_rate:
        target = round(len(data) * sample_rate / rate)
        data = np.interp(np.linspace(0, len(data) - 1, target), np.arange(len(data)), data).astype(np.float32)
    if frame_range:
        data = data[int(frame_range[0]) * hop:int(frame_range[1]) * hop]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ids-from", default="", help="JSON có khóa clip_ids (vd v32_cer100.json)")
    parser.add_argument("--sample", type=int, default=0, help="lấy ngẫu nhiên N row (0 = tất cả)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--whisper", default="small")
    parser.add_argument("--worst", type=int, default=30)
    parser.add_argument("--html-dir", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids_from:
        wanted = set(json.loads(Path(args.ids_from).read_text(encoding="utf-8"))["clip_ids"])
        rows = [row for row in rows if row["id"] in wanted]
        if len(rows) != len(wanted):
            print(f"⚠️  chỉ tìm thấy {len(rows)}/{len(wanted)} id trong manifest")
    if args.sample and args.sample < len(rows):
        rows = random.Random(args.seed).sample(rows, args.sample)
    print(f"{len(rows)} row · whisper {args.whisper}")

    whisper = _load_whisper(args.whisper)
    per_clip: list[dict] = []
    labels: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    t0 = time.time()
    for index, row in enumerate(rows, start=1):
        wav = _slice_wav(row["audio_path"], row.get("frame_range"), args.hop, args.sample_rate)
        heard = _transcribe(whisper, wav, args.sample_rate)
        cer = _cer(row["text"], heard)
        errors = _alignment_errors(row["text"], heard)
        for item in errors:
            labels[item["label"]] += 1
            if item["ref"] and item["hyp"]:
                pairs[(item["ref"], item["hyp"])] += 1
        per_clip.append({"id": row["id"], "cer": round(cer, 4), "frames": (row.get("frame_range") or [0, 0])[1]
                         - (row.get("frame_range") or [0, 0])[0], "text": row["text"], "whisper": heard,
                         "errors": errors[:40]})
        if index % 20 == 0:
            print(f"  {index}/{len(rows)} · {time.time() - t0:.0f}s · CER tb {np.mean([c['cer'] for c in per_clip]):.3f}",
                  flush=True)

    cers = np.array([c["cer"] for c in per_clip])
    stats = {"n": len(per_clip), "cer_mean": round(float(cers.mean()), 4),
             "cer_median": round(float(np.median(cers)), 4),
             "cer_p90": round(float(np.percentile(cers, 90)), 4),
             "cer_p97": round(float(np.percentile(cers, 97)), 4)}
    coda_labels = {k: v for k, v in labels.items() if k.startswith("coda:")}
    result = {"manifest": args.manifest, "whisper": args.whisper, "stats": stats,
              "error_labels": dict(labels.most_common()), "coda_labels": dict(sorted(coda_labels.items(),
                                                                                     key=lambda kv: -kv[1])),
              "top_pairs": [{"ref": a, "hyp": b, "n": n} for (a, b), n in pairs.most_common(30)],
              "per_clip": per_clip}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"stats: {json.dumps(stats, ensure_ascii=False)}")
    print(f"coda: {json.dumps(result['coda_labels'], ensure_ascii=False)}")
    print(f"→ {out}")

    worst = sorted(per_clip, key=lambda c: -c["cer"])[:args.worst]
    if args.html_dir:
        import soundfile as sf

        html_dir = Path(args.html_dir)
        (html_dir / "worst").mkdir(parents=True, exist_ok=True)
        table = []
        for rank, item in enumerate(worst, start=1):
            row = next(r for r in rows if r["id"] == item["id"])
            wav = _slice_wav(row["audio_path"], row.get("frame_range"), args.hop, args.sample_rate)
            name = f"worst/{rank:02d}_{item['id'].replace('/', '_')}.wav"
            sf.write(str(html_dir / name), wav, args.sample_rate)
            ref = item["text"].replace("<", "&lt;")
            hyp = item["whisper"].replace("<", "&lt;")
            table.append(f"<tr><td>{rank}</td><td>{item['id']}</td><td>{item['cer']:.3f}</td>"
                         f"<td><audio controls src='{name}'></audio></td><td>{ref}</td><td>{hyp}</td></tr>")
        page = ("<!doctype html><meta charset='utf-8'><title>30 segment lệch nhất (audio THẬT)</title>"
                "<body style='font-family:sans-serif;max-width:1200px;margin:2em auto'>"
                f"<h1>30 segment lệch nhất — CER thật tb {stats['cer_mean']:.3f} (median {stats['cer_median']:.3f})</h1>"
                "<p>Cột trái→phải: hạng · id · CER · audio thật · transcript manifest (dùng để train) · Whisper nghe được. "
                "Chỗ khác nhau = lỗi transcript hoặc lỗi ASR.</p>"
                "<table border='1' cellpadding='6' style='border-collapse:collapse;font-size:13px'>"
                "<tr><th>#</th><th>id</th><th>CER</th><th>nghe</th><th>manifest</th><th>whisper</th></tr>"
                + "".join(table) + "</table></body>")
        (html_dir / "index.html").write_text(page, encoding="utf-8")
        print(f"→ {html_dir}/index.html")
    with (out.with_suffix(".csv")).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "cer", "frames", "text", "whisper"])
        for item in sorted(per_clip, key=lambda c: -c["cer"]):
            writer.writerow([item["id"], item["cer"], item["frames"], item["text"], item["whisper"]])


if __name__ == "__main__":
    main()

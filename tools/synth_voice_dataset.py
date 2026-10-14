"""tools/synth_voice_dataset.py — sinh dữ liệu giọng nói TỔNG HỢP (clone giọng từ clip mẫu).

Dùng cho kế hoạch "bù ô dữ liệu hiếm" (V11S3, 15/10): VieNeu-TTS v3 Turbo (**Apache 2.0 —
dùng thương mại được**, chạy ONNX CPU trên Mac) clone giọng từ clip tham chiếu 3–8 s, sinh
câu đọc cho các ô dữ liệu mỏng (NGÃ/NẶNG + coda vang · coda tắc · câu hỏi).

⚠️ Bài học GTA (§41): audio AI vào TRAIN phải qua **lọc + ablation** — tool này chỉ SINH;
`tools/audit_transcripts.py` (ASR vòng tròn) + so phân bố đặc trưng là bước lọc bắt buộc.

Chạy (venv riêng vì `vieneu` không nằm trong .venv của repo):
    ~/vieneu-venv/bin/python tools/synth_voice_dataset.py \
        --sentences artifacts/synth_data/pilot_sentences.json --ref artifacts/synth_data/ref_voice_8s.wav \
        --out artifacts/synth_data/pilot --manifest artifacts/synth_data/pilot.jsonl --limit 10

Kết quả: `<out>/<id>.wav` (48 kHz) + `<manifest>` JSONL `{id, text, audio_path, source: "synth"}`
+ `synth_report.json` (RTF trung bình, số câu, tổng giây).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    """CLI: sinh từng câu → wav + manifest + báo cáo RTF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", required=True, help="JSON {hand: [...], mined: [...]} hoặc [câu…]")
    parser.add_argument("--ref", required=True, help="Clip giọng mẫu 3–8 s")
    parser.add_argument("--out", required=True, help="Thư mục wav")
    parser.add_argument("--manifest", required=True, help="Manifest JSONL đầu ra")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ sinh N câu đầu (0 = hết)")
    parser.add_argument("--start", type=int, default=0, help="Bỏ qua N câu đầu (để chạy tiếp)")
    parser.add_argument("--mode", default="v3turbo", help="v3turbo (mặc định) | v3nano")
    args = parser.parse_args()

    from vieneu import Vieneu

    payload = json.loads(Path(args.sentences).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        sentences = list(payload.get("hand") or []) + list(payload.get("mined") or [])
    else:
        sentences = list(payload)
    if args.start:
        sentences = sentences[args.start:]
    if args.limit:
        sentences = sentences[: args.limit]

    engine = Vieneu(mode=args.mode)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    total_audio = total_compute = 0.0
    done = 0
    with Path(args.manifest).open("w", encoding="utf-8") as handle:
        for index, text in enumerate(sentences, start=args.start):
            text = text.strip()
            if len(text) < 8:
                continue
            started = time.time()
            try:
                audio = engine.infer(text, ref_audio=args.ref, denoise=True)
            except Exception as error:  # noqa: BLE001 — 1 câu lỗi không làm sập cả lượt
                print(f"  ⚠ câu {index}: {type(error).__name__}: {error}")
                continue
            seconds = len(audio) / 48000.0
            name = f"synth_{index:04d}"
            engine.save(audio, str(out_dir / f"{name}.wav"))
            handle.write(json.dumps({"id": name, "text": text,
                                     "audio_path": str(out_dir / f"{name}.wav"),
                                     "source": "synth", "sample_rate": 48000},
                                    ensure_ascii=False) + "\n")
            total_audio += seconds
            total_compute += time.time() - started
            done += 1
            if done % 10 == 0:
                print(f"  … {done} câu · {total_audio:.0f}s audio · "
                      f"RTF {total_compute / max(total_audio, 1e-6):.2f}", flush=True)

    report = {"sentences": len(sentences), "generated": done, "audio_seconds": round(total_audio, 1),
              "audio_hours": round(total_audio / 3600, 3), "compute_seconds": round(total_compute, 1),
              "rtf": round(total_compute / max(total_audio, 1e-6), 3), "ref": args.ref, "mode": args.mode}
    Path(args.out).parent.joinpath(Path(args.out).name + "_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"xong {done}/{len(sentences)} câu · {total_audio / 60:.1f} phút audio · RTF {report['rtf']}")
    print(f"→ {args.manifest}")


if __name__ == "__main__":
    main()

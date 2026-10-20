"""tools/steady_demo.py — demo A/B cho chế độ "đọc đều" (pause cố định theo dấu câu).

Sinh 3 biến thể cho mỗi câu trong `--texts` (JSON list `{id, text}`):

  normal/<id>.wav  — b8 như hiện tại: model tự đặt pause (gate 0,5)
  flat/<id>.wav    — chặn pause model (gate 0,999) nhưng KHÔNG chèn pause dấu câu
  steady/<id>.wav  — chặn pause model + nghỉ CỐ ĐỊNH:
                     sau phẩy/chấm phẩy `--pause-comma` giây,
                     sau chấm/hỏi/than `--pause-period` giây,
                     mảnh không dấu `--pause-default` giây.

Chạy:
  .venv/bin/python tools/steady_demo.py \
      --config runs/acoustic_v2_b8/resolved_config.json \
      --checkpoint runs/acoustic_v2_b8/best.pt \
      --out artifacts/listen_small_filt/steady --device mps
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import wave
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # cho phép chạy `python tools/steady_demo.py`
    sys.path.insert(0, str(_ROOT))


def _duration(path: Path) -> float:
    """Độ dài WAV (giây) để hiện trong bảng nghe thử."""
    with wave.open(str(path), "rb") as fh:
        return fh.getnframes() / float(fh.getframerate())


def main() -> None:
    """Sinh WAV 3 biến thể + index.html."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--texts", default=str(_ROOT / "examples/steady_texts.json"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--clips", type=int, default=0, help="Giới hạn số câu (0 = tất cả)")
    parser.add_argument("--device", default="cpu", help="cpu (mặc định) | mps | cuda")
    parser.add_argument("--sharpen-alpha", type=float, default=1.5,
                        help="làm nét mel (production = 1,5 — sh25, §33)")
    parser.add_argument("--gate", type=float, default=0.5, help="Ngưỡng gate của bản normal")
    parser.add_argument("--steady-gate", type=float, default=0.999,
                        help="Ngưỡng gate của bản flat/steady (chặn pause model học)")
    parser.add_argument("--pause-comma", type=float, default=0.25)
    parser.add_argument("--pause-period", type=float, default=0.55)
    parser.add_argument("--pause-default", type=float, default=0.08)
    args = parser.parse_args()

    import numpy as np
    import yaml

    from vifasttts.infer import Synthesizer, write_wav
    from vifasttts.runtime.postprocess import postprocess

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    if args.clips > 0:
        rows = rows[: args.clips]
    out = Path(args.out)
    for sub in ("normal", "flat", "steady"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    engine = Synthesizer(config, args.checkpoint, device=args.device)
    sample_rate = engine.sample_rate
    for row in rows:
        rid = str(row["id"])
        text = str(row["text"])
        # 1) normal: như CLI hiện tại
        engine.synthesize(text, out / "normal" / f"{rid}.wav", gate_threshold=args.gate,
                          sharpen_alpha=args.sharpen_alpha)
        # 2) flat: 1 đoạn duy nhất + gate cao (không pause dấu câu)
        pieces = [audio for audio, _ in engine.iter_synthesize(
            text, gate_threshold=args.steady_gate, sharpen_alpha=args.sharpen_alpha,
            max_chars=10**6)]
        waveform = postprocess(np.concatenate(pieces), sample_rate)
        write_wav(out / "flat" / f"{rid}.wav", waveform, sample_rate)
        # 3) steady: cắt tại dấu câu + nghỉ cố định
        engine.synthesize_steady(
            text, out / "steady" / f"{rid}.wav", gate_threshold=args.steady_gate,
            sharpen_alpha=args.sharpen_alpha, pause_comma=args.pause_comma,
            pause_period=args.pause_period, pause_default=args.pause_default)
        print(f"[steady_demo] {rid}: 3 biến thể xong")

    def tag(sub: str, rid: str) -> str:
        src = f"{sub}/{rid}.wav"
        dur = _duration(out / src)
        return f'<audio controls preload="none" src="{src}"></audio><br><small>{dur:.2f}s</small>'

    lines = [
        "<!doctype html><html lang='vi'><head><meta charset='utf-8'>",
        "<title>Chế độ “đọc đều” — pause cố định theo dấu câu</title>",
        ("<style>body{font-family:sans-serif;max-width:1200px;margin:24px auto}"
         "table{border-collapse:collapse;width:100%;font-size:14px}"
         "td,th{border:1px solid #ccc;padding:6px;vertical-align:top}audio{width:230px}</style>"),
        "</head><body>",
        "<h1>“Đọc đều” — model không tự ngắt, runtime nghỉ theo dấu câu</h1>",
        (f"<p><b>normal</b> = b8 như hiện tại (gate {args.gate}) · "
         f"<b>flat</b> = chặn pause model (gate {args.steady_gate}) nhưng không nghỉ dấu câu · "
         f"<b>steady</b> = chặn pause model + nghỉ cố định: phẩy {args.pause_comma}s · "
         f"chấm/hỏi {args.pause_period}s · mặc định {args.pause_default}s.</p>"),
        "<table><tr><th>câu</th><th>nội dung</th><th>normal</th><th>flat</th><th>steady</th></tr>",
    ]
    for row in rows:
        rid = html.escape(str(row["id"]))
        text = html.escape(str(row["text"]))
        lines.append(
            f"<tr><td>{rid}</td><td>{text}</td>"
            f"<td>{tag('normal', str(row['id']))}</td>"
            f"<td>{tag('flat', str(row['id']))}</td>"
            f"<td>{tag('steady', str(row['id']))}</td></tr>")
    lines.append("</table></body></html>")
    index_path = out / "index.html"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nXong: mở {index_path}")


if __name__ == "__main__":
    main()

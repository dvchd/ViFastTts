"""tools/f0_scale_ab.py — trang nghe A/B cho 8b (`--f0-scale` k = 1,0 / 1,25 / 1,45).

Mọi file cân −23 LUFS; cùng model P2 @2k, cùng câu. Kỳ vọng từ số đo: **gần như không khác** —
trang này để xác nhận bằng tai rằng scaling f0 ở runtime không sửa được prosody.

Chạy:
  .venv/bin/python tools/f0_scale_ab.py --config configs/small_codelung_v32.yaml \
      --out artifacts/listen_small_filt/f0scale_ab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint",
                        default="artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt")
    parser.add_argument("--texts", default="examples/drift_texts.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer, write_wav
    from vifasttts.runtime.loudness import normalize_loudness
    from vifasttts.runtime.postprocess import postprocess

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = json.loads(Path(args.texts).read_text(encoding="utf-8"))[:6]
    rows.insert(0, {"id": "d06", "text": "Mình sẽ mở trình duyệt, gõ địa chỉ, rồi bấm tìm kiếm. "
                                             "Sau đó chọn kết quả đầu tiên và đọc phần mô tả."})
    rows.insert(0, {"id": "d04", "text": "Cái ách treo trên tường đã cũ, còn con ếch thì ngồi im "
                                          "bên bờ ao. Mình muốn nghe rõ phụ âm cuối của từng từ."})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    sr = engine.sample_rate
    scales = (1.0, 1.25, 1.45)
    for k in scales:
        (out / f"k{k:.2f}").mkdir(parents=True, exist_ok=True)
        for row in rows:
            pieces = [a for a, _ in engine.iter_synthesize_steady(
                str(row["text"]), sharpen_alpha=0.0, f0_scale=k)]
            waveform = postprocess(np.concatenate(pieces), sr)
            waveform, _info = normalize_loudness(waveform, sr, target_lufs=-23.0)
            write_wav(out / f"k{k:.2f}" / f"{row['id']}.wav", waveform, sr)
    page = ["<!doctype html><meta charset='utf-8'><title>f0-scale A/B</title>",
            ("<style>body{font:15px system-ui;background:#111;color:#eee;max-width:1100px;margin:20px auto}"
             "td{padding:6px} audio{width:320px}</style>"),
            "<h2>8b — giãn biên độ f0 ở runtime (k = 1,0 / 1,25 / 1,45)</h2>",
            "<p>Mọi file −23 LUFS. Số đo: slope 0,785 / 0,780 / 0,783 — gần như không đổi.</p>",
            "<p><b>Nghe xem có khác gì không; nếu không khác thì 8b thất bại (đúng như số đo).</b></p>"]
    for row in rows:
        page.append(f"<h3>{row['id']}: {row['text']}</h3><table><tr>"
                    + "".join(f"<th>k={k:.2f}</th>" for k in scales) + "</tr><tr>"
                    + "".join(f"<td><audio controls src='k{k:.2f}/{row['id']}.wav'></audio></td>"
                              for k in scales) + "</tr></table>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"trang: {out / 'index.html'}")


if __name__ == "__main__":
    main()

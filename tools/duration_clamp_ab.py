"""tools/duration_clamp_ab.py — nghe A/B: pause nội câu cũ (0,999) vs chặn tuyệt đối (1,01) vs + kẹp duration.

Bối cảnh: "rồi" trong d06 bị 42 frame + pause 29 frame; corpus thật: median 17, p90 30, dấu phẩy
nằm TRƯỚC "rồi". Trang này cho nghe 3 biến thể trên cùng model P2 @2k, mọi file −23 LUFS.

Chạy:
  .venv/bin/python tools/duration_clamp_ab.py --config configs/small_codelung_v32.yaml \
      --out artifacts/listen_small_filt/duration_ab
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
    variants = [("cu", 0.999, False), ("moi", 1.01, False), ("clamp", 1.01, True)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    sr = engine.sample_rate
    for name, gate, clamp in variants:
        (out / name).mkdir(parents=True, exist_ok=True)
        for row in rows:
            pieces = [a for a, _ in engine.iter_synthesize_steady(
                str(row["text"]), sharpen_alpha=0.0, gate_threshold=gate, duration_clamp=clamp)]
            waveform = postprocess(np.concatenate(pieces), sr)
            waveform, _info = normalize_loudness(waveform, sr, target_lufs=-23.0)
            write_wav(out / name / f"{row['id']}.wav", waveform, sr)
    labels = {"cu": "CŨ (gate 0,999)", "moi": "MỚI (chặn pause 1,01)", "clamp": "MỚI + kẹp duration"}
    page = ["<!doctype html><meta charset='utf-8'><title>duration A/B</title>",
            ("<style>body{font:15px system-ui;background:#111;color:#eee;max-width:1100px;margin:20px auto}"
             "td{padding:6px} audio{width:300px}</style>"),
            "<h2>Pause nội câu &amp; kẹp duration — P2 @2k, −23 LUFS</h2>",
            ("<p><b>CŨ</b>: model tự chèn pause trong mệnh đề (0,999 vẫn lọt gate bão hoà 1,0). "
             "<b>MỚI</b>: chặn tuyệt đối, pause chỉ do runtime đặt theo dấu câu. "
             "<b>+kẹp</b>: thêm kẹp duration vào p10–p90 corpus (sửa “rồi” 42→30 frame).</p>")]
    for row in rows:
        page.append(f"<h3>{row['id']}: {row['text']}</h3><table><tr>"
                    + "".join(f"<th>{labels[n]}</th>" for n, _, _ in variants) + "</tr><tr>"
                    + "".join(f"<td><audio controls src='{n}/{row['id']}.wav'></audio></td>"
                              for n, _, _ in variants) + "</tr></table>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"trang: {out / 'index.html'}")


if __name__ == "__main__":
    main()

"""tools/clamp_v14_ab.py — nghe A/B v14: câu liền vs đọc đều (không kẹp / kẹp mới) + v3.2.

Mỗi bản ghi rõ checkpoint. Mọi file −23 LUFS.
  - sentence : P2 @2k, đọc CẢ CÂU một lần (không tách dấu câu, không kẹp).
  - steady   : P2 @2k, chế độ đọc đều (gate 1,01), không kẹp.
  - clamp    : P2 @2k, đọc đều + kẹp MỚI (chỉ 1–2 âm tiết đầu chunk, chỉ slot N/C, giữ O).
  - v32      : v3.2, đọc đều, không kẹp (đối chứng).

Chạy:
  .venv/bin/python tools/clamp_v14_ab.py --config configs/small_codelung_v32.yaml \
      --out artifacts/listen_small_filt/clamp_v14
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

D03 = ("Bầu trời hôm nay rất xanh, gió lạnh thổi qua con đường quanh co trước cửa nhà. "
       "Buổi sáng trời lạnh nên mình tưới ít nước.")
D06 = ("Mình sẽ mở trình duyệt, gõ địa chỉ, rồi bấm tìm kiếm. "
       "Sau đó chọn kết quả đầu tiên và đọc phần mô tả.")
D04 = ("Cái ách treo trên tường đã cũ, còn con ếch thì ngồi im bên bờ ao. "
       "Mình muốn nghe rõ phụ âm cuối của từng từ.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--texts", default="examples/drift_texts.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = [{"id": "d03", "text": D03}, {"id": "d06", "text": D06}, {"id": "d04", "text": D04}]
    rows += json.loads(Path(args.texts).read_text(encoding="utf-8"))[:3]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
    engines = {"p2": Synthesizer(config, p2, device=args.device),
               "v32": Synthesizer(config, "runs/acoustic_v32/best.pt", device=args.device)}
    variants = [
        ("sentence_p2", "p2", "sentence"),
        ("steady_p2", "p2", "steady"),
        ("clamp_p2", "p2", "clamp"),
        ("steady_v32", "v32", "steady"),
    ]
    for name, engine_key, mode in variants:
        (out / name).mkdir(parents=True, exist_ok=True)
        engine = engines[engine_key]
        for row in rows:
            if mode == "sentence":
                engine.synthesize(str(row["text"]), out / name / f"{row['id']}.wav",
                                  gate_threshold=1.01, target_lufs=-23.0)
            else:
                engine.synthesize_steady(
                    str(row["text"]), out / name / f"{row['id']}.wav", gate_threshold=1.01,
                    duration_clamp=(mode == "clamp"), target_lufs=-23.0)
    labels = {
        "sentence_p2": "P2 @2k · CẢ CÂU (không tách)",
        "steady_p2": "P2 @2k · đọc đều, KHÔNG kẹp",
        "clamp_p2": "P2 @2k · đọc đều + KẸP MỚI",
        "steady_v32": "v3.2 · đọc đều, KHÔNG kẹp",
    }
    page = ["<!doctype html><meta charset='utf-8'><title>clamp v14 A/B</title>",
            ("<style>body{font:15px system-ui;background:#111;color:#eee;max-width:1200px;margin:20px auto}"
             "td{padding:6px} audio{width:290px}</style>"),
            "<h2>v14 — câu liền vs đọc đều, kẹp cũ/mới (mọi file −23 LUFS)</h2>",
            ("<p>Kẹp mới: chỉ 1–2 âm tiết ĐẦU chunk, chỉ co slot N/C, giữ O, không đụng âm tiết cuối. "
             "Mỗi cột ghi rõ checkpoint.</p>")]
    for row in rows:
        page.append(f"<h3>{row['id']}: {row['text']}</h3><table><tr>"
                    + "".join(f"<th>{labels[n]}</th>" for n, _, _ in variants) + "</tr><tr>"
                    + "".join(f"<td><audio controls src='{n}/{row['id']}.wav'></audio></td>"
                              for n, _, _ in variants) + "</tr></table>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"trang: {out / 'index.html'}")


if __name__ == "__main__":
    main()

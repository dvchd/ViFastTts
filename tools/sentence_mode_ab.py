"""tools/sentence_mode_ab.py — nghe A/B: đọc đều tách chunk (cũ) vs đọc CẢ CÂU (v14).

v14: tách chunk tại dấu phẩy làm mất ngữ cảnh — "rồi" đầu chunk 42 frame (thật 17–24),
"nhà" cuối chunk mất onset (nghe thành "là"). Bản "cả câu" chạy 1 forward cho trọn câu và
chèn pause qua `gaps` của model, nên giữ nguyên ngữ cảnh.

Mọi file −23 LUFS; mỗi cột ghi rõ checkpoint.

Chạy:
  .venv/bin/python tools/sentence_mode_ab.py --config configs/small_codelung_v32.yaml \
      --out artifacts/listen_small_filt/sentence_mode
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
    drift = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    rows += drift[:6]
    rows.append({"id": "long", "text": " ".join(str(r["text"]) for r in drift[:10])})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
    engines = {"p2": Synthesizer(config, p2, device=args.device),
               "v32": Synthesizer(config, "runs/acoustic_v32/best.pt", device=args.device)}
    variants = [
        ("chunk_p2", "p2", "chunk"),
        ("sentence_p2", "p2", "sentence"),
        ("sentence_v32", "v32", "sentence"),
    ]
    for name, engine_key, mode in variants:
        (out / name).mkdir(parents=True, exist_ok=True)
        engine = engines[engine_key]
        for row in rows:
            path = out / name / f"{row['id']}.wav"
            if mode == "sentence":
                engine.synthesize_sentence(str(row["text"]), path, gate_threshold=1.01,
                                           target_lufs=-23.0)
            else:
                engine.synthesize_steady(str(row["text"]), path, gate_threshold=1.01,
                                         target_lufs=-23.0)
    labels = {
        "chunk_p2": "P2 @2k · TÁCH CHUNK (cũ)",
        "sentence_p2": "P2 @2k · CẢ CÂU (v14)",
        "sentence_v32": "v3.2 · CẢ CÂU (v14)",
    }
    page = ["<!doctype html><meta charset='utf-8'><title>sentence mode A/B</title>",
            ("<style>body{font:15px system-ui;background:#111;color:#eee;max-width:1200px;margin:20px auto}"
             "td{padding:6px} audio{width:320px}</style>"),
            "<h2>Đọc đều tách chunk (cũ) vs đọc CẢ CÂU + pause qua gaps (v14)</h2>",
            ("<p>Mọi file −23 LUFS. Bản “cả câu” chạy một forward cho trọn câu nên giữ ngữ cảnh; "
             "pause 0,25 s sau phẩy và 0,55 s sau chấm do runtime chèn.</p>")]
    for row in rows:
        page.append(f"<h3>{row['id']}: {row['text']}</h3><table><tr>"
                    + "".join(f"<th>{labels[n]}</th>" for n, _, _ in variants) + "</tr><tr>"
                    + "".join(f"<td><audio controls src='{n}/{row['id']}.wav'></audio></td>"
                              for n, _, _ in variants) + "</tr></table>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"trang: {out / 'index.html'}")


if __name__ == "__main__":
    main()

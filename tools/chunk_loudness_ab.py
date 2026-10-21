"""tools/chunk_loudness_ab.py — trang nghe A/B cho 6d (chuẩn hoá độ to từng chunk).

Vì sao: 6d làm drift giảm (đều hơn) nhưng giọng thật vốn KHÔNG đều (p90 7,59 dB), nên phải để
tai người dùng quyết định có bật hay không. Trang này phát cùng câu, cùng vocoder, TẮT/BẬT 6d,
mọi file cân −23 LUFS để so công bằng; kèm drift của từng câu.

Chạy:
  .venv/bin/python tools/chunk_loudness_ab.py --config configs/small_codelung_v32.yaml \
      --out artifacts/listen_small_filt/6d_ab
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def voiced_db(x: np.ndarray, sr: int) -> float:
    hop = max(1, int(sr * 0.01))
    n = max(1, len(x) // hop)
    frames = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    db = 20 * np.log10(frames + 1e-12)
    keep = db > db.max() - 30.0
    return float(10 * np.log10(np.mean(frames[keep] ** 2) + 1e-12)) if keep.any() else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--texts", default="examples/drift_texts.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sentences", type=int, default=6)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer, write_wav
    from vifasttts.runtime.loudness import normalize_loudness
    from vifasttts.runtime.postprocess import postprocess

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = json.loads(Path(args.texts).read_text(encoding="utf-8"))[: args.sentences]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = [
        ("v3.2", "runs/acoustic_v32/best.pt"),
        ("P2@2k", "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"),
        ("P2@6k", "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_06000.pt"),
    ]
    results = {}
    for label, ckpt in checkpoints:
        engine = Synthesizer(config, ckpt, device=args.device)
        sr = engine.sample_rate
        for mode in ("off", "on"):
            (out / label.replace("@", "") / mode).mkdir(parents=True, exist_ok=True)
            for row in rows:
                pieces = [a for a, _ in engine.iter_synthesize_steady(
                    str(row["text"]), sharpen_alpha=0.0, chunk_loudness=(mode == "on"))]
                levels = [voiced_db(a, sr) for a in pieces if float(np.max(np.abs(a))) > 1e-3]
                waveform = postprocess(np.concatenate(pieces), sr)
                waveform, _info = normalize_loudness(waveform, sr, target_lufs=-23.0)
                write_wav(out / label.replace("@", "") / mode / f"{row['id']}.wav", waveform, sr)
                drift = float(max(levels) - min(levels)) if len(levels) >= 2 else float("nan")
                results[(label, mode, row["id"])] = drift
        print(f"{label} xong")

    page = ["<!doctype html><meta charset='utf-8'><title>6d A/B</title>",
            ("<style>body{font:15px system-ui;background:#111;color:#eee;max-width:1100px;margin:20px auto}"
             "td{padding:6px 10px;vertical-align:top} .b{color:#9ae6b4} .o{color:#f6ad55}"
             "audio{width:300px}</style>"),
            "<h2>6d — chuẩn hoá độ to từng chunk (TẮT vs BẬT)</h2>",
            ("<p>Mọi file cân −23 LUFS. Cột “drift” là chênh mức giữa các mệnh đề của chính câu đó "
             "(nhỏ = đều hơn). Giọng thật trung vị 3,41 · p90 7,59 dB.</p>"),
            "<p><b>Nghe xem bản BẬT có bị phẳng/mất nhấn không.</b></p>"]
    for row in rows:
        page.append(f"<h3>{row['id']}: {row['text']}</h3><table><tr><th></th><th>TẮT</th><th>BẬT</th></tr>")
        for label, _ in checkpoints:
            tag = label.replace("@", "")
            off = results[(label, "off", row["id"])]
            on = results[(label, "on", row["id"])]
            page.append(
                f"<tr><td>{label}</td>"
                f"<td class='o'>drift {off:.2f} dB<br><audio controls src='{tag}/off/{row['id']}.wav'></audio></td>"
                f"<td class='b'>drift {on:.2f} dB<br><audio controls src='{tag}/on/{row['id']}.wav'></audio></td></tr>")
        page.append("</table>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"trang: {out / 'index.html'}")


if __name__ == "__main__":
    main()

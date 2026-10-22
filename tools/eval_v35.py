"""tools/eval_v35.py — trang nghe MÙ + bảng độ nét cho các ứng viên v3.5 (A0 v17 A7).

Mỗi câu: 4 ứng viên, **xáo thứ tự** bằng seed cố định, nhãn A/B/C/D; đáp án ghi riêng vào JSON
(không hiện trên trang). Đo kèm: flatness 2–6 kHz toàn câu (thước "rè"), độ dài audio.

Chạy:
  .venv/bin/python tools/eval_v35.py --config configs/small_codelung_v35.yaml \
      --candidates "v32=runs/acoustic_v32/best.pt,p2=artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt,v35=runs/acoustic_v35/best_ema.pt,gan2k=runs/adv_v35/eval_2k_ema.pt" \
      --out artifacts/listen_small_filt/a7_blind --json artifacts/bench/a7_blind.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

D03 = ("Bầu trời hôm nay rất xanh, gió lạnh thổi qua con đường quanh co trước cửa nhà. "
       "Buổi sáng trời lạnh nên mình tưới ít nước.")
D06 = ("Mình sẽ mở trình duyệt, gõ địa chỉ, rồi bấm tìm kiếm. "
       "Sau đó chọn kết quả đầu tiên và đọc phần mô tả.")
D04 = ("Cái ách treo trên tường đã cũ, còn con ếch thì ngồi im bên bờ ao. "
       "Mình muốn nghe rõ phụ âm cuối của từng từ.")
LONG = ("Buổi sáng, mình mở cửa sổ, thấy trời xanh và gió nhẹ. "
        "Sau bữa sáng, mình ngồi vào bàn, mở máy tính, rồi đọc lại ghi chú của hôm qua. "
        "Có mấy chỗ mình chưa hiểu, nên mình ghi thêm câu hỏi ra giấy. "
        "Đến trưa, mình đi bộ một vòng quanh khu nhà, ghé qua quán nước quen. "
        "Chiều về, mình sắp lại sách vở, tưới mấy chậu cây ngoài ban công. "
        "Tối đến, mình nấu cơm, nghe lại bản ghi cũ, và sửa từng lỗi nhỏ.")


def flatness(x: np.ndarray, sr: int, n_fft: int = 1024) -> float:
    """Flatness phổ 2–6 kHz (0 = âm sạch, gần 1 = nhiễu/ rè)."""
    if x.size < n_fft:
        return float("nan")
    hop = n_fft // 4
    window = np.hanning(n_fft)
    frames = np.stack([x[i:i + n_fft] * window for i in range(0, x.size - n_fft + 1, hop)])
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= 2000) & (freqs <= 6000)
    power = spec[:, band] + 1e-12
    return float(np.mean(np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True, help="tag=path,tag=path,…")
    parser.add_argument("--texts", default="examples/drift_texts.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--lufs", type=float, default=-23.0)
    args = parser.parse_args()

    from vifasttts.infer import Synthesizer, write_wav
    from vifasttts.runtime.loudness import normalize_loudness
    from vifasttts.runtime.postprocess import postprocess

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sr = int(config.get("sample_rate", 24000))
    candidates = []
    for chunk in args.candidates.split(","):
        tag, rest = chunk.split("=", 1)
        path, _, cfg = rest.partition("|")
        candidates.append((tag.strip(), path.strip(), (cfg or args.config).strip()))

    # id phải DUY NHẤT: drift_texts cũng có d03/d04/d06 ⇒ trùng id sẽ ghi đè WAV + đáp án
    rows = [{"id": "local_d03", "text": D03}, {"id": "local_d06", "text": D06},
            {"id": "local_d04", "text": D04}]
    rows += json.loads(Path(args.texts).read_text(encoding="utf-8"))[:8]
    rows.append({"id": "long", "text": LONG})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # checkpoint cũ train bằng mel stats CŨ ⇒ mỗi ứng viên có thể cần config riêng (`path|config`)
    seen = [row["id"] for row in rows]
    if len(seen) != len(set(seen)):
        raise ValueError(f"id câu trùng nhau: {sorted({i for i in seen if seen.count(i) > 1})}")
    engines = {tag: Synthesizer(yaml.safe_load(Path(cfg).read_text(encoding="utf-8")), path,
                                device=args.device) for tag, path, cfg in candidates}
    rng = random.Random(2026)
    metrics: dict[str, dict[str, list[float]]] = {tag: {"flat": [], "dur": []} for tag, _, _ in candidates}
    key: dict[str, dict[str, str]] = {}
    page = ["<!doctype html><meta charset='utf-8'><title>A7 nghe mù v3.5</title>",
            ("<style>body{font:15px system-ui;background:#111;color:#eee;max-width:1100px;margin:20px auto}"
             "td{padding:6px;vertical-align:top} audio{width:300px} .lbl{font-weight:600}</style>"),
            "<h2>A7 — nghe MÙ 4 ứng viên (P2 @2k · v3.5 · v3.5+GAN · v3.2)</h2>",
            ("<p>Thứ tự A/B/C/D xáo riêng từng câu. Chỉ nghe, không thiên vị nhãn. "
             "Chế độ cả câu · −23 LUFS · α=0.</p>")]
    for row in rows:
        page.append(f"<h3>{row['id']}: {row['text'][:110]}…</h3><table><tr>")
        order = [tag for tag, _, _ in candidates]
        rng.shuffle(order)
        labels = "ABCD"
        mapping = {}
        cells = []
        for label, tag in zip(labels, order):
            engine = engines[tag]
            pieces = [audio for audio, _ in engine.iter_synthesize_sentence(str(row["text"]))]
            wav = postprocess(np.concatenate(pieces), sr).astype(np.float32)
            wav, _ = normalize_loudness(wav, sr, target_lufs=args.lufs)
            folder = out / tag
            folder.mkdir(parents=True, exist_ok=True)
            write_wav(folder / f"{row['id']}.wav", wav, sr)
            metrics[tag]["flat"].append(flatness(wav, sr))
            metrics[tag]["dur"].append(wav.size / sr)
            mapping[label] = tag
            cells.append(f"<td><div class='lbl'>{label}</div>"
                         f"<audio controls src='{tag}/{row['id']}.wav'></audio></td>")
        key[row["id"]] = mapping
        page.append("".join(cells) + "</tr></table>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")

    summary = {tag: {"flat_median": round(float(np.nanmedian(v["flat"])), 4),
                     "dur_median": round(float(np.median(v["dur"])), 2)}
               for tag, v in metrics.items()}
    result = {"summary": summary, "answer_key": key, "n_texts": len(rows),
              "candidates": {tag: path for tag, path, _ in candidates}}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("trang:", out / "index.html")


if __name__ == "__main__":
    main()

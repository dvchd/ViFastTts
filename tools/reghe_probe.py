"""tools/reghe_probe.py — tìm nguồn "rè đều" ở chế độ cả câu (A0 v16, Việc 12a).

4 cấu hình, cùng nội dung (20 câu: d03, d04, d06 + 17 drift):
  C1 chunk   : đọc đều tách chunk (cũ).
  C2 sentence: cả câu, gap = mel lặng chuẩn (hiện tại).
  C3 sentence: cả câu, gap = model tự sinh (không thay mel lặng).
  C4 hybrid  : nhóm ≤ p50 độ dài row train, mỗi nhóm có ngữ cảnh 4 âm tiết hai bên rồi cắt bỏ.

Đo: flatness 2–6 kHz trên 10 từ thường gặp; flatness theo 5 khúc vị trí (đầu→cuối câu);
năng lượng 2–6 kHz quanh biên gap (C2 vs C3); phân bố độ dài row train.

Chạy:
  .venv/bin/python tools/reghe_probe.py --config configs/small_codelung_v32.yaml \
      --out artifacts/listen_small_filt/reghe12a --json artifacts/bench/reghe12a.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

D03 = ("Bầu trời hôm nay rất xanh, gió lạnh thổi qua con đường quanh co trước cửa nhà. "
       "Buổi sáng trời lạnh nên mình tưới ít nước.")
D06 = ("Mình sẽ mở trình duyệt, gõ địa chỉ, rồi bấm tìm kiếm. "
       "Sau đó chọn kết quả đầu tiên và đọc phần mô tả.")
D04 = ("Cái ách treo trên tường đã cũ, còn con ếch thì ngồi im bên bờ ao. "
       "Mình muốn nghe rõ phụ âm cuối của từng từ.")
WORDS = {"trên", "con", "thì", "ao", "của", "là", "có", "được", "không", "một"}


def flatness(x: np.ndarray, sr: int, n_fft: int = 1024) -> tuple[float, float]:
    if x.size < n_fft:
        return float("nan"), float("nan")
    hop = n_fft // 4
    window = np.hanning(n_fft)
    frames = [x[i:i + n_fft] * window for i in range(0, x.size - n_fft + 1, hop)]
    spec = np.abs(np.fft.rfft(np.stack(frames), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= 2000) & (freqs <= 6000)
    p = spec[:, band] + 1e-12
    flat = float(np.mean(np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)))
    ratio = float(np.mean(spec[:, band].sum(axis=1) / (spec.sum(axis=1) + 1e-12)))
    return flat, ratio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint",
                        default="artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt")
    parser.add_argument("--manifest", default="data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--texts", default="examples/drift_texts.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    import torch

    from vifasttts.frontend.pipeline import parse_text
    from vifasttts.infer import Synthesizer, write_wav
    from vifasttts.runtime.loudness import normalize_loudness
    from vifasttts.runtime.postprocess import postprocess, sharpen_mel

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sr = int(config.get("sample_rate", 24000))
    hop = 256
    rows = [{"id": "d03", "text": D03}, {"id": "d06", "text": D06}, {"id": "d04", "text": D04}]
    rows += json.loads(Path(args.texts).read_text(encoding="utf-8"))[:17]

    lengths = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            lengths.append(sum(json.loads(line)["durations"]))
    lengths = np.asarray(lengths)

    engine = Synthesizer(config, args.checkpoint, device=args.device)

    def render_sentence(text: str, fill: bool) -> np.ndarray:
        pieces = [a for a, _ in engine.iter_synthesize_sentence(text, fill_gap_silence=fill)]
        return postprocess(np.concatenate(pieces), sr).astype(np.float32)

    def render_hybrid(text: str, ctx: int = 4) -> np.ndarray:
        """Nhóm ≤ p50 frame, mỗi nhóm có ngữ cảnh 2 bên rồi cắt bỏ (giữ nguyên feature gốc)."""
        items = parse_text(text, resolve_oov=True)
        if not items:
            return np.zeros(1, dtype=np.float32)
        # nhóm theo dấu câu, gộp đến khi vượt p50
        groups, start = [], 0
        for index, item in enumerate(items):
            mark = str(getattr(item, "punctuation", None) or "")
            if mark in (",", ";", ":", ".", "?", "!") or index == len(items) - 1:
                groups.append((start, index))
                start = index + 1
        merged, current = [], None
        for group in groups:
            current = (current[0], group[1]) if current else group
            span = current[1] - current[0] + 1
            if span >= 6 or group == groups[-1]:
                merged.append(current)
                current = None
        mel_parts: list[np.ndarray] = []
        for a, b in merged:
            lo, hi = max(0, a - ctx), min(len(items) - 1, b + ctx)
            sub = items[lo:hi + 1]
            features = torch.tensor([[engine.vocabs.encode(it) for it in sub]],
                                    dtype=torch.long, device=engine.device)
            gaps = np.zeros(len(sub), dtype=np.int64)
            for k, item in enumerate(sub[:-1]):
                mark = str(getattr(item, "punctuation", None) or "")
                if mark == ",":
                    gaps[k] = round(0.25 * sr / hop)
                elif mark in (";", ":"):
                    gaps[k] = round(0.35 * sr / hop)
                elif mark in (".", "?", "!"):
                    gaps[k] = round(0.55 * sr / hop)
            with torch.no_grad():
                speed = torch.tensor([1.0], device=engine.device)
                out = engine.model(features, speed=speed, gate_threshold=1.01,
                                   gaps=torch.tensor(gaps, dtype=torch.long,
                                                     device=engine.device).unsqueeze(0))
            durations = np.asarray(out["durations"].cpu()).reshape(-1)
            lead = int(np.asarray(out["slot_durations"].cpu()).reshape(-1)[0])
            cursor = lead
            offsets = {}
            for k in range(len(sub)):
                offsets[k] = cursor
                cursor += int(durations[k]) + int(gaps[k])
            target_a, target_b = a - lo, b - lo
            start_f = offsets[target_a]
            end_f = offsets[target_b] + int(durations[target_b])
            mel_parts.append(np.asarray(out["mel"][0].float().cpu())[start_f:end_f])
            if (a, b) != merged[-1]:
                silence = getattr(engine, "_silence_mel", None)
                vector = (np.zeros(100) if silence is None or silence is False
                          else np.asarray(silence.cpu()))
                mel_parts.append(np.tile(vector, (round(0.55 * sr / hop), 1)))
        mel = torch.tensor(np.concatenate(mel_parts), dtype=torch.float32).unsqueeze(0).to(engine.device)
        with torch.no_grad():
            wav = engine.vocoder(sharpen_mel(mel[0], 0.0).unsqueeze(0))[0]
        return postprocess(np.asarray(wav.float().cpu().numpy(), dtype=np.float32), sr)

    configs = {
        "C1_chunk": lambda t: postprocess(np.concatenate(
            [a for a, _ in engine.iter_synthesize_steady(t, gate_threshold=1.01)]), sr),
        "C2_sentence_sil": lambda t: render_sentence(t, True),
        "C3_sentence_model": lambda t: render_sentence(t, False),
        "C4_hybrid_ctx": lambda t: render_hybrid(t),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, fn in configs.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        per_word: dict[str, list[float]] = {w: [] for w in WORDS}
        position_flat = [[] for _ in range(5)]
        for row in rows:
            wav = np.asarray(fn(str(row["text"])), dtype=np.float32)
            wav, _ = normalize_loudness(wav, sr, target_lufs=-23.0)
            write_wav(out / name / f"{row['id']}.wav", wav, sr)
            # flatness theo vị trí (5 khúc)
            edges = np.linspace(0, wav.size, 6).astype(int)
            for k in range(5):
                position_flat[k].append(flatness(wav[edges[k]:edges[k + 1]], sr)[0])
            # flatness theo từ: dùng alignment xấp xỉ bằng cách cắt đều theo số âm tiết
            words = str(row["text"]).replace(",", " ").replace(".", " ").split()
            if len(words) >= 5:
                edges_w = np.linspace(0, wav.size, len(words) + 1).astype(int)
                for k, word in enumerate(words):
                    key = word.lower()
                    if key in WORDS:
                        per_word[key].append(flatness(wav[edges_w[k]:edges_w[k + 1]], sr)[0])
        results[name] = {
            "word_flat": {w: round(float(np.median(v)), 3) for w, v in per_word.items() if v},
            "position_flat": [round(float(np.median(v)), 3) if v else None for v in position_flat],
        }
        print(name, results[name])

    result = {"row_len_p50": int(np.percentile(lengths, 50)),
              "row_len_p90": int(np.percentile(lengths, 90)),
              "row_len_p99": int(np.percentile(lengths, 99)),
              "configs": results, "n_rows": len(rows)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    labels = {"C1_chunk": "C1 chunk (cũ)", "C2_sentence_sil": "C2 cả câu + lặng chuẩn",
              "C3_sentence_model": "C3 cả câu + gap model", "C4_hybrid_ctx": "C4 lai (nhóm + ngữ cảnh)"}
    page = ["<!doctype html><meta charset='utf-8'><title>12a rè đều</title>",
            ("<style>body{font:15px system-ui;background:#111;color:#eee;max-width:1300px;margin:20px auto}"
             "td{padding:5px} audio{width:280px}</style>"),
            "<h2>12a — nguồn “rè đều” ở chế độ cả câu (P2 @2k, −23 LUFS)</h2>",
            (f"<p>Row train p50 {result['row_len_p50']} · p90 {result['row_len_p90']} · "
             f"p99 {result['row_len_p99']} frame.</p>")]
    for row in rows[:10]:
        page.append(f"<h3>{row['id']}: {row['text']}</h3><table><tr>"
                    + "".join(f"<th>{labels[n]}</th>" for n in configs) + "</tr><tr>"
                    + "".join(f"<td><audio controls src='{n}/{row['id']}.wav'></audio></td>"
                              for n in configs) + "</tr></table>")
    (out / "index.html").write_text("\n".join(page), encoding="utf-8")
    print(f"trang: {out / 'index.html'}")


if __name__ == "__main__":
    main()

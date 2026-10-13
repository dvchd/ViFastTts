"""tools/listen_demo.py — bộ nghe thử A/B cho một checkpoint acoustic (B0).

Sinh cho mỗi clip (lấy từ manifest, thường là `eval_clips_v1.jsonl`):
  orig/<id>.wav          — audio gốc 24 kHz (copy nguyên file dataset)
  copy/<id>.wav          — copy-synthesis: mel THẬT → Vocos (nghe riêng vocoder)
  <ckpt>/synth_<id>.wav  — free-run: text → acoustic → Vocos (nghe cả pipeline)
  <ckpt>/mp_<i>_<từ>.wav — minimal pairs trong câu khung "Tôi đọc chữ ___ một lần."
Kèm `index.html` để nghe theo từng clip + bảng minimal pairs.

Chạy:
  python tools/listen_demo.py --config configs/small_codelung.yaml \
      --checkpoint runs/small_codelung_v2/best.pt \
      --checkpoint runs/small_codelung_v2/last.pt \
      --manifest data/codelung/eval_clips_v1.jsonl \
      --clips 6 --out artifacts/listen_b0
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # cho phép chạy `python tools/listen_demo.py`
    sys.path.insert(0, str(_ROOT))

CARRIER = "Tôi đọc chữ {word} một lần."


def _slug(text: str) -> str:
    """Bỏ dấu tiếng Việt để đặt tên file an toàn (ASCII)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    stripped = unicodedata.normalize("NFD", text)
    ascii_text = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in ascii_text).strip("_") or "x"


def main() -> None:
    """Sinh WAV gốc/copy/synth + minimal pairs + index.html."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Checkpoint acoustic (lặp lại được, vd best.pt + last.pt)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clips", type=int, default=6)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu", help="cpu (mặc định) | mps | cuda")
    parser.add_argument("--minimal-pairs", default=str(_ROOT / "examples/minimal_pairs.txt"))
    parser.add_argument("--no-minimal-pairs", action="store_true")
    args = parser.parse_args()

    import torch
    import yaml

    from vifasttts.data.features import canonical_mel_config, extract_mel_energy, load_wav
    from vifasttts.infer import Synthesizer, write_wav
    from vifasttts.models.istftnet import build_generator

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sample_rate = int(config.get("sample_rate", 24000))
    mel_config = canonical_mel_config(config)
    device = torch.device(args.device)

    out = Path(args.out)
    (out / "orig").mkdir(parents=True, exist_ok=True)
    (out / "copy").mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    picked = [row for row in rows if (_ROOT / row["audio_path"]).exists()][: args.clips]
    if not picked:
        raise SystemExit("Không tìm thấy clip nào có audio")

    pairs: list[str] = []
    if not args.no_minimal_pairs:
        pairs = [line.strip() for line in Path(args.minimal_pairs).read_text(encoding="utf-8").splitlines()
                 if line.strip()]

    # --- 1) orig + copy-synthesis (không phụ thuộc checkpoint) ---
    generator = build_generator(config["vocoder_training"], config["mel"])
    generator.to(device).eval()
    for row in picked:
        audio_path = _ROOT / row["audio_path"]
        shutil.copyfile(audio_path, out / "orig" / f"{row['id']}.wav")
        wav = load_wav(str(audio_path), sample_rate)
        mel, _ = extract_mel_energy(wav, mel_config)
        with torch.no_grad():
            synth = generator(mel.unsqueeze(0).to(device))[0].float().cpu()
        write_wav(out / "copy" / f"{row['id']}.wav", synth.numpy(), sample_rate)
        print(f"[copy]  {row['id']}  {wav.numel() / sample_rate:.1f}s → {synth.numel() / sample_rate:.1f}s")

    # --- 2) free-run + minimal pairs cho từng checkpoint ---
    checkpoint_labels: list[tuple[str, str]] = []
    for checkpoint in args.checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        step = int(state.get("step", 0))
        label = f"{Path(checkpoint).stem} (bước {step})"
        subdir = f"{Path(checkpoint).stem}_{step}"
        (out / subdir).mkdir(parents=True, exist_ok=True)
        checkpoint_labels.append((label, subdir))

        engine = Synthesizer(config, checkpoint, device=args.device)
        for row in picked:
            target = out / subdir / f"synth_{row['id']}.wav"
            engine.synthesize(row["text"], target)
            print(f"[synth] {subdir}/{row['id']} → {target.name}")
        for index, line in enumerate(pairs):
            words = [word.strip() for word in line.split("|") if word.strip()]
            for position, word in enumerate(words):
                engine.synthesize(CARRIER.format(word=word),
                                  out / subdir / f"mp_{index:02d}_{position}_{_slug(word)}.wav")

    # --- 3) index.html ---
    def audio_tag(path: Path) -> str:
        src = path.relative_to(out).as_posix()
        return f'<audio controls preload="none" src="{src}"></audio>'

    lines = [
        "<!doctype html><html lang='vi'><head><meta charset='utf-8'>",
        "<title>Nghe thử B0 — acoustic v1 + Vocos zero-shot</title>",
        ("<style>body{font-family:sans-serif;max-width:1200px;margin:24px auto}"
         "table{border-collapse:collapse;width:100%;font-size:14px}"
         "td,th{border:1px solid #ccc;padding:6px;vertical-align:top}audio{width:230px}</style>"),
        "</head><body>",
        "<h1>Nghe thử B0 — acoustic v1 + Vocos zero-shot</h1>",
        ("<p><b>orig</b> = bản ghi gốc · <b>copy</b> = mel thật → Vocos (nghe riêng vocoder) · "
         "<b>synth</b> = text → acoustic → Vocos (nghe cả pipeline).</p>"),
        "<h2>Clip</h2><table><tr><th>id</th><th>nội dung (rút gọn)</th><th>gốc</th><th>copy (vocoder)</th>",
    ]
    for label, _ in checkpoint_labels:
        lines.append(f"<th>synth — {html.escape(label)}</th>")
    lines.append("</tr>")
    for row in picked:
        rid = str(row["id"])
        text = str(row.get("text", ""))
        lines.append(f"<tr><td>{html.escape(rid)}</td><td>{html.escape(text[:160])}</td>")
        lines.append(f"<td>{audio_tag(out / 'orig' / (rid + '.wav'))}</td>")
        lines.append(f"<td>{audio_tag(out / 'copy' / (rid + '.wav'))}</td>")
        for _, subdir in checkpoint_labels:
            lines.append(f"<td>{audio_tag(out / subdir / ('synth_' + rid + '.wav'))}</td>")
        lines.append("</tr>")
    lines.append("</table>")

    if pairs:
        lines.append("<h2>Minimal pairs — câu khung “Tôi đọc chữ ___ một lần.”</h2><table>")
        header = "<tr><th>cặp</th><th>từ</th>"
        for label, _ in checkpoint_labels:
            header += f"<th>{html.escape(label)}</th>"
        lines.append(header + "</tr>")
        for index, line in enumerate(pairs):
            words = [word.strip() for word in line.split("|") if word.strip()]
            for position, word in enumerate(words):
                cells = ""
                for _, subdir in checkpoint_labels:
                    mp_name = f"mp_{index:02d}_{position}_{_slug(word)}.wav"
                    cells += f"<td>{audio_tag(out / subdir / mp_name)}</td>"
                pair_text = html.escape(line) if position == 0 else ""
                lines.append(f"<tr><td>{pair_text}</td><td>{html.escape(word)}</td>{cells}</tr>")
        lines.append("</table>")

    lines.append("</body></html>")
    index_path = out / "index.html"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nXong: mở {index_path} bằng trình duyệt để nghe")


if __name__ == "__main__":
    main()

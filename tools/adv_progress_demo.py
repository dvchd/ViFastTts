"""tools/adv_progress_demo.py — trang nghe **tiến trình** fine-tune adversarial (từng checkpoint).

Vì sao: tai người là cổng chính (A0), mà mỗi checkpoint 2k chỉ hơn kém nhau rất ít về số.
Tool gộp `ckpt_*.pt` (EMA) vào checkpoint gốc (giữ `config`/`arch`/`f0_stats`), sinh WAV chế độ
**đọc đều** bằng `tools/steady_demo.py`, rồi dựng 1 trang: mỗi hàng 1 câu, mỗi cột 1 checkpoint.

Chạy lại bất cứ lúc nào — checkpoint đã có WAV thì bỏ qua, chỉ thêm cột mới:
  .venv/bin/python tools/adv_progress_demo.py --run runs/adv_p1_20k \
      --base runs/acoustic_v32/best.pt --config configs/small_codelung_v32.yaml \
      --out artifacts/listen_small_filt/adv_progress --clips 4 --device mps
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _merge(base_path: Path, adv_path: Path, target: Path, use_ema: bool = True) -> int:
    """Gộp trọng số (EMA mặc định) vào checkpoint gốc → ghi `target`, trả số bước."""
    import torch

    base = torch.load(base_path, map_location="cpu", weights_only=False)
    adv = torch.load(adv_path, map_location="cpu", weights_only=False)
    state = adv.get("ema") if use_ema and adv.get("ema") else adv["model"]
    merged = dict(base)
    merged["model"] = state
    merged["step"] = int(adv.get("step", 0))
    merged["adv_source"] = adv_path.name
    torch.save(merged, target)
    return merged["step"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="thư mục có ckpt_*.pt")
    parser.add_argument("--base", required=True, help="checkpoint gốc (đủ metadata)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--texts", default=str(_ROOT / "examples/steady_texts.json"))
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--alphas", default="0", help="mỗi checkpoint sinh 1 cột cho mỗi α")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    run_dir = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = json.loads(Path(args.texts).read_text(encoding="utf-8"))[: args.clips]

    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    def ensure(name: str, checkpoint: Path) -> None:
        """Sinh WAV cho một cột nếu chưa có (mỗi α một thư mục)."""
        for alpha in alphas:
            target = out / f"{name}_a{alpha:g}"
            if (target / "steady").exists() and len(list((target / "steady").glob("*.wav"))) >= len(texts):
                continue
            subprocess.run([sys.executable, str(_ROOT / "tools/steady_demo.py"),
                            "--config", args.config, "--checkpoint", str(checkpoint),
                            "--out", str(target), "--clips", str(args.clips), "--device", args.device,
                            "--sharpen-alpha", str(alpha)],
                           check=True, stdout=subprocess.DEVNULL)

    ensure("v32", Path(args.base))
    columns: list[tuple[int, str]] = []
    for ckpt in sorted(run_dir.glob("ckpt_*.pt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))):
        step = int(re.search(r"(\d+)", ckpt.stem).group(1))
        merged_path = out / "_ckpt" / f"step_{step:05d}.pt"
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        if not merged_path.exists():
            _merge(Path(args.base), ckpt, merged_path)
        name = f"step_{step // 1000:02d}k"
        ensure(name, merged_path)
        for alpha in alphas:
            columns.append((step, f"{name}_a{alpha:g}"))

    head = "".join(f"<th>step {step // 1000}k<br>α={name.rsplit('_a', 1)[-1]}</th>" for step, name in columns)
    rows = []
    for item in texts:
        cells = []
        for _, name in columns:
            src = f"{name}/steady/{item['id']}.wav"
            cells.append(f"<td><audio controls src='{src}'></audio></td>" if (out / src).exists() else "<td>—</td>")
        rows.append(f"<tr><td style='max-width:360px'>{html.escape(item['text'])}</td>"
                    f"<td><audio controls src='v32_a{alphas[0]:g}/steady/{item['id']}.wav'></audio></td>"
                    + "".join(cells) + "</tr>")
    page = ("<!doctype html><meta charset='utf-8'><title>Tiến trình adversarial</title>"
            "<body style='font-family:sans-serif;max-width:1400px;margin:2em auto'>"
            f"<h1>Tiến trình fine-tune adversarial — {run_dir.name}</h1>"
            "<p>Chế độ đọc đều, α=1,5, trọng số <b>EMA</b>. Cùng câu, cùng vocoder. "
            "Cột trái nhất = v3.2 gốc. Nghe lần lượt từ trái sang phải để thấy 'rè' đổi thế nào.</p>"
            "<table border='1' cellpadding='6' style='border-collapse:collapse;font-size:13px'>"
            f"<tr><th>câu</th><th>v3.2 α={alphas[0]:g}</th>{head}</tr>" + "".join(rows) + "</table></body>")
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"→ {out / 'index.html'} · {len(columns)} checkpoint")


if __name__ == "__main__":
    main()

"""tools/build_gta_mel.py — sinh mel **của model acoustic** (GTA) cho từng row manifest.

Vì sao: mel do model sinh **mượt hơn mel thật 58–64 %** (docs/ACOUSTIC_V2.md §32.2, §37.1).
GTA fine-tune Vocos = cho vocoder học cặp (**mel model → audio THẬT của chính row**) để nó
học nghịch đảo đúng độ mờ hệ thống; vòng train ở `tools/train_vocoder_gta.py`.

Cách làm: model chạy **teacher forcing** (`features` + `durations` + `seg_durations` + `gaps`
+ `lead` — đúng như lúc train, train_v2._validate) → `outputs["mel"]` [T,100] **không gian z**
(đúng bằng độ dài row) → lưu `<out>/<row_id>.pt` = `{"mel", "frames", "id"}`.
Không cần audio/cache: chỉ manifest + checkpoint (nên sinh rất nhanh).

Chạy:
    .venv/bin/python tools/build_gta_mel.py --config configs/small_codelung_v2_base.yaml \
        --checkpoint runs/acoustic_v2_base/last.pt \
        --manifest data/codelung/train_clean_v2_dur_seg_slot4.jsonl \
        --rows 2000 --seed 42 --out cache/gta_base \
        --manifest-out artifacts/vocoder_gta/train_gta.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # cho phép chạy `python tools/build_gta_mel.py`
    sys.path.insert(0, str(_ROOT))


def _row_frames(row: dict) -> int:
    """Số frame của row: `frame_range` (segment) hoặc tổng spans (row clip đầy đủ)."""
    frame_range = row.get("frame_range")
    if frame_range:
        return int(frame_range[1]) - int(frame_range[0])
    return int(sum(row.get("durations") or [])) + int(sum(row.get("gaps") or [])) + int(row.get("lead_frames") or 0)


def _load_rows(manifest: str, rows: int, seed: int) -> list[dict]:
    """Đọc manifest; `rows > 0` → lấy mẫu ngẫu nhiên seed cố định (tái lập được)."""
    from vifasttts.data.dataset import ManifestDataset

    all_rows = ManifestDataset(manifest).rows
    if rows and rows < len(all_rows):
        picked = random.Random(seed).sample(range(len(all_rows)), rows)
        return [all_rows[index] for index in sorted(picked)]
    return all_rows


def _group_by_syllables(rows: list[dict], batch_size: int) -> list[list[dict]]:
    """Gom row **cùng N âm tiết** thành batch (model v2 yêu cầu cùng N trong batch)."""
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(len(row.get("syllables") or []), []).append(row)
    batches: list[list[dict]] = []
    for group in groups.values():
        for start in range(0, len(group), batch_size):
            batches.append(group[start:start + batch_size])
    return batches


def main() -> None:
    """Sinh mel model (teacher forcing) cho từng row; ghi `.pt` + manifest con + meta."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint acoustic (vd runs/acoustic_v2_base/last.pt)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="Thư mục lưu <row_id>.pt")
    parser.add_argument("--rows", type=int, default=0, help="0 = toàn bộ manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=8, help="Batch trong nhóm cùng N (MPS)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--manifest-out", default=None, help="Ghi manifest con (chỉ row đã sinh) — cho trainer")
    parser.add_argument("--f0-stats", default=None)
    parser.add_argument("--limit-frames", type=int, default=0, help="Bỏ row dài hơn (0 = không lọc)")
    args = parser.parse_args()

    import torch
    import yaml

    from vifasttts.data.dataset import encode_syllables
    from vifasttts.frontend.vocab import ComponentVocabs

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = _load_rows(args.manifest, args.rows, args.seed)
    if args.limit_frames:
        rows = [row for row in rows if _row_frames(row) <= args.limit_frames]
    if not rows:
        raise SystemExit("không có row nào để sinh mel")

    # Nạp model: ưu tiên f0_stats trong checkpoint → config → mặc định repo (như eval_acoustic).
    from vifasttts.models.factory import arch_of_checkpoint, build_model

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
             len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = arch_of_checkpoint(state if isinstance(state, dict) else {})
    model = build_model(config, sizes, arch=arch)
    model.load_state_dict(state.get("model", state))
    f0_stats = None
    if isinstance(state, dict) and state.get("f0_stats"):
        f0_stats = dict(state["f0_stats"])
    else:
        for candidate in (args.f0_stats, config.get("f0_stats_path"), "data/codelung/f0_stats_v2.json"):
            if candidate and Path(str(candidate)).exists():
                f0_stats = json.loads(Path(str(candidate)).read_text(encoding="utf-8"))
                break
    if arch == "v2_slot5" and f0_stats and hasattr(model, "load_f0_stats"):
        model.load_f0_stats(f0_stats)
    device = torch.device(args.device)
    model.to(device).eval()
    print(f"Model {arch} · {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M · {device} · "
          f"{len(rows)} row · batch {args.batch} (nhóm cùng N)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    batches = _group_by_syllables(rows, args.batch)
    total_frames, done, started = 0, 0, time.perf_counter()
    for batch in batches:
        features = torch.tensor([encode_syllables(row["syllables"], vocabs) for row in batch],
                                dtype=torch.long, device=device)
        durations = torch.tensor([row["durations"] for row in batch], dtype=torch.long, device=device)
        seg_durations = torch.tensor([row["seg_durations"] for row in batch], dtype=torch.long, device=device)
        # gaps dài N-1 trong row segment → chèn 0 cuối (khớp `SlotDataset.__getitem__`).
        gaps = torch.tensor(
            [(list(row.get("gaps") or []) + [0] * len(row["syllables"]))[:len(row["syllables"])]
             for row in batch],
            dtype=torch.long, device=device)
        lead = torch.tensor([[int(row.get("lead_frames") or 0)] for row in batch], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(features, durations=durations, seg_durations=seg_durations, gaps=gaps, lead=lead)
        mel_batch = outputs["mel"]
        for index, row in enumerate(batch):
            frames = _row_frames(row)
            if frames <= 0:
                continue
            mel = mel_batch[index, :frames].detach().to("cpu", torch.float32)
            # Bất biến: mel model phải đúng độ dài row, nếu không thì audio cắt theo frame_range sẽ lệch.
            if mel.size(0) != frames:
                raise ValueError(f"row {row.get('id')}: mel {mel.size(0)} frame ≠ row {frames} frame")
            torch.save({"mel": mel, "frames": frames, "id": str(row.get("id"))}, out / f"{row['id']}.pt")
            total_frames += frames
            done += 1
        elapsed = time.perf_counter() - started
        if done % 200 < args.batch or done == len(rows):
            if str(device).startswith("mps"):
                torch.mps.empty_cache()  # M3: graph cache MPS phình theo shape T khác nhau
            print(f"  {done}/{len(rows)} row · {total_frames} frame · {elapsed:.1f}s "
                  f"({elapsed / max(1, done) * 1000:.0f} ms/row)", flush=True)

    if args.manifest_out:
        manifest_out = Path(args.manifest_out)
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        kept = 0
        with manifest_out.open("w", encoding="utf-8") as handle:
            for row in rows:
                if (out / f"{row['id']}.pt").exists():
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
        print(f"Manifest con: {manifest_out} ({kept} row)")

    meta = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "rows": done,
        "frames": total_frames,
        "seconds": round(time.perf_counter() - started, 2),
        "seed": args.seed,
        "batch": args.batch,
        "device": args.device,
        "checkpoint_step": int(state.get("step", 0)) if isinstance(state, dict) else 0,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Xong: {done} row · {total_frames} frame · {meta['seconds']}s → {out}")


if __name__ == "__main__":
    main()

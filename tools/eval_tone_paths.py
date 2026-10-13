"""tools/eval_tone_paths.py — đo 3 đường A/B/C bằng thước `tools/tone_ruler.ToneRuler` (CPU).

(Chuyển từ artifacts/tone_path/pipeline_abc_ruler.py vào tools/ để được version-control; số liệu:
ACOUSTIC_V2 §35 (small b8) · §36.3–§36.4 (base).)

Sinh `<--out>.json` (mặc định artifacts/tone_path/pipeline_abc_ruler.json — file local).

- **A** = audio THẬT: đúng đường `tone_ruler.py --score-real` (cache f0/vuv + phonation
  cache v3, cửa sổ `seg_durations` manifest) — mốc trần của thước.
- **B** = copy-synthesis: `mel` THẬT (SlotDataset, cache v3) → Vocos của pipeline
  (`runs/acoustic_v2_b8/resolved_config.json` → `vocoder_training.pretrained_path`) → WAV
  → `features_from_wav` (f0/vuv/hnr/jitter/glottalization trích **từ chính wav tổng hợp**
  — cảnh báo tác giả thước) → `item_from_synthesis` (cửa sổ theo `seg_durations` THẬT).
- **C** = free-run: `features` của row → model acoustic (`runs/acoustic_v2_b8/best.pt`,
  `arch=v2_slot5`) → `mel` + `slot_durations` (cửa sổ **theo model**) → Vocos → WAV →
  `features_from_wav` → `item_from_synthesis` (cửa sổ theo `slot_durations` của model).
  Chấm so với **thanh trong features đầu vào** (nhãn manifest).

Chạy từ repo root:
  .venv/bin/python artifacts/tone_path/pipeline_abc_ruler.py --clips 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

TONES = ["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="data/codelung/val_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--config", default="runs/acoustic_v2_b8/resolved_config.json")
    parser.add_argument("--checkpoint", default="runs/acoustic_v2_b8/best.pt")
    parser.add_argument("--vocoder", default="artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin")
    parser.add_argument("--ruler", default="artifacts/tone_path/tone_ruler.pt")
    parser.add_argument("--f0-stats", default="data/codelung/f0_stats_v2.json")
    parser.add_argument("--cache-dir", default="cache/codelung_v2")
    parser.add_argument("--clips", type=int, default=60, help="số row đầu (mặc định 60 — khớp diagnosis.json)")
    parser.add_argument("--out", default="artifacts/tone_path/pipeline_abc_ruler.json")
    parser.add_argument("--old", default="artifacts/tone_path/diagnosis.json",
                        help="số thước CŨ để đối chiếu paired")
    parser.add_argument("--device", default="cpu", help="cpu ONLY (MPS đang train)")
    return parser.parse_args()


def _entry(row: str, syllable: int, tone_true: str | None, *, valid: bool, pred: str | None,
           prob_true: float | None, span: float, n_voiced: float, reason: str) -> dict:
    """Một âm tiết đã chấm (hoặc bị bỏ) của một đường."""
    return {"row": row, "syllable": int(syllable), "tone_true": tone_true, "valid": bool(valid),
            "tone_pred": pred, "prob_true": (None if prob_true is None else round(float(prob_true), 4)),
            "span": round(float(span), 2), "n_voiced": round(float(n_voiced), 2), "reason": reason}


def _confusion(scored: list[dict]) -> list[list[int]]:
    """Ma trận nhầm `[6,6]` (hàng = thật, cột = đoán) trên các âm tiết đã chấm."""
    matrix = [[0] * len(TONES) for _ in TONES]
    for entry in scored:
        if entry["tone_true"] in TONES and entry["tone_pred"] in TONES:
            matrix[TONES.index(entry["tone_true"])][TONES.index(entry["tone_pred"])] += 1
    return matrix


def _summary(entries: list[dict]) -> dict:
    """Tổng hợp một đường: n, bỏ (kèm lý do), accuracy tổng + per-thanh + xác suất đúng TB."""
    scored = [entry for entry in entries if entry["valid"]]
    skipped = [entry for entry in entries if not entry["valid"]]
    reasons: dict[str, int] = {}
    for entry in skipped:
        reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
    correct = sum(1 for entry in scored if entry["tone_pred"] == entry["tone_true"])
    per_tone: dict[str, dict] = {}
    for tone in TONES:
        rows = [entry for entry in scored if entry["tone_true"] == tone]
        hits = sum(1 for entry in rows if entry["tone_pred"] == tone)
        per_tone[tone] = {"n": len(rows), "correct": hits,
                          "accuracy": round(hits / len(rows), 4) if rows else None}
    probabilities = [entry["prob_true"] for entry in scored if entry["prob_true"] is not None]
    return {
        "n_total": len(entries), "n_scored": len(scored), "n_skipped": len(skipped),
        "skip_reasons": reasons, "n_correct": correct,
        "accuracy": round(correct / len(scored), 4) if scored else None,
        "accuracy_drop_as_error": round(correct / len(entries), 4) if entries else None,
        "drop_rate": round(len(skipped) / len(entries), 4) if entries else None,
        "mean_prob_true": round(float(np.mean(probabilities)), 4) if probabilities else None,
        "per_tone": per_tone, "confusion": _confusion(scored), "tones": TONES,
    }


def _paired_summary(per_path: dict[str, list[dict]], keys: list[tuple[str, int]]) -> dict:
    """Accuracy các đường trên **cùng tập âm tiết** (paired) — so công bằng B/C với A."""
    output: dict = {"n": len(keys), "paths": {}}
    for path, entries in per_path.items():
        lookup = {(entry["row"], entry["syllable"]): entry for entry in entries}
        chosen = [lookup[key] for key in keys if key in lookup]
        correct = sum(1 for entry in chosen if entry["tone_pred"] == entry["tone_true"])
        probabilities = [entry["prob_true"] for entry in chosen if entry["prob_true"] is not None]
        output["paths"][path] = {
            "n": len(chosen), "correct": correct,
            "accuracy": round(correct / len(chosen), 4) if chosen else None,
            "mean_prob_true": round(float(np.mean(probabilities)), 4) if probabilities else None,
            "per_tone": {tone: {
                "n": sum(1 for entry in chosen if entry["tone_true"] == tone),
                "correct": sum(1 for entry in chosen if entry["tone_true"] == tone
                               and entry["tone_pred"] == tone)}
                for tone in TONES},
        }
    return output


def _agreement(per_path: dict[str, list[dict]], first: str, second: str) -> dict:
    """Tỉ lệ hai đường đoán cùng thanh (trên giao tập hợp lệ)."""
    left = {(entry["row"], entry["syllable"]): entry for entry in per_path[first] if entry["valid"]}
    right = {(entry["row"], entry["syllable"]): entry for entry in per_path[second] if entry["valid"]}
    keys = set(left) & set(right)
    agree = sum(1 for key in keys if left[key]["tone_pred"] == right[key]["tone_pred"])
    return {"n": len(keys), "agree": agree,
            "rate": round(agree / len(keys), 4) if keys else None,
            "by_tone_true": {tone: {
                "n": sum(1 for key in keys if left[key]["tone_true"] == tone),
                "agree": sum(1 for key in keys if left[key]["tone_true"] == tone
                             and left[key]["tone_pred"] == right[key]["tone_pred"])}
                for tone in TONES}}


def _old_crosscheck(per_path: dict[str, list[dict]], old_path: Path) -> dict:
    """Đối chiếu thước CŨ (`diagnosis.json`) trên **đúng các âm tiết này**.

    - `old_own`: số của thước cũ tự báo (nguyên văn) để tham chiếu.
    - paired: giao (mới hợp lệ) ∩ (cũ đã chấm) → accuracy cũ vs mới trên cùng tập.
    """
    if not old_path.exists():
        return {"status": "missing", "path": str(old_path)}
    payload = json.loads(old_path.read_text(encoding="utf-8"))
    old_records: dict[tuple[str, int], dict] = {}
    for record in payload.get("records", []):
        old_records[(str(record.get("row")), int(record.get("syllable", -1)))] = record
    output: dict = {
        "status": "ok", "source": str(old_path),
        "old_own": {path: {"accuracy": payload["paths"][path]["accuracy"],
                           "n_scored": payload["paths"][path]["n_scored"]}
                    for path in ("A", "B", "C") if path in payload.get("paths", {})},
        "old_paired_own": {path: {"accuracy": payload["paired_abc"][path]["accuracy"],
                                  "n": payload["paired_abc"][path]["n_total"]}
                           for path in ("A", "B", "C") if path in payload.get("paired_abc", {})},
        "paired": {}}
    for path, entries in per_path.items():
        new_valid = {(entry["row"], entry["syllable"]): entry for entry in entries if entry["valid"]}
        both = [key for key in new_valid if key in old_records
                and old_records[key][path].get("pred") is not None]
        old_correct = sum(1 for key in both
                         if old_records[key][path]["pred"] == old_records[key]["tone"])
        new_correct = sum(1 for key in both
                          if new_valid[key]["tone_pred"] == new_valid[key]["tone_true"])
        agree = sum(1 for key in both
                    if old_records[key][path]["pred"] == new_valid[key]["tone_pred"])
        output["paired"][path] = {
            "n_paired": len(both),
            "old_accuracy": round(old_correct / len(both), 4) if both else None,
            "new_accuracy": round(new_correct / len(both), 4) if both else None,
            "agreement_old_new": round(agree / len(both), 4) if both else None,
            "new_valid_old_unscored": sum(1 for key in new_valid if key in old_records
                                          and old_records[key][path].get("pred") is None),
        }
    return output


def main() -> None:
    args = _parse_args()
    import tone_ruler as tr
    import torch

    from tools.eval_acoustic import load_eval_model, resolve_f0_stats
    from tools.eval_vocoder import load_vocoder
    from vifasttts.data.dataset_v2 import SlotDataset

    started = time.time()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    f0_stats = resolve_f0_stats(args.checkpoint, config, args.f0_stats)
    config["f0_stats_path"] = args.f0_stats  # để resolve_f0_stats dùng lại được
    ruler = tr.ToneRuler.load(args.ruler)
    print(f"[1/4] Nạp ruler (val acc {ruler.metrics.get('val', {}).get('accuracy')}) + "
          f"model + Vocos (CPU)…", flush=True)
    model, arch = load_eval_model(args.checkpoint, config, f0_stats=f0_stats)
    model.eval()
    vocoder = load_vocoder(config, args.vocoder, device=args.device)
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir)
    rows = dataset.rows[: args.clips]

    # ── A: audio THẬT (đường --score-real) ────────────────────────────────────
    print(f"[2/4] A: audio thật — {len(rows)} row…", flush=True)
    records, tone_ids, mask, info = tr.extract_manifest(
        args.manifest, cache_dir=args.cache_dir, config_path=args.config,
        f0_stats=ruler.f0_stats, spec=ruler.spec, limit=args.clips)
    keep = mask & (tone_ids >= 0)
    index = np.flatnonzero(keep)
    matrix = tr.assemble_features({key: value[keep] for key, value in records.items()},
                                  spec=ruler.spec)
    probabilities = tr.ensemble_probabilities(ruler.models, matrix)
    row_info = info["row_info"]
    entries_a: list[dict] = []
    for global_index in range(len(row_info)):
        row_id, syllable = row_info[global_index]
        tone_true = TONES[int(tone_ids[global_index])] if tone_ids[global_index] >= 0 else None
        if tone_true is None:
            continue  # không phải tiếng — không tính vào thước
        if keep[global_index]:
            local = int(np.searchsorted(index, global_index))
            entries_a.append(_entry(row_id, syllable, tone_true, valid=True,
                                    pred=TONES[int(probabilities[local].argmax())],
                                    prob_true=probabilities[local][TONES.index(tone_true)],
                                    span=records["span"][global_index],
                                    n_voiced=records["n_voiced"][global_index], reason=""))
        else:
            reason = ("ngắn" if records["span"][global_index] < ruler.spec.get("min_span", 4)
                      else "ít hữu thanh")
            entries_a.append(_entry(row_id, syllable, tone_true, valid=False, pred=None,
                                    prob_true=None, span=records["span"][global_index],
                                    n_voiced=records["n_voiced"][global_index], reason=reason))
    # giữ thứ tự theo (row, syllable) — cùng chuẩn với B/C
    entries_a.sort(key=lambda entry: (entry["row"], entry["syllable"]))
    print(f"  A xong: {_summary(entries_a)['accuracy']} trên "
          f"{_summary(entries_a)['n_scored']} âm tiết", flush=True)

    # ── B + C: vòng theo row (Vocos + trích lại từ wav tổng hợp) ──────────────
    entries_b: list[dict] = []
    entries_c: list[dict] = []
    per_row: list[dict] = []
    speed = torch.tensor([1.0], dtype=torch.float32)
    hop = int(config["mel"]["hop_length"])
    print(f"[3/4] B (copy-synthesis) + C (free-run) — {len(rows)} row…", flush=True)
    for position, row in enumerate(rows):
        row_id = str(row.get("id", position))
        item = dataset[position]
        tones = tr.syllable_tones(row)
        detail = {"row": row_id, "n_syllables": len(tones)}
        try:
            # ── B: mel THẬT → Vocos → wav → trích từ wav ──
            mel_real = item["mel"]
            with torch.no_grad():
                wav_b = vocoder(mel_real.unsqueeze(0).to(args.device))[0].float().cpu()
            item_b = {"seg_durations": row["seg_durations"], "durations": row["durations"],
                      "gaps": row.get("gaps"), "lead": row.get("lead_frames") or 0}
            merged_b = tr.item_from_synthesis(item_b, wav_b, config, tones=tones)
            scored_b = ruler.score_items([merged_b])[0]
            detail["B_mel_frames"] = int(mel_real.size(0))
            detail["B_wav_frames"] = int(wav_b.numel() // hop)
            for entry in scored_b:
                tone_true = entry["tone_true"]  # score_items đã trả chuỗi thanh (khớp `tones`)
                valid = bool(entry["valid"])
                entries_b.append(_entry(
                    row_id, entry["index"], entry["tone_true"], valid=valid, pred=entry["tone"],
                    prob_true=(entry["probs"][tone_true] if valid and tone_true in entry["probs"]
                               else None),
                    span=entry["span"], n_voiced=entry["n_voiced"],
                    reason="" if valid else (entry["reason"] or "bỏ")))
        except Exception as error:  # noqa: BLE001 — ghi lỗi vào JSON, không chết cả run
            import traceback

            traceback.print_exc()
            detail["B_error"] = f"{type(error).__name__}: {error}"
            for syllable, tone in enumerate(tones):
                entries_b.append(_entry(row_id, syllable, tone, valid=False, pred=None,
                                        prob_true=None, span=0, n_voiced=0,
                                        reason=f"error:{type(error).__name__}"))
        try:
            # ── C: features → model → mel+slot_durations → Vocos → wav → trích từ wav ──
            with torch.no_grad():
                outputs = model(item["features"].unsqueeze(0).to(args.device),
                                speed=speed.to(args.device))
            mel_c = outputs["mel"][0].float().cpu()
            slots = outputs["slot_durations"][0].float().cpu()
            with torch.no_grad():
                wav_c = vocoder(mel_c.unsqueeze(0).to(args.device))[0].float().cpu()
            merged_c = tr.item_from_synthesis({"slot_durations": slots}, wav_c, config, tones=tones)
            scored_c = ruler.score_items([merged_c])[0]
            detail["C_mel_frames"] = int(mel_c.size(0))
            detail["C_slot_sum"] = round(float(slots.sum()))
            detail["C_wav_frames"] = int(wav_c.numel() // hop)
            detail["C_length_ratio"] = round(float(mel_c.size(0)) / max(1, int(mel_real.size(0))), 3)
            for entry in scored_c:
                tone_true = entry["tone_true"]
                valid = bool(entry["valid"])
                entries_c.append(_entry(
                    row_id, entry["index"], tone_true, valid=valid, pred=entry["tone"],
                    prob_true=(entry["probs"][tone_true] if valid and tone_true in entry["probs"]
                               else None),
                    span=entry["span"], n_voiced=entry["n_voiced"],
                    reason="" if valid else (entry["reason"] or "bỏ")))
        except Exception as error:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            detail["C_error"] = f"{type(error).__name__}: {error}"
            for syllable, tone in enumerate(tones):
                entries_c.append(_entry(row_id, syllable, tone, valid=False, pred=None,
                                        prob_true=None, span=0, n_voiced=0,
                                        reason=f"error:{type(error).__name__}"))
        per_row.append(detail)
        print(f"  [{position + 1:3d}/{len(rows)}] {row_id[:44]:44s} N={len(tones):3d} "
              f"B {detail.get('B_wav_frames', '?'):>5} C {detail.get('C_wav_frames', '?'):>5} "
              f"({time.time() - started:.0f}s)", flush=True)

    # ── Tổng hợp + đối chiếu ─────────────────────────────────────────────────
    print("[4/4] Tổng hợp…", flush=True)
    entries_b = [entry for entry in entries_b if entry["tone_true"] in TONES]
    entries_c = [entry for entry in entries_c if entry["tone_true"] in TONES]
    per_path = {"A": entries_a, "B": entries_b, "C": entries_c}
    summary = {path: _summary(entries) for path, entries in per_path.items()}
    valid_keys = sorted(set.intersection(*[
        {(entry["row"], entry["syllable"]) for entry in entries if entry["valid"]}
        for entries in per_path.values()]))
    paired = _paired_summary(per_path, valid_keys)
    payload = {
        "status": "ok", "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),        "tool": "artifacts/tone_path/pipeline_abc_ruler.py",
        "ruler": {"model": args.ruler, "val_accuracy_train": ruler.metrics.get("val", {}).get("accuracy"),
                  "spec_version": ruler.spec.get("version"),
                  "self_check_note": "tự kiểm chứng trước: tools/tone_ruler.py --score-real → 0.7623 val"},
        "config": {
            "manifest": args.manifest, "clips": len(rows), "device": args.device,
            "checkpoint": args.checkpoint, "arch": arch, "vocoder": args.vocoder,
            "cache_dir": args.cache_dir, "f0_stats": args.f0_stats,
            "A": "audio thật (cache f0/vuv/phonation v3, cửa sổ seg_durations manifest)",
            "B": "mel THẬT → Vocos b8 → wav → features_from_wav (cửa sổ seg_durations THẬT)",
            "C": "features → model b8 (free-run, speed=1) → mel+slot_durations → Vocos → wav → "
                 "features_from_wav (cửa sổ theo model)",
        },
        "paths": summary,
        "accuracy_drop_vs_A": {
            path: (round(summary["A"]["accuracy"] - summary[path]["accuracy"], 4)
                   if summary[path]["accuracy"] is not None else None)
            for path in ("B", "C")},
        "paired_abc": paired,
        "paired_diffs": {
            "B_minus_A": round(paired["paths"]["B"]["accuracy"]
                               - paired["paths"]["A"]["accuracy"], 4),
            "C_minus_B": round(paired["paths"]["C"]["accuracy"]
                               - paired["paths"]["B"]["accuracy"], 4),
            "C_minus_A": round(paired["paths"]["C"]["accuracy"]
                               - paired["paths"]["A"]["accuracy"], 4),
        },
        "pairwise_agreement": {"AB": _agreement(per_path, "A", "B"),
                               "BC": _agreement(per_path, "B", "C"),
                               "AC": _agreement(per_path, "A", "C")},
        "old_ruler_crosscheck": _old_crosscheck(per_path, Path(args.old)),
        "per_row": per_row,
        "notes": [
            ("B/C trích f0/vuv/hnr/jitter/glottalization TỪ WAV TỔNG HỢP (features_from_wav) "
             "— không dùng seg_phonation pooled của dataset (giảm ~2 điểm nếu làm sai)."),
            "C chấm so thanh của features đầu vào; số âm tiết = N của features (model không đổi N).",
            ("Thước có trần 0,762 trên audio thật (weak label từ chính tả) — chênh lệch trong ±0,02 "
             "coi như nhiễu. A ở đây 0,7955 vì 60 row đầu dễ hơn mức trung bình val."),
        ],
        "runtime_seconds": round(time.time() - started, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    syllables_path = out.with_name(out.stem + "_syllables.json")
    syllables_path.write_text(json.dumps(per_path, ensure_ascii=False), encoding="utf-8")

    for path in ("A", "B", "C"):
        entry = summary[path]
        print(f"  {path}: {entry['n_correct']}/{entry['n_scored']} = {entry['accuracy']} "
              f"(bỏ {entry['n_skipped']}/{entry['n_total']} {entry['skip_reasons']}) "
              f"· P(đúng TB) {entry['mean_prob_true']}")
    print(f"  paired n={paired['n']}: "
          + " · ".join(f"{path} {paired['paths'][path]['accuracy']}" for path in ("A", "B", "C"))
          + f" → Δ(B−A) {payload['paired_diffs']['B_minus_A']} · "
          f"Δ(C−B) {payload['paired_diffs']['C_minus_B']} · "
          f"Δ(C−A) {payload['paired_diffs']['C_minus_A']}")
    print(f"Đã ghi {out} + {syllables_path} ({payload['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

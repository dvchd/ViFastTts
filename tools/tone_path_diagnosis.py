"""tools/tone_path_diagnosis.py — chẩn đoán thanh điệu "mất ở khâu nào" (T3.4).

Câu hỏi (docs/ACOUSTIC_V2.md §31.2, §32): model v2 chấm thanh từ **audio tổng hợp**
chỉ ~0,19–0,22 (gần ngẫu nhiên 1/6 = 0,167) nhưng từ **f0 head** của model là 0,58–0,61.
Vậy contour F0 mất ở **mel của model** hay ở **Vocos**?

Bốn đường, cùng một bộ phân loại (`tools/eval_tone_pairs.py`) + cùng cách trích F0
(`_robust_f0`: harvest → dio → autocorr, chọn bộ nhiều frame hữu thanh nhất):

- **A** audio THẬT của segment (`frame_range` → `wav[start*hop:end*hop]`) → harvest F0 →
  classify. Đây là **mốc trần** của bộ phân loại; cổng: **A > 0,4** (thấp hơn ⇒
  alignment/harvest sai, phải rà trước khi báo cáo).
- **B** copy-synthesis: mel THẬT (cache) → Vocos → harvest F0 → classify
  (đường mel→audio có giữ thanh không).
- **C** free-run: text (features của row) → model → mel → Vocos → harvest F0 → classify
  (đúng pipeline hiện tại; cửa sổ theo `slot_durations` **do model sinh**).
- **D** f0 head của model (log-F0 tự nhiên, `exp`) → classify, cửa sổ theo `slot_durations`,
  **không** lọc hữu thanh (giống `--source model` của T3.3) — đối chiếu mốc 0,58–0,61.

Cửa sổ đo mặc định là **nucleus** (N slot) — khớp định nghĩa `shape_mean`/`tone_prior_z`
của R5b/T3.2; đường A/B dùng `seg_durations` của row, đường C/D dùng `slot_durations`
của model. Chỉ chấm âm tiết có ≥ 5 frame hữu thanh (như T3.3); số bỏ ghi rõ.

Kiểm tra alignment (bắt buộc trước khi đo): harvest lại F0 trên đoạn audio vừa cắt rồi
so với f0 THẬT trong cache (chỉ frame cùng hữu thanh) — tương quan trung vị phải > 0,8;
không đạt ⇒ **DỪNG** (exit 2) vì mọi số sau đó vô nghĩa. Lưu ý B/C có thể rụng hữu thanh
(ảnh Vocos) — vì vậy JSON có thêm bảng **paired** (giao A∩B∩C) để so công bằng.

Ví dụ:
  .venv/bin/python tools/tone_path_diagnosis.py \
      --config runs/acoustic_v2_b8/resolved_config.json \
      --checkpoint runs/acoustic_v2_b8/best.pt \
      --manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl \
      --cache-dir cache/codelung_v2 --clips 60 --device cpu \
      --out artifacts/tone_path/diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # cho phép chạy `python tools/tone_path_diagnosis.py`
    sys.path.insert(0, str(_ROOT))

from tools.eval_acoustic import resolve_f0_stats  # checkpoint → config → mặc định repo
from tools.eval_tone_pairs import (
    TONES,
    _robust_f0,
    classify_shape,
    contour_to_sk_z,
    target_frame_window,
    tone_prior_z_from_stats,
)

#: Số frame hữu thanh tối thiểu trong cửa sổ để chấm (như T3.3).
MIN_VOICED_FRAMES = 5
#: Cổng alignment: tương quan trung vị harvest (đoạn cắt) vs f0 cache phải vượt mức này.
ALIGNMENT_MIN_CORR = 0.8
#: Cổng đường A: dưới mức này coi như alignment/harvest sai — không kết luận.
GATE_A_MIN_ACC = 0.4
#: Ngưỡng "mất thanh" cho gợi ý kết luận (paired): chênh accuracy ≥ mức này.
HINT_LOSS = 0.15

PATH_LABELS = {
    "A": "A audio thật",
    "B": "B mel thật→Vocos",
    "C": "C model→Vocos (free)",
    "D": "D f0 head model",
}


def _slot_list(seg_durations: list, gaps: list, lead: int) -> list[int]:
    """`[lead] + (O,M,N,C,G) × N` — cùng layout `slot_durations` của model (dùng chung cửa sổ)."""
    slots = [int(lead)]
    for index, segment in enumerate(seg_durations):
        slots.extend(int(v) for v in segment)
        slots.append(int(gaps[index]) if index < len(gaps) else 0)
    return slots


def _classify_audio_window(segment, sample_rate: int, hop: int, stats: dict,
                           priors: np.ndarray, min_voiced: int) -> dict:
    """Audio (torch tensor) → dict `{pred, n_voiced, s_k_z, f0_log, frames}`.

    `f0_log`/`frames` = log-F0 harvest + index frame **hữu thanh** trong cửa sổ (để đối chiếu head
    của model trên đúng frame); None khi bị lọc.
    """
    empty = {"pred": None, "n_voiced": 0, "s_k_z": None, "f0_log": None, "frames": None}
    if segment.numel() < hop:  # cửa sổ rỗng/quá ngắn
        return empty
    if segment.numel() < 1024:  # `_robust_f0` cần frame 1024 mẫu cho nhánh autocorr
        from vifasttts.data.features import estimate_f0_pyworld

        f0, vuv = estimate_f0_pyworld(segment, sample_rate, hop, f_min=70, f_max=400,
                                      interpolate=False)
    else:
        f0, vuv = _robust_f0(segment, sample_rate, hop)
    voiced = (vuv > 0.5).numpy()
    count = int(voiced.sum())
    if count < min_voiced:
        return {**empty, "n_voiced": count}
    values = f0.numpy()[voiced]
    s_k_z = contour_to_sk_z(values, f0_log_mean=float(stats["f0_log_mean"]),
                            f0_log_std=float(stats["f0_log_std"]),
                            shape_mean=np.asarray(stats["shape_mean"], dtype=np.float64),
                            shape_std=np.asarray(stats["shape_std"], dtype=np.float64))
    return {"pred": classify_shape(s_k_z, priors), "n_voiced": count, "s_k_z": s_k_z,
            "f0_log": np.log(np.maximum(values, 1e-6)), "frames": np.flatnonzero(voiced)}


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    """Tương quan Pearson (None nếu < 5 điểm hoặc một bên gần như hằng)."""
    if left.size < 5 or right.size < 5 or left.size != right.size:
        return None
    if left.std() < 1e-9 or right.std() < 1e-9:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _follow_head(f0_log: np.ndarray | None, frames: np.ndarray | None,
                 reference_log: np.ndarray) -> tuple[float | None, float | None, int]:
    """Audio có bám contour tham chiếu không: tương quan (thô, bỏ mức) trên frame hữu thanh."""
    if f0_log is None or frames is None or frames.size == 0:
        return None, None, 0
    keep = frames < reference_log.size  # Vocos ra (T−1)*hop mẫu → cửa sổ cuối có thể bị cắt
    index = frames[keep]
    if index.size < 5:
        return None, None, int(index.size)
    values = f0_log[keep]
    reference = reference_log[index]
    raw = _corr(values, reference)
    shape = _corr(values - values.mean(), reference - reference.mean())
    return raw, shape, int(index.size)


def _classify_model_f0(f0_log_window: np.ndarray, stats: dict, priors: np.ndarray):
    """log-F0 của head model (đơn vị tự nhiên) → thanh dự đoán (không lọc hữu thanh, như T3.3)."""
    raw = np.asarray(f0_log_window, dtype=np.float64)
    if raw.size < 2:
        return None, int(raw.size), None
    s_k_z = contour_to_sk_z(np.exp(raw), f0_log_mean=float(stats["f0_log_mean"]),
                            f0_log_std=float(stats["f0_log_std"]),
                            shape_mean=np.asarray(stats["shape_mean"], dtype=np.float64),
                            shape_std=np.asarray(stats["shape_std"], dtype=np.float64))
    return classify_shape(s_k_z, priors), int(raw.size), s_k_z


def _new_bucket() -> dict:
    """Bộ đếm một đường: tổng/chấm/bỏ/per-thanh/ma trận nhầm."""
    return {
        "n_total": 0, "n_scored": 0, "n_correct": 0,
        "n_not_scored": 0, "n_not_scored_empty_window": 0,
        "per_tone": {tone: {"n": 0, "correct": 0} for tone in TONES},
        "confusion": {true: {pred: 0 for pred in TONES} for true in TONES},
    }


def _add(bucket: dict, tone_true: str, pred: str | None) -> None:
    """Cộng một âm tiết vào bộ đếm (`pred=None` = bị lọc, không tính accuracy)."""
    bucket["n_total"] += 1
    if pred is None:
        bucket["n_not_scored"] += 1
        return
    bucket["n_scored"] += 1
    bucket["n_correct"] += int(pred == tone_true)
    bucket["per_tone"][tone_true]["n"] += 1
    bucket["per_tone"][tone_true]["correct"] += int(pred == tone_true)
    bucket["confusion"][tone_true][pred] += 1


def _finalize(bucket: dict) -> dict:
    """Thêm accuracy tổng/per-thanh (chỉ trên âm tiết đã chấm) + biến thể 'bỏ = sai' (như T3.3)."""
    for counts in bucket["per_tone"].values():
        counts["accuracy"] = round(counts["correct"] / counts["n"], 4) if counts["n"] else None
    bucket["accuracy"] = (round(bucket["n_correct"] / bucket["n_scored"], 4)
                          if bucket["n_scored"] else None)
    bucket["accuracy_drop_as_error"] = (round(bucket["n_correct"] / bucket["n_total"], 4)
                                        if bucket["n_total"] else None)
    return bucket


def _segment_follow(f0, vuv, reference_log: np.ndarray,
                    reference_mask: np.ndarray | None) -> float | None:
    """Bám contour ở thang **toàn segment** (đỡ nhiễu hơn cửa sổ ngắn): corr log-F0 harvest ↔ tham chiếu."""
    frames = min(int(f0.numel()), int(reference_log.size))
    if frames < 50:
        return None
    mask = vuv[:frames].numpy() > 0.5
    if reference_mask is not None:
        mask = mask & reference_mask[:frames]
    if int(mask.sum()) < 50:
        return None
    values = np.log(np.maximum(f0[:frames].numpy(), 1e-6))[mask]
    reference = reference_log[:frames][mask]
    return _corr(values, reference)


def _pairwise_agreement(records: list[dict]) -> dict:
    """Tỷ lệ 2 đường đoán cùng thanh (trên âm tiết cả hai đều chấm được)."""
    keys = ("A", "B", "C", "D")
    out: dict[str, dict] = {}
    for position, first in enumerate(keys):
        for second in keys[position + 1:]:
            pairs = [(entry[first]["pred"], entry[second]["pred"]) for entry in records
                     if entry[first]["pred"] and entry[second]["pred"]]
            agree = sum(1 for left, right in pairs if left == right)
            out[f"{first}{second}"] = {
                "n": len(pairs), "agree": agree,
                "rate": round(agree / len(pairs), 4) if pairs else None,
            }
    return out


def _round_or_none(values: np.ndarray | None) -> list[float] | None:
    """`s_k` z → list float gọn cho JSON."""
    return None if values is None else [round(float(v), 4) for v in values]


def run_alignment(rows_sample: list[dict], dataset, sample_rate: int, hop: int,
                  parent_wav) -> dict:
    """Cổng alignment: harvest f0 đoạn cắt vs f0 cache (chỉ frame cùng hữu thanh)."""
    from vifasttts.data.features import estimate_f0_pyworld

    entries: list[dict] = []
    correlations: list[float] = []
    for index in range(len(rows_sample)):
        row = dataset.rows[index]
        item = dataset[index]
        frame_range = row.get("frame_range")
        if not frame_range:
            raise SystemExit(f"Row {row.get('id')} thiếu frame_range — không cắt được audio thật")
        start, end = int(frame_range[0]), int(frame_range[1])
        segment = parent_wav(row["audio_path"])[start * hop:end * hop]
        f0, vuv = estimate_f0_pyworld(segment, sample_rate, hop, f_min=70, f_max=400,
                                      interpolate=False)
        cached_f0, cached_vuv = item["f0"], item["vuv"]
        frames = min(int(f0.numel()), int(cached_f0.numel()), end - start)
        both = (vuv[:frames] > 0.5) & (cached_vuv[:frames] > 0.5)
        count = int(both.sum())
        correlation = None
        if count >= 10:
            left = f0[:frames][both].numpy()
            right = cached_f0[:frames][both].numpy()
            if left.std() > 0 and right.std() > 0:
                correlation = float(np.corrcoef(left, right)[0, 1])
                correlations.append(correlation)
        entries.append({
            "id": row.get("id"), "frames": frames, "both_voiced": count,
            "corr": None if correlation is None else round(correlation, 4),
            "voiced_harvest": int((vuv[:frames] > 0.5).sum()),
            "voiced_cache": int((cached_vuv[:frames] > 0.5).sum()),
        })
    median = float(np.median(correlations)) if correlations else None
    return {
        "rows": entries, "n_rows": len(entries), "n_with_corr": len(correlations),
        "median_corr": None if median is None else round(median, 4),
        "min_corr": None if not correlations else round(float(min(correlations)), 4),
        "threshold": ALIGNMENT_MIN_CORR,
        "status": "ok" if (median is not None and median > ALIGNMENT_MIN_CORR) else "failed",
        "note": ("harvest lại trên đoạn audio đã cắt (manifest frame_range) vs f0 cache, "
                 "chỉ frame cùng hữu thanh (vuv>0.5); tương quan trung vị > 0,8 ⇒ alignment đúng"),
    }


def _conclusion_hint(paired: dict) -> str:
    """Gợi ý khâu mất thanh từ bảng paired (heuristic, ngưỡng `HINT_LOSS`)."""
    accuracy = {key: paired[key]["accuracy"] for key in ("A", "B", "C")}
    if paired["A"]["n_scored"] < 30 or any(value is None for value in accuracy.values()):
        return "chưa đủ dữ liệu paired để kết luận (n_scored < 30)"
    loss_b = accuracy["A"] - accuracy["B"]  # mel thật → Vocos
    loss_c = accuracy["B"] - accuracy["C"]  # mel model (free-run)
    if loss_b >= HINT_LOSS and loss_c >= HINT_LOSS:
        return (f"CẢ HAI khâu mất thanh: Vocos (A−B = {loss_b:.2f}) và mel model (B−C = {loss_c:.2f})")
    if loss_b >= HINT_LOSS:
        return f"Vocos là khâu mất contour (A−B = {loss_b:.2f} ≥ {HINT_LOSS}; B−C = {loss_c:.2f})"
    if loss_c >= HINT_LOSS:
        return f"mel của model là khâu mất contour (B−C = {loss_c:.2f} ≥ {HINT_LOSS}; A−B = {loss_b:.2f})"
    note = ""
    if accuracy["A"] < 0.5:
        note = f" — lưu ý trần A thấp ({accuracy['A']:.2f}): độ nhạy hữu hạn"
    return (f"không khâu nào mất ≥ {HINT_LOSS} (A−B = {loss_b:.2f}, B−C = {loss_c:.2f}) "
            f"— xem số chi tiết/per-thanh{note}")


def diagnose(args, config: dict) -> dict:
    """Cổng alignment rồi đo 4 đường trên `--clips` row đầu của manifest (không train)."""
    import torch

    from vifasttts.data.dataset_v2 import SlotDataset
    from vifasttts.data.features import load_wav
    from vifasttts.infer import Synthesizer

    hop = int(config["mel"]["hop_length"])
    sample_rate = int(config.get("sample_rate", 24000))
    device_t = torch.device(args.device)

    f0_stats = resolve_f0_stats(args.checkpoint, config, args.f0_stats)
    priors = tone_prior_z_from_stats(f0_stats)
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir)
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    if engine.arch != "v2_slot5":
        raise SystemExit(f"Tool này cần checkpoint v2 (arch=v2_slot5), gặp {engine.arch}")

    count = min(int(args.clips), len(dataset))
    rows_sample = dataset.rows[:count]
    parents: dict = {}

    def parent_wav(path: str):
        """Nạp WAV clip cha một lần (nhiều row dùng chung)."""
        if path not in parents:
            parents[path] = load_wav(path, sample_rate)
        return parents[path]

    started = time.perf_counter()
    alignment = run_alignment(rows_sample, dataset, sample_rate, hop, parent_wav)
    print(f"Alignment: corr trung vị {alignment['median_corr']} trên {alignment['n_with_corr']} row "
          f"(cổng > {ALIGNMENT_MIN_CORR}) → {alignment['status']}")
    if alignment["status"] != "ok":
        return {
            "status": "alignment_failed", "alignment": alignment,
            "hint": "Alignment/harvest sai — DỪNG, không chấm tiếp (xem alignment.rows).",
        }

    buckets = {key: _new_bucket() for key in ("A", "B", "C", "D")}
    paired = {key: _new_bucket() for key in ("A", "B", "C", "D")}
    records: list[dict] = []
    segment_follow: dict[str, list[float]] = {"B": [], "C": []}
    speed_t = torch.tensor([float(args.speed)], dtype=torch.float32, device=device_t)

    from vifasttts.data.features import estimate_f0_pyworld

    for index in range(count):
        row = dataset.rows[index]
        item = dataset[index]
        syllable_count = int(item["features"].size(0))
        tones = [TONES[int(v)] for v in item["features"][:, 4].tolist()]

        frame_range = row["frame_range"]
        start, end = int(frame_range[0]), int(frame_range[1])
        real_audio = parent_wav(row["audio_path"])[start * hop:end * hop]

        with torch.no_grad():
            synth_b = engine.vocoder(item["mel"].unsqueeze(0).to(device_t))[0].float().cpu()
            outputs = engine.model(item["features"].unsqueeze(0).to(device_t), speed=speed_t)
            synth_c = engine.vocoder(outputs["mel"][0].unsqueeze(0))[0].float().cpu()
            f0_head = outputs["f0"][0].cpu().numpy()
            slots_model = [int(v) for v in outputs["slot_durations"][0].cpu().tolist()]

        # Bám contour thang toàn segment (B: mel thật→Vocos vs f0 cache; C: mel model→Vocos vs head).
        real_f0_log = np.log(np.maximum(item["f0"].numpy(), 1e-6))
        f0_b, vuv_b = estimate_f0_pyworld(synth_b, sample_rate, hop, f_min=70, f_max=400,
                                          interpolate=False)
        f0_c, vuv_c = estimate_f0_pyworld(synth_c, sample_rate, hop, f_min=70, f_max=400,
                                          interpolate=False)
        cover_b = _segment_follow(f0_b, vuv_b, real_f0_log, item["vuv"].numpy() > 0.5)
        cover_c = _segment_follow(f0_c, vuv_c, f0_head, f0_head > 1.0)
        if cover_b is not None:
            segment_follow["B"].append(cover_b)
        if cover_c is not None:
            segment_follow["C"].append(cover_c)

        slots_real = _slot_list(item["seg_durations"].tolist(), item["gaps"].tolist(),
                                int(item["lead"]))
        for syllable in range(syllable_count):
            tone_true = tones[syllable]
            predictions: dict[str, str | None] = {}
            details: dict[str, dict] = {}
            for name, slots, audio in (("A", slots_real, real_audio),
                                       ("B", slots_real, synth_b),
                                       ("C", slots_model, synth_c)):
                window_start, window_span = target_frame_window(slots, syllable, args.window)
                lo = max(window_start, 0) * hop
                hi = min((window_start + window_span) * hop, audio.numel())
                info = _classify_audio_window(
                    audio[lo:hi], sample_rate, hop, f0_stats, priors, args.min_voiced)
                predictions[name] = info["pred"]
                details[name] = {"pred": info["pred"], "n_voiced": info["n_voiced"],
                                 "span": int(window_span),
                                 "s_k_z": _round_or_none(info["s_k_z"])}
                if info["pred"] is None and window_span <= 0:
                    buckets[name]["n_not_scored_empty_window"] += 1
                _add(buckets[name], tone_true, info["pred"])
                start_frame = max(window_start, 0)
                if name == "C":
                    reference = f0_head[start_frame:start_frame + window_span]
                else:  # A/B: tham chiếu = f0 THẬT trong cache trên đúng cửa sổ
                    reference = np.log(np.maximum(
                        item["f0"][start_frame:start_frame + window_span].numpy(), 1e-6))
                raw, shape, follow_n = _follow_head(info["f0_log"], info["frames"], reference)
                details[name].update({
                    "follow_corr": None if raw is None else round(raw, 3),
                    "follow_corr_shape": None if shape is None else round(shape, 3),
                    "follow_frames": follow_n,
                })
            window_start, window_span = target_frame_window(slots_model, syllable, args.window)
            pred_d, frames_d, s_k_z_d = _classify_model_f0(
                f0_head[window_start:window_start + window_span], f0_stats, priors)
            predictions["D"] = pred_d
            details["D"] = {"pred": pred_d, "n_voiced": frames_d, "span": int(window_span),
                            "s_k_z": _round_or_none(s_k_z_d)}
            if pred_d is None:
                buckets["D"]["n_not_scored_empty_window"] += 1
            _add(buckets["D"], tone_true, pred_d)

            if all(predictions[key] is not None for key in ("A", "B", "C")):
                for key in ("A", "B", "C", "D"):
                    _add(paired[key], tone_true, predictions[key])
            raw = (row["syllables"][syllable].get("raw")
                   if syllable < len(row["syllables"]) else None)
            records.append({"row": row.get("id"), "syllable": syllable, "raw": raw,
                            "tone": tone_true, **details})

    paired_final = {key: _finalize(bucket) for key, bucket in paired.items()}
    follow = {}
    for key in ("A", "B", "C"):
        values = [entry[key]["follow_corr"] for entry in records
                  if entry[key].get("follow_corr") is not None]
        shapes = [entry[key]["follow_corr_shape"] for entry in records
                  if entry[key].get("follow_corr_shape") is not None]
        follow[key] = {
            "n_windows": len(values),
            "median_corr": None if not values else round(float(np.median(values)), 4),
            "median_corr_shape": None if not shapes else round(float(np.median(shapes)), 4),
        }
    result = {
        "status": "ok",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "manifest": args.manifest, "checkpoint": args.checkpoint,
            "cache_dir": args.cache_dir, "f0_stats": args.f0_stats or "(checkpoint→config→mặc định)",
            "clips": count, "device": args.device, "speed": float(args.speed),
            "window": args.window, "min_voiced_frames": int(args.min_voiced),
        },
        "method": {
            "classifier": "tools/eval_tone_pairs.py (contour_to_sk_z → tone_prior_z gần nhất)",
            "f0_harvest": "_robust_f0 (harvest/dio/autocorr, chọn bộ nhiều frame hữu thanh nhất)",
            "window": ("nucleus (N slot) — khớp prior R5b/T3.2; A/B theo seg_durations của row, "
                       "C/D theo slot_durations của model"),
            "A": "audio thật (frame_range) → harvest F0 → classify (mốc trần)",
            "B": "mel thật (cache) → Vocos → harvest F0 → classify",
            "C": "features của row → model free-run (speed trong graph) → mel → Vocos → harvest F0",
            "D": "f0 head model (log-F0 tự nhiên → exp) trên cửa sổ model, KHÔNG lọc hữu thanh",
            "filter": f"chỉ chấm khi ≥ {args.min_voiced} frame hữu thanh (như T3.3)",
            "paired_abc": "giao A∩B∩C (cùng âm tiết) để so công bằng khi Vocos làm rụng hữu thanh",
            "accuracy_drop_as_error": ("accuracy khi âm tiết bị lọc tính là sai (đúng cách đếm của "
                                       "T3.3 — tiện so mốc 0,19–0,22 của run 2)"),
            "pairwise_agreement": "2 đường đoán cùng thanh; C dùng cửa sổ slot model nên không cùng cửa sổ A/B",
            "contour_follow": ("bám contour theo cửa sổ nucleus: A/B so f0 cache, C so f0 head; "
                               "trần của phép đo là A (cùng estimator trên cùng audio thật)"),
            "caveats": [
                "A/B dùng seg_durations đã đóng băng của manifest (nhãn yếu MFA-lai), C dùng slot model tự sinh.",
                "Câu là segment cắt từ clip cha: vài âm tiết đầu/cuối thiếu ngữ cảnh (không ảnh hưởng F0 đích).",
                ("D là thang đo THIÊN VỊ (f0 head sinh contour bằng chính 4 hệ số DCT mà bộ phân loại "
                 "dùng) → coi là trần, không phải bằng chứng độc lập."),
                ("A (trần của thước đo trên audio thật) chỉ ~0,45 trên câu chạy ⇒ độ nhạy hữu hạn: "
                 "chỉ kết luận được 'không mất đo được', không phải 'contour hoàn hảo'."),
            ],
        },
        "alignment": alignment,
        "paths": {key: _finalize(bucket) for key, bucket in buckets.items()},
        "paired_abc": paired_final,
        "pairwise_agreement": _pairwise_agreement(records),
        "contour_follow": {
            "A_vs_real_f0": follow["A"],
            "B_vs_real_f0": follow["B"],
            "C_vs_head": follow["C"],
            "segment_scale": {
                "B_vs_real_f0": None if not segment_follow["B"]
                else round(float(np.median(segment_follow["B"])), 4),
                "C_vs_head": None if not segment_follow["C"]
                else round(float(np.median(segment_follow["C"])), 4),
                "n_rows": len(segment_follow["B"]),
                "note": ("corr toàn segment (nhiều trăm frame) — A tham chiếu ở đây là "
                         "alignment.median_corr (harvest audio thật vs f0 cache)"),
            },
            "note": ("tương quan log-F0 harvest trong cửa sổ ↔ tham chiếu, trên cùng frame hữu thanh "
                     "(raw giữ mức, shape bỏ mức): A/B tham chiếu = f0 THẬT trong cache, C = f0 head "
                     "của model. A là trần của phép đo; B−A = mất do Vocos+harvest trên mel THẬT; "
                     "C−B = mất do mel của model."),
        },
        "conclusion_hint": _conclusion_hint(paired_final),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "records": records,
    }
    return result


def print_table(result: dict) -> None:
    """In bảng ngắn: 4 đường + bảng paired + gợi ý kết luận."""
    if result["status"] != "ok":
        print(f"status = {result['status']}: {result.get('hint', '')}")
        return
    header = " ".join(f"{tone[:4]:>4s}" for tone in TONES)
    print(f"\n{'Đường':22s} {'n':>5s} {'bỏ':>5s} {'acc':>6s} {'acc*':>6s}   per-tone ({header})")
    for key in ("A", "B", "C", "D"):
        entry = result["paths"][key]
        per_tone = " ".join(
            f"{entry['per_tone'][tone]['accuracy']:.3f}"
            if entry["per_tone"][tone]["accuracy"] is not None else "  -- "
            for tone in TONES)
        accuracy = f"{entry['accuracy']:.3f}" if entry["accuracy"] is not None else "  n/a"
        drop_error = (f"{entry['accuracy_drop_as_error']:.3f}"
                      if entry["accuracy_drop_as_error"] is not None else "  n/a")
        print(f"{PATH_LABELS[key]:22s} {entry['n_scored']:5d} {entry['n_not_scored']:5d} "
              f"{accuracy:>6s} {drop_error:>6s}   {per_tone}")
    print("acc* = tính âm tiết bị lọc là sai (cách đếm T3.3)")
    paired = result["paired_abc"]
    summary = " · ".join(
        f"{key}={paired[key]['accuracy']:.3f}" if paired[key]["accuracy"] is not None else f"{key}=n/a"
        for key in ("A", "B", "C", "D"))
    print(f"\nPaired A∩B∩C (n={paired['A']['n_scored']}): {summary}")
    follow = result["contour_follow"]
    print("Bám contour (median corr raw/shape, trên frame hữu thanh): "
          f"A↔f0 thật {follow['A_vs_real_f0']['median_corr']}/{follow['A_vs_real_f0']['median_corr_shape']}"
          f" · B↔f0 thật {follow['B_vs_real_f0']['median_corr']}/{follow['B_vs_real_f0']['median_corr_shape']}"
          f" · C↔head {follow['C_vs_head']['median_corr']}/{follow['C_vs_head']['median_corr_shape']}")
    scale = follow["segment_scale"]
    print(f"  (thang toàn segment: A {result['alignment']['median_corr']} · B {scale['B_vs_real_f0']}"
          f" · C {scale['C_vs_head']}, {scale['n_rows']} row)")
    print("Đồng thuận đoán cặp: " + " · ".join(
        f"{pair}={stats['rate']}" for pair, stats in result["pairwise_agreement"].items()))
    print(f"Gợi ý: {result['conclusion_hint']}")
    print(f"Thời gian chạy: {result['runtime_seconds']} s")


def main() -> None:
    """Đọc tham số, chẩn đoán, ghi JSON + in bảng; exit 2/3 khi trượt cổng."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/acoustic_v2_b8/resolved_config.json")
    parser.add_argument("--checkpoint", default="runs/acoustic_v2_b8/best.pt")
    parser.add_argument("--manifest", default="data/codelung/val_clean_v2_dur_seg_slot4.jsonl")
    parser.add_argument("--cache-dir", default="cache/codelung_v2")
    parser.add_argument("--f0-stats", default=None,
                        help="JSON f0_stats v2 (mặc định: checkpoint → config → repo)")
    parser.add_argument("--clips", type=int, default=60, help="số row đầu của manifest")
    parser.add_argument("--device", default="cpu", help="cpu (mặc định) | mps | cuda")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--window", choices=["nucleus", "syllable"], default="nucleus")
    parser.add_argument("--min-voiced", type=int, default=MIN_VOICED_FRAMES)
    parser.add_argument("--out", default="artifacts/tone_path/diagnosis.json")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = diagnose(args, config)

    if result["status"] == "ok":
        accuracy_a = result["paths"]["A"]["accuracy"]
        if accuracy_a is None or accuracy_a <= GATE_A_MIN_ACC:
            result["status"] = "gate_A_failed"
            result["hint"] = (f"Đường A chỉ {accuracy_a} ≤ {GATE_A_MIN_ACC} — alignment/harvest "
                              "nghi sai, phải rà lại trước khi báo cáo.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {out}")
    print_table(result)

    if result["status"] == "alignment_failed":
        sys.exit(2)
    if result["status"] == "gate_A_failed":
        sys.exit(3)


if __name__ == "__main__":
    main()

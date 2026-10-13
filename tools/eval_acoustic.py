"""tools/eval_acoustic.py — cổng nghiệm thu acoustic (T1.2 / B6).

Chế độ:
  tf_syl — cho `d_syl`/`gaps`/`lead` thật (v1: truyền `spans`); đo mel L1 (z/raw/dB),
           F0 RMSE theo 6 thanh (chỉ frame hữu thanh thật, mask `vuv`), VUV, energy,
           duration (từ `log_duration`).
  tf_seg — v2 (cho thêm `seg_durations`); v1: raise NotImplementedError.
  free   — không cho gì: CER (Whisper), RTF, RAM + **nhịp nghỉ** (`gap_sum_mae`,
           `pause_f1`, `mel_gap`) + **ngữ điệu** (`slope_all`, `slope_final_question`,
           `slope_final_period`, `ngang_declination_cents`) + `mel_gap` khi tf.

Mở rộng 14/10/2026 (kiểm toán "hiện tượng mù" — chỉ THÊM, mặc định không đổi):
  `--group-by coda,tone,…` — tách metric cấp âm tiết theo nhóm (khoá `groups` trong JSON);
    nhóm ghép `coda:tone` = tích hai nhóm (kiểm hiệu ứng coda có bị lẫn với thanh);
  `--no-cer` — bỏ CER ở free mode (khi chỉ cần nhịp nghỉ/ngữ điệu; CER vẫn là mặc định).

B1 (§43.2 — sửa THƯỚC ngữ điệu câu hỏi): `slope_final_question|period` **chỉ** tính khi dấu ở
  âm tiết **cuối** row (bản cũ = mọi vị trí, giữ dưới tên `*_any`); thêm **Δ ghép cặp** model−thật
  (`intonation_paired`: median + CI 95% bootstrap 1000 lần) và **Δ theo âm tiết cuối row**
  (`intonation_by_final` với nhóm `punctuation` / `tone` / `punctuation:tone` = `QUESTION|NGANG`).

Mask vùng đo lấy từ manifest (Addendum 3 §2.1): mel trên [lead, T); F0/VUV chỉ
frame hữu thanh của âm tiết (`durations`), bỏ gap + lead.

Ví dụ:
  python tools/eval_acoustic.py --config configs/small_codelung.yaml \
      --checkpoint runs/small_codelung_v2/last.pt --mode tf_syl \
      --manifest data/codelung/val_clean_v2_dur_seg.jsonl \
      --cache-dir cache/codelung_v2 --clips 300 \
      --out artifacts/eval_acoustic/b0_last20k_tf_syl.json

  python tools/eval_acoustic.py --config configs/small_codelung.yaml \
      --checkpoint runs/small_codelung_v2/last.pt --mode free \
      --manifest data/codelung/eval_clips_v1.jsonl --clips 20 \
      --out artifacts/eval_acoustic/b0_last20k_free.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TONES = ["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]
_VI_CHARS = "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"

#: Cột trong `features [N,7]` theo thứ tự `ComponentVocabs.encode`.
_FEATURE_COLUMNS = {"onset": 0, "medial": 1, "nucleus": 2, "coda": 3, "tone": 4,
                    "boundary": 5, "punctuation": 6}
#: Nhóm hợp lệ cho `--group-by`: 7 thành phần + `status` (nhãn trong manifest, nếu có).
GROUP_KEYS = (*_FEATURE_COLUMNS, "status")
#: Ngưỡng "đặt pause" — khớp `data/v2_targets.GATE_MIN_FRAMES` (gate BCE target `gap ≥ 3`).
PAUSE_GATE_MIN_FRAMES = 3
#: Số frame hữu thanh tối thiểu để tin một hồi quy độ dốc F0.
SLOPE_MIN_FRAMES = 8
#: Như trên nhưng cho hồi quy trên 2 âm tiết cuối (cửa sổ ngắn ⇒ cần nhiều frame hơn).
PAD_SLOPE_MIN_FRAMES = 12
#: Các metric ngữ điệu trong khoá `intonation` của JSON free-run.
#: `slope_final_*` chỉ tính khi dấu ở **âm tiết cuối row** (B1/§43.2); `*_any` = bản CŨ
#: (dấu ở bất kỳ vị trí — trộn câu hỏi GIỮA câu ⇒ đọc sai hướng ngữ điệu).
_SLOPE_METRICS = ("slope_all", "slope_all_ngang",
                  "slope_final_question", "slope_final_question_any",
                  "slope_final_period", "slope_final_period_any",
                  "ngang_declination_cents")
#: Nhóm để cắt Δ ghép cặp ngữ điệu theo **âm tiết cuối row** (B1) — cùng máy móc nhóm ghép
#: như `coda:tone` (`_group_labels`); nhãn `punctuation:tone` = `"QUESTION|NGANG"`
#: (nhãn là TÊN trong `ComponentVocabs`, không phải ký tự `?`).
_SLOPE_FINAL_GROUPS = ("punctuation", "tone", "punctuation:tone")
#: Bootstrap CI 95% cho Δ ghép cặp (median) + seed cố định ⇒ JSON tất định khi chạy lại.
PAIRED_BOOTSTRAP_DRAWS = 1000
PAIRED_BOOTSTRAP_SEED = 20261014
_GROUP_NOTE = ("metric cấp âm tiết (mỗi âm tiết 1 giá trị; mel/F0 trên frame của chính âm tiết "
               "trong timeline target) — mean/median trên các âm tiết của nhóm")
_GAP_NOTE = ("free-run: so theo vị trí frame (chưa align) ⇒ chỉ tin khi `length_ratio` ≈ 1; "
             "`pause_f1`/`gap_sum_mae_frames` **bỏ vị trí gap cuối row** (row cũ chỉ ghi N-1 gap, "
             "khớp `pause_mask` cũ) — manifest A2 ghi đủ gap ⇒ đọc `pause_f1_all`/`*_all_frames`")
_SLOPE_NOTE = ("logF0 hồi quy theo vị trí frame, chỉ frame hữu thanh (≥ 8 frame; 2 âm tiết cuối "
               "≥ 12) — đơn vị cent/giây; đọc **median** (mean bị vài cửa sổ dốc đứng kéo lệch); "
               "tag `true` = timeline thật, `model` = timeline model tự sinh. "
               "**B1/§43.2**: `slope_final_question|period` chỉ tính khi dấu ở **âm tiết CUỐI row**; "
               "bản cũ (mọi vị trí — trộn `?` giữa câu) giữ nguyên dưới tên `*_any`. "
               "Đọc Δ ở `intonation_paired` (ghép cặp model−thật, median + CI 95% bootstrap "
               "1000 lần) và `intonation_by_final` (Δ tách theo nhãn âm tiết cuối, gồm "
               "`punctuation:tone` = `QUESTION|NGANG`) — **đừng** so median rời (n nhỏ, mất cặp).")


def _read_rows(manifest: str | Path) -> list[dict]:
    """Đọc manifest JSONL."""
    return [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _quantiles(values: list[float]) -> dict[str, float]:
    """n / mean / median cho một danh sách số."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {"n": int(array.size), "mean": round(float(array.mean()), 4),
            "median": round(float(np.median(array)), 4)}


def _quantiles_std(values: list[float]) -> dict[str, float]:
    """Như `_quantiles` + `std` (mẫu) — chỉ dùng trong `groups` để tính SE = std/√n.

    Không đổi `_quantiles` (metric chính giữ nguyên JSON cũ).
    """
    stats = _quantiles(values)
    array = np.asarray(values, dtype=np.float64)
    stats["std"] = round(float(array.std(ddof=1)), 4) if array.size > 1 else None
    return stats


def _char_cer(reference: str, hypothesis: str) -> float:
    """CER cấp ký tự sau chuẩn hoá (bỏ dấu câu, thường hoá, gộp khoảng trắng)."""
    def clean(text: str) -> str:
        text = text.lower()
        text = re.sub(rf"[^\w\s{_VI_CHARS}]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    ref, hyp = clean(reference), clean(hypothesis)
    if not ref:
        return 0.0
    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, start=1):
        current = [i]
        for j, hyp_char in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1] / len(ref)


def _load_model(config: dict, checkpoint: str):
    """Nạp model acoustic từ checkpoint (hỗ trợ cả checkpoint `train` lẫn trần state)."""
    import torch

    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.models.vifasttts import ViFastTtsModel

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
             len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ViFastTtsModel(config, sizes)
    model.load_state_dict(state.get("model", state))
    model.eval()
    return model


def run_tf_syl(args, config: dict) -> dict:
    """Teacher-forced với d_syl/gaps thật: đo mel/F0/VUV/energy/duration."""
    import torch

    from vifasttts.data.dataset import TtsDataset

    dataset = TtsDataset(args.manifest, config, cache_dir=args.cache_dir)
    model = _load_model(config, args.checkpoint)
    mel_stats = json.loads(Path(config["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean = torch.tensor(mel_stats["mean"])
    std = torch.tensor(mel_stats["std"]).clamp_min(1e-4)
    hop = int(config["mel"]["hop_length"])
    sample_rate = int(config.get("sample_rate", 24000))
    frame_ms = hop / sample_rate * 1000.0

    tone_f0: dict[str, list[float]] = {tone: [] for tone in TONES}
    per_clip: list[dict] = []
    vuv_hits = vuv_total = 0
    energy_errors: list[float] = []
    duration_errors: list[float] = []

    for index in range(min(args.clips, len(dataset))):
        row = dataset.rows[index]
        item = dataset[index]
        with torch.no_grad():
            outputs = model(item["features"].unsqueeze(0), durations=item["spans"].unsqueeze(0))

        pred_mel, target_mel = outputs["mel"][0], item["mel"]
        frames = min(pred_mel.size(0), target_mel.size(0))
        l1_z = float((pred_mel[:frames] - target_mel[:frames]).abs().mean())
        raw_pred = pred_mel[:frames] * std + mean
        raw_target = target_mel[:frames] * std + mean
        l1_raw = float((raw_pred - raw_target).abs().mean())
        # Mel trên frame "có tiếng" (mean log-magnitude > -8) — tránh bị chi phối bởi frame lặng/floor.
        speech = raw_target.mean(dim=1) > -8.0
        speech_frac = float(speech.float().mean())
        speech_db = (float((raw_pred[speech] - raw_target[speech]).abs().mean() * 8.686)
                     if bool(speech.any()) else None)

        predicted_duration = torch.round((torch.exp(outputs["log_duration"][0]) - 1).clamp_min(1))
        duration_errors.extend((predicted_duration - item["durations"]).abs().tolist())

        predicted_vuv = (torch.sigmoid(outputs["vuv_logits"][0]) > 0.5).float()
        vuv_hits += int((predicted_vuv == item["vuv"]).sum())
        vuv_total += int(item["vuv"].numel())
        energy_errors.append(float((outputs["energy"][0] - item["energy"]).abs().mean()))

        extracted = dataset.cache.get(row) if dataset.cache is not None else None
        if extracted is not None:
            frame_range = row.get("frame_range")
            position = int(frame_range[0]) if frame_range else 0
            predicted_f0 = outputs["f0"][0].tolist()
            tones = [syllable["abstract"]["tone"] for syllable in row["syllables"]]
            for syllable_index, (duration, span) in enumerate(
                    zip(item["durations"].tolist(), item["spans"].tolist())):
                true_f0 = extracted["f0"][position:position + duration].numpy()
                true_vuv = extracted["vuv"][position:position + duration].numpy()
                voiced = true_vuv > 0.5
                if voiced.any():
                    ratio = np.maximum(predicted_f0[syllable_index], 1e-6) / np.maximum(
                        true_f0[voiced], 1e-6)
                    cents = 1200.0 * np.abs(np.log2(np.maximum(ratio, 1e-9)))
                    tone_f0[tones[syllable_index]].extend(cents.tolist())
                position += span

        per_clip.append({
            "id": row["id"], "frames": frames,
            "mel_l1_z": round(l1_z, 4), "mel_l1_raw": round(l1_raw, 4),
            "mel_mae_db": round(8.686 * l1_raw, 3),
            "mel_mae_db_speech": round(speech_db, 3) if speech_db is not None else None,
            "speech_frame_frac": round(speech_frac, 3),
            "duration_mae_frames": round(float((predicted_duration - item["durations"]).abs().mean()), 3),
        })

    summary = {
        "mode": "tf_syl", "checkpoint": args.checkpoint, "manifest": args.manifest,
        "clips": len(per_clip), "clip_ids": [entry["id"] for entry in per_clip],
        "mel_l1_z": _quantiles([entry["mel_l1_z"] for entry in per_clip]),
        "mel_l1_raw": _quantiles([entry["mel_l1_raw"] for entry in per_clip]),
        "mel_mae_db": _quantiles([entry["mel_mae_db"] for entry in per_clip]),
        "mel_mae_db_speech": _quantiles([entry["mel_mae_db_speech"] for entry in per_clip
                                         if entry["mel_mae_db_speech"] is not None]),
        "speech_frame_frac_mean": round(float(np.mean([entry["speech_frame_frac"]
                                                       for entry in per_clip])), 3),
        "f0_rmse_cents_by_tone": {tone: _quantiles(values) for tone, values in tone_f0.items()},
        "vuv_accuracy": round(vuv_hits / max(vuv_total, 1), 4),
        "energy_mae": round(float(np.mean(energy_errors)), 4) if energy_errors else None,
        "duration_mae_frames": _quantiles(duration_errors),
        "duration_mae_ms": round(float(np.mean(duration_errors)) * frame_ms, 2) if duration_errors else None,
        "per_clip": per_clip,
    }
    return summary


V2_DEFAULT_F0_STATS = "data/codelung/f0_stats_v2.json"
#: Ghi rõ cách chọn VUV v2: accuracy trên slot **active cấu trúc** (onset/medial/coda ≠ NONE,
#: nucleus luôn active) — đúng mask loss train (`slot_losses`: BCE `seg_vuv_logits` vs
#: `seg_vuv ≥ 0.5`); thêm `vuv_accuracy_nucleus` để so per-âm-tiết.
V2_VUV_NOTE = ("slot-active (mask loss train) vs seg_vuv>=0.5; "
               "vuv_accuracy_nucleus = riêng cột nucleus")
#: Energy v2: MAE giữa `seg_energy` thô của model và target pool — đúng phép so của loss
#: train (không phải thang tự nhiên như outputs["energy"]).
V2_ENERGY_NOTE = "MAE seg_energy (thang loss train) trên slot active cấu trúc"
#: F0 v2: model ra log-F0 tự nhiên → exp sang Hz rồi đo cents như v1.
V2_F0_NOTE = "exp(outputs f0 log) vs f0 Hz trên frame hữu thanh (vuv>0.5) trong durations"


def _checkpoint_arch(checkpoint: str) -> str:
    """Đọc `arch` từ checkpoint (thiếu → `v1_syl`, tương thích ngược)."""
    import torch

    from vifasttts.models.factory import arch_of_checkpoint

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return arch_of_checkpoint(state if isinstance(state, dict) else {})


def resolve_f0_stats(checkpoint: str, config: dict, cli_value: str | None = None) -> dict:
    """Ưu tiên: `--f0-stats` > `state['f0_stats']` > `config f0_stats_path` > mặc định repo."""
    import torch

    if cli_value:
        return json.loads(Path(cli_value).read_text(encoding="utf-8"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and state.get("f0_stats"):
        return dict(state["f0_stats"])
    for candidate in (config.get("f0_stats_path"), V2_DEFAULT_F0_STATS):
        if candidate and Path(candidate).exists():
            return json.loads(Path(candidate).read_text(encoding="utf-8"))
    raise SystemExit("v2 cần f0_stats: truyền --f0-stats hoặc để f0_stats trong checkpoint")


def load_eval_model(checkpoint: str, config: dict, *,
                    f0_stats: dict | None = None):
    """Nạp model theo `arch` trong checkpoint; v2 nạp thêm `f0_stats` + `eval()`."""
    import torch

    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.models.factory import arch_of_checkpoint, build_model

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda),
             len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = arch_of_checkpoint(state if isinstance(state, dict) else {})
    model = build_model(config, sizes, arch=arch)
    model.load_state_dict(state.get("model", state))
    if arch == "v2_slot5" and f0_stats is not None and hasattr(model, "load_f0_stats"):
        model.load_f0_stats(f0_stats)
    model.eval()
    return model, arch


def _v2_syllable_windows(durations: list[int], gaps: list[int], lead: int) -> list[tuple[int, int]]:
    """Cửa sổ `[start, end)` từng âm tiết trong feature đã cắt (bỏ gap/lead)."""
    windows: list[tuple[int, int]] = []
    cursor = int(lead)
    for index, duration in enumerate(durations):
        windows.append((cursor, cursor + int(duration)))
        cursor += int(duration) + (int(gaps[index]) if index < len(gaps) else 0)
    return windows


def _v2_structural_active(features) -> list[list[bool]]:
    """Mask slot active cấu trúc theo id (NONE = 0; nucleus luôn active) — mask loss train."""
    feats = features.tolist()
    return [[bool(row[0]), bool(row[1]), True, bool(row[3])] for row in feats]


# ── nhóm theo thành phần/nhãn (`--group-by`) ──────────────────────────────────
def _vocab_decoders(vocabs) -> dict[str, dict[int, str]]:
    """{tên thành phần: {id: tên giá trị}} để giải ID trong cột `features`."""
    return {name: {value: key for key, value in getattr(vocabs, name).items()}
            for name in _FEATURE_COLUMNS}


def _group_labels(row: dict, features, vocabs, group_by: list[str]) -> tuple[dict[str, list[str]],
                                                                            list[str]]:
    """Nhãn nhóm của từng âm tiết + danh sách nhóm **bị bỏ** (manifest thiếu dữ liệu).

    Thành phần (`onset`…`punctuation`) giải ID trong `features` qua `ComponentVocabs`;
    `status` đọc `row["syllables"][i]["status"]` — manifest không ghi khoá này thì bỏ nhóm
    (ghi vào `groups_skipped`, không bịa nhãn). Nhóm **ghép** dạng `coda:tone` = tích hai nhóm
    (nhãn `"T|SAC"`) — để kiểm tra hiệu ứng coda có bị lẫn với thanh (sắc/nặng mới có coda tắc).
    """
    plain: list[str] = []
    for group in group_by:
        for part in (group.split(":") if ":" in group else [group]):
            if part and part not in plain:
                plain.append(part)
    labels, skipped = _plain_group_labels(row, features, vocabs, plain)
    count = int(features.size(0))
    for group in group_by:
        if ":" not in group:
            continue
        parts = [part for part in group.split(":") if part]
        if any(part in skipped for part in parts):
            skipped.append(group)
            continue
        labels[group] = ["|".join(labels[part][index] for part in parts) for index in range(count)]
    return labels, sorted(set(skipped))


def _plain_group_labels(row: dict, features, vocabs,
                        group_by: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Nhãn của từng nhóm **đơn** (không ghép) — xem `_group_labels`."""
    decoders = _vocab_decoders(vocabs)
    labels: dict[str, list[str]] = {}
    skipped: list[str] = []
    count = int(features.size(0))
    for group in group_by:
        if group in decoders:
            table, column = decoders[group], _FEATURE_COLUMNS[group]
            labels[group] = [table.get(int(features[index][column]),
                                       f"id={int(features[index][column])}")
                             for index in range(count)]
        elif group == "status":
            values = [str(syllable["status"]) for syllable in (row.get("syllables") or [])
                      if syllable.get("status") is not None]
            if len(values) == count:
                labels[group] = values
            else:
                skipped.append(group)
        else:
            raise ValueError(f"--group-by không hỗ trợ {group!r} (chọn trong {GROUP_KEYS})")
    return labels, skipped


def _group_add(store: dict, group: str, label: str, metric: str, value: float) -> None:
    """Cộng một giá trị metric cho (nhóm, giá trị nhóm)."""
    store.setdefault(group, {}).setdefault(label, {}).setdefault(metric, []).append(float(value))


def _group_vuv_add(store: dict, group: str, label: str, hits: int, total: int) -> None:
    """Cộng hit/total VUV cho (nhóm, giá trị nhóm) — accuracy gộp theo âm tiết."""
    entry = store.setdefault(group, {}).setdefault(label, {"hits": 0, "total": 0})
    entry["hits"] += int(hits)
    entry["total"] += int(total)


def _group_report(store: dict, vuv_store: dict, group_by: list[str]) -> dict:
    """{nhóm: {giá trị: {n_syllables, mel_z, mel_db, dur_mae, f0_cents, vuv_acc}}}.

    Mỗi metric là quantiles (n/mean/median) trên các **âm tiết** thuộc giá trị đó; `n_syllables`
    để thấy nhóm nào ít mẫu (cẩn thận khi kết luận). Nhóm không có mẫu nào ⇒ không xuất hiện
    (người gọi ghi vào `groups_skipped`).
    """
    report: dict[str, dict] = {}
    for group in group_by:
        values: dict[str, dict] = {}
        metrics_by_label = store.get(group, {})
        for label in sorted(metrics_by_label):
            metrics = metrics_by_label[label]
            vuv = vuv_store.get(group, {}).get(label) or {}
            values[label] = {
                "n_syllables": len(metrics.get("mel_z", [])),
                "mel_z": _quantiles_std(metrics.get("mel_z", [])),
                "mel_db": _quantiles_std(metrics.get("mel_db", [])),
                "dur_mae": _quantiles_std(metrics.get("dur_mae", [])),
                "f0_cents": _quantiles_std(metrics.get("f0_cents", [])),
                "vuv_acc": (round(vuv["hits"] / vuv["total"], 4) if vuv.get("total") else None),
            }
        if values:
            report[group] = values
    return report


# ── nhịp nghỉ (free-run) ──────────────────────────────────────────────────────
def _gap_frame_mask_from(durations, gaps, lead: int, frames: int) -> np.ndarray:
    """Mask [frames] True ở frame thuộc `G_lead` hoặc slot G (gap) — như `train_v2._gap_frame_mask`."""
    mask = np.zeros(int(frames), dtype=bool)
    mask[:min(int(lead), int(frames))] = True
    cursor = int(lead)
    for index, duration in enumerate(durations):
        cursor += int(duration)
        gap = int(gaps[index]) if index < len(gaps) else 0
        end = min(cursor + gap, int(frames))
        if end > cursor:
            mask[cursor:end] = True
        cursor += gap
    return mask


def _pause_counts(pred_pause, true_pause) -> tuple[int, int, int]:
    """(TP, FP, FN) của "đặt pause" trên các vị trí đã mask."""
    hits = int((pred_pause & true_pause).sum())
    false = int((pred_pause & ~true_pause).sum())
    missed = int((~pred_pause & true_pause).sum())
    return hits, false, missed


def _f1(hits: int, false: int, missed: int) -> float | None:
    """F1 = 2TP / (2TP + FP + FN); None khi không có dương tính nào (thật lẫn dự đoán)."""
    denominator = 2 * hits + false + missed
    return round(2 * hits / denominator, 4) if denominator else None


def _paired_delta(true_values: list[float], model_values: list[float], *,
                  draws: int = PAIRED_BOOTSTRAP_DRAWS,
                  seed: int = PAIRED_BOOTSTRAP_SEED) -> dict | None:
    """Δ **ghép cặp** (model − thật) của một metric: median + CI 95% bootstrap.

    Ghép cặp = chỉ dùng các **cặp cùng cửa sổ** (một clip/metric một cặp) — bắt buộc khi so
    thật vs model (B1/§43.2: so median rời nhau vừa lệch cỡ mẫu vừa trộn cửa sổ khác nhau ⇒ đọc
    sai dấu của hiệu ứng). CI = percentile 2,5/97,5 của **median** Δ khi resample có hoàn lại
    (`draws` lần, seed cố định ⇒ chạy lại ra cùng JSON). `None` khi không có cặp nào (không bịa số);
    `ci95 = None` khi n = 1 (không bootstrap được).
    """
    true_array = np.asarray(true_values, dtype=np.float64)
    model_array = np.asarray(model_values, dtype=np.float64)
    if true_array.size == 0 or true_array.size != model_array.size:
        return None
    delta = model_array - true_array
    entry: dict = {
        "n": int(delta.size),
        "median_true": round(float(np.median(true_array)), 3),
        "median_model": round(float(np.median(model_array)), 3),
        "delta_median": round(float(np.median(delta)), 3),
        "delta_mean": round(float(delta.mean()), 3),
        "frac_rising": round(float((delta > 0).mean()), 3),  # tỷ lệ cửa sổ model dốc LÊN hơn thật
    }
    if delta.size > 1 and int(draws) > 0:
        rng = np.random.default_rng(seed)
        index = rng.integers(0, delta.size, size=(int(draws), delta.size))
        medians = np.median(delta[index], axis=1)
        entry["ci95"] = [round(float(np.percentile(medians, 2.5)), 3),
                         round(float(np.percentile(medians, 97.5)), 3)]
    else:
        entry["ci95"] = None
    return entry


def _paired_report(pairs: dict[str, list[list[float]]]) -> dict[str, dict]:
    """{metric: `_paired_delta`} cho các cặp `[[true, model], …]` (bỏ metric không có cặp)."""
    report: dict[str, dict] = {}
    for metric, values in sorted(pairs.items()):
        entry = _paired_delta([pair[0] for pair in values], [pair[1] for pair in values])
        if entry is not None:
            report[metric] = entry
    return report


def _paired_report_nested(pairs: dict[str, dict[str, dict[str, list[list[float]]]]]) -> dict:
    """`{nhóm: {nhãn: {metric: Δ ghép cặp}}}` — bỏ nhãn/metric không có cặp (không bịa số)."""
    report: dict = {}
    for group, by_label in pairs.items():
        labels = {label: _paired_report(metrics) for label, metrics in by_label.items()}
        labels = {label: metrics for label, metrics in labels.items() if metrics}
        if labels:
            report[group] = labels
    return report


def _pred_gap_slots(slot_durations) -> np.ndarray:
    """Gap model tự dự đoán: `slot_durations [1+5N]` → [N] (slot G của từng âm tiết)."""
    return slot_durations[5::5].detach().cpu().numpy() if hasattr(slot_durations, "detach") \
        else np.asarray(slot_durations[5::5])


def _slot_to_frames(slot_values, slot_durations, frames: int) -> np.ndarray:
    """Rải giá trị cấp slot ra cấp frame theo `slot_durations` (cùng mẹo cumsum của LengthRegulator)."""
    durations = np.asarray(slot_durations, dtype=np.int64)
    cumulative = np.cumsum(durations)
    steps = np.arange(int(frames))
    index = (cumulative[:, None] <= steps[None, :]).sum(axis=0)
    index = np.clip(index, 0, durations.size - 1)
    return np.asarray(slot_values)[index]


# ── ngữ điệu / declination ────────────────────────────────────────────────────
def _log_f0_slope_cents_per_second(positions: np.ndarray, log_f0: np.ndarray,
                                   frame_rate: float,
                                   min_frames: int = SLOPE_MIN_FRAMES) -> float | None:
    """Độ dốc logF0 theo frame (hồi quy bình phương nhỏ nhất) → cent/giây; None nếu quá ít frame.

    `min_frames` nâng lên cho cửa sổ ngắn (2 âm tiết cuối) vì hồi quy trên ~8 frame bị nhiễu
    chi phối (đo thật: mean nhảy ±300 cent/giây giữa các cửa sổ).
    """
    if len(positions) < max(int(min_frames), 2):
        return None
    x = np.asarray(positions, dtype=np.float64)
    y = np.asarray(log_f0, dtype=np.float64)
    centred = x - x.mean()
    denominator = float((centred ** 2).sum())
    if denominator <= 1e-6:
        return None
    slope = float((centred * (y - y.mean())).sum()) / denominator
    return 1200.0 / math.log(2.0) * slope * float(frame_rate)


def _slope_block(true_series: tuple[np.ndarray, np.ndarray],
                 pred_series: tuple[np.ndarray, np.ndarray],
                 windows_true: list[tuple[int, int]], windows_pred: list[tuple[int, int]],
                 tones: list[str], punctuation_ids: list[int], punct_codes: dict[str, int],
                 frame_rate: float) -> dict[str, dict[str, float | None]]:
    """Bộ số ngữ điệu một clip: `{metric: {"true": …, "model": …}}`.

    - `slope_all` — dốc logF0 toàn câu (cent/giây, chỉ frame hữu thanh);
    - `slope_all_ngang` — như trên nhưng chỉ frame thuộc âm tiết thanh NGANG;
    - `slope_final_question` / `slope_final_period` — dốc trên **2 âm tiết cuối** trước `?`/`.`
      (nối frame của hai âm tiết trong timeline tương ứng) — **chỉ khi dấu ở âm tiết CUỐI row**
      (B1/§43.2: trước đây cộng cửa sổ của `?` ở bất kỳ vị trí ⇒ 12/16 cửa sổ val300 là câu hỏi
      giữa câu, đọc sai hướng); bản cũ giữ ở `slope_final_*_any`;
    - `ngang_declination_cents` — chênh logF0 trung bình âm tiết NGANG nửa đầu − nửa cuối câu.

    `*_series` = `(vị trí frame, logF0)` chỉ frame **hữu thanh** của timeline tương ứng;
    `windows_*` = cửa sổ frame từng âm tiết của timeline đó (`windows_true` == `windows_pred`
    khi model được cho thời lượng thật — tf mode).
    """
    block: dict[str, dict[str, float | None]] = {}
    for tag, series, windows in (("true", true_series, windows_true),
                                 ("model", pred_series, windows_pred)):
        metric_values: dict[str, float | None] = {}
        positions, values = series
        metric_values["slope_all"] = _log_f0_slope_cents_per_second(positions, values, frame_rate)
        ngang = [index for index, tone in enumerate(tones) if tone == "NGANG"]
        positions_ngang, values_ngang = _subset_series(
            series, [windows[index] for index in ngang if index < len(windows)])
        metric_values["slope_all_ngang"] = _log_f0_slope_cents_per_second(
            positions_ngang, values_ngang, frame_rate)
        for mark in ("QUESTION", "PERIOD"):
            code = punct_codes.get(mark)
            chosen: list[tuple[int, int]] = []      # B1: CHỈ dấu ở âm tiết cuối row
            chosen_any: list[tuple[int, int]] = []  # bản cũ: mọi vị trí (giữ để so lịch sử)
            for index, punct in enumerate(punctuation_ids):
                if punct != code or index >= len(windows):
                    continue
                pair = [windows[max(index - 1, 0)], windows[index]]  # hai âm tiết cuối trước dấu
                chosen_any.extend(pair)
                if index == len(punctuation_ids) - 1:
                    chosen.extend(pair)
            positions_mark, values_mark = _subset_series(series, chosen)
            metric_values[f"slope_final_{mark.lower()}"] = _log_f0_slope_cents_per_second(
                positions_mark, values_mark, frame_rate, min_frames=PAD_SLOPE_MIN_FRAMES)
            positions_any, values_any = _subset_series(series, chosen_any)
            metric_values[f"slope_final_{mark.lower()}_any"] = _log_f0_slope_cents_per_second(
                positions_any, values_any, frame_rate, min_frames=PAD_SLOPE_MIN_FRAMES)
        metric_values["ngang_declination_cents"] = _ngang_declination(series, windows, tones)
        for metric, value in metric_values.items():
            block.setdefault(metric, {})[tag] = value
    return block


def _subset_series(series: tuple[np.ndarray, np.ndarray],
                   windows: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Lọc `(vị trí, logF0)` theo danh sách cửa sổ `[low, high)`."""
    positions, values = series
    keep = np.zeros(positions.shape, dtype=bool)
    for low, high in windows:
        keep |= (positions >= low) & (positions < high)
    return positions[keep], values[keep]


def _ngang_declination(series: tuple[np.ndarray, np.ndarray],
                       windows: list[tuple[int, int]], tones: list[str]) -> float | None:
    """Chênh logF0 trung bình của âm tiết NGANG **nửa đầu − nửa cuối** câu (cent).

    Dương = NGANG đầu câu cao hơn cuối câu (declination); ≈ 0 = contour ngang không trôi.
    Cần ≥ 4 âm tiết NGANG có ≥ 2 frame hữu thanh để tin (chia đôi theo thứ tự âm tiết NGANG).
    """
    ngang_indices = [index for index, tone in enumerate(tones) if tone == "NGANG"]
    means: list[float] = []
    for index in ngang_indices:
        if index >= len(windows):
            continue
        _, values = _subset_series(series, [windows[index]])
        if values.size >= 2:
            means.append(float(values.mean()))
    if len(means) < 4:
        return None
    middle = len(means) // 2
    first, second = means[:middle], means[middle:]
    if len(first) < 2 or len(second) < 2:
        return None
    return round(1200.0 / math.log(2.0) * (float(np.mean(first)) - float(np.mean(second))), 2)


def evaluate_tf_v2(model, dataset, config: dict, *, use_seg: bool,
                   device: str = "cpu", limit: int | None = None,
                   group_by: list[str] | None = None) -> dict:
    """Teacher-forced v2 thuần (test gọi trực tiếp): `use_seg` = cho `seg_durations` (tf_seg).

    Đo đúng key v1 (mel L1 z/raw/dB + speech, F0 cents theo 6 thanh, VUV, energy,
    duration) + `arch` và `slot_sum_mae_frames` (B2 tự chia ở tf_syl). Thêm `mel_gap_z`/
    `mel_gap_db` (L1 mel **chỉ frame gap/lead thật** — như `train_v2._mel_split`).
    `group_by` (vd `["coda", "tone"]`) → thêm khoá `groups`: cùng bộ metric nhưng **cấp âm tiết**,
    tách theo giá trị của thành phần trong `features`/nhãn `status` của manifest.
    """
    import torch

    mode = "tf_seg" if use_seg else "tf_syl"
    device_t = torch.device(device)
    model.to(device_t)
    model.eval()
    mel_stats = json.loads(Path(config["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean = torch.tensor(mel_stats["mean"])
    std = torch.tensor(mel_stats["std"]).clamp_min(1e-4)
    hop = int(config["mel"]["hop_length"])
    sample_rate = int(config.get("sample_rate", 24000))
    frame_ms = hop / sample_rate * 1000.0

    group_by = list(group_by or [])
    vocabs = getattr(dataset, "vocabs", None)
    if group_by and vocabs is None:
        from vifasttts.frontend.vocab import ComponentVocabs

        vocabs = ComponentVocabs.default()
    group_store: dict = {}
    group_vuv: dict = {}
    groups_skipped: list[str] = []

    tone_f0: dict[str, list[float]] = {tone: [] for tone in TONES}
    per_clip: list[dict] = []
    vuv_hits = vuv_total = 0
    vuv_nuc_hits = vuv_nuc_total = 0
    energy_errors: list[float] = []
    duration_errors: list[float] = []
    slot_sum_errors: list[float] = []
    gap_z_errors: list[float] = []
    gap_db_errors: list[float] = []
    count = len(dataset) if limit is None else min(limit, len(dataset))

    for index in range(count):
        row = dataset.rows[index]
        item = dataset[index]
        features = item["features"].unsqueeze(0).to(device_t)
        durations = item["durations"].unsqueeze(0).to(device_t)
        gaps = item["gaps"].unsqueeze(0).to(device_t)
        lead = item["lead"].reshape(1).to(device_t)
        seg = item["seg_durations"].unsqueeze(0).to(device_t) if use_seg else None
        with torch.no_grad():
            outputs = model(features, durations=durations, seg_durations=seg,
                            gaps=gaps, lead=lead)

        pred_mel = outputs["mel"][0].cpu()
        target_mel = item["mel"]
        frames = min(pred_mel.size(0), target_mel.size(0))
        lead_frames = int(item["lead"])
        lo = min(lead_frames, frames)  # mel đo trên [lead, T) — bỏ vùng lead
        l1_z = float((pred_mel[:frames] - target_mel[:frames]).abs().mean())
        raw_pred = pred_mel[:frames] * std + mean
        raw_target = target_mel[:frames] * std + mean
        l1_raw = float((raw_pred - raw_target).abs().mean())
        speech = raw_target.mean(dim=1) > -8.0
        speech[0:lo] = False
        speech_frac = float(speech.float().mean())
        speech_db = (float((raw_pred[speech] - raw_target[speech]).abs().mean() * 8.686)
                     if bool(speech.any()) else None)

        predicted_duration = torch.round((torch.exp(outputs["log_duration"][0].cpu()) - 1).clamp_min(1))
        duration_errors.extend((predicted_duration - item["durations"]).abs().tolist())

        # VUV/energy trên slot active cấu trúc (mask loss train).
        active = _v2_structural_active(item["features"])
        active_t = torch.tensor(active)
        pred_vuv = (torch.sigmoid(outputs["seg_vuv_logits"][0].cpu()) > 0.5)
        true_vuv = item["seg_vuv"] >= 0.5
        vuv_hits += int((pred_vuv[active_t] == true_vuv[active_t]).sum())
        vuv_total += int(active_t.sum())
        vuv_nuc_hits += int((pred_vuv[:, 2] == true_vuv[:, 2]).sum())
        vuv_nuc_total += int(pred_vuv.size(0))
        energy_errors.append(
            float((outputs["seg_energy"][0].cpu() - item["seg_energy"]).abs()[active_t].mean()))

        # F0: exp log → Hz, đo cents trên frame hữu thanh trong cửa sổ âm tiết.
        pred_log = outputs["f0"][0].cpu().clamp(-1.0, 8.0)
        pred_hz = torch.exp(pred_log)
        true_f0, true_vuv_frame = item["f0"], item["vuv"]
        windows = _v2_syllable_windows(item["durations"].tolist(), item["gaps"].tolist(),
                                       lead_frames)
        tones = [syllable["abstract"]["tone"] for syllable in row["syllables"]]
        labels: dict[str, list[str]] = {}
        if group_by:
            labels, row_skipped = _group_labels(row, item["features"], vocabs, group_by)
            groups_skipped.extend(row_skipped)
        for syllable_index, (low, high) in enumerate(windows):
            low_c = max(low, lo)
            high_c = min(high, pred_hz.size(0), true_f0.size(0))
            if group_by and syllable_index < len(item["durations"]) and high_c > low_c:
                for group, values in labels.items():
                    label = values[syllable_index]
                    _group_add(group_store, group, label, "mel_z", float(
                        (pred_mel[low_c:high_c] - target_mel[low_c:high_c]).abs().mean()))
                    window_speech = speech[low_c:high_c]
                    if bool(window_speech.any()):
                        _group_add(group_store, group, label, "mel_db", float(
                            (raw_pred[low_c:high_c][window_speech]
                             - raw_target[low_c:high_c][window_speech]).abs().mean() * 8.686))
                    _group_add(group_store, group, label, "dur_mae", abs(float(
                        predicted_duration[syllable_index] - item["durations"][syllable_index])))
                    active_syllable = torch.tensor(
                        _v2_structural_active(item["features"][syllable_index:syllable_index + 1]))[0]
                    _group_vuv_add(group_vuv, group, label, int((pred_vuv[syllable_index]
                                                                 == true_vuv[syllable_index]
                                                                 )[active_syllable].sum()),
                                   int(active_syllable.sum()))
            if high_c <= low_c or syllable_index >= len(tones):
                continue
            voiced = true_vuv_frame[low_c:high_c] > 0.5
            if bool(voiced.any()):
                ratio = torch.clamp(pred_hz[low_c:high_c][voiced], min=1e-6) / torch.clamp(
                    true_f0[low_c:high_c][voiced], min=1e-6)
                cents = 1200.0 * torch.abs(torch.log2(torch.clamp(ratio, min=1e-9)))
                tone_f0[tones[syllable_index]].extend(cents.tolist())
                if group_by:
                    for group, values in labels.items():
                        _group_add(group_store, group, values[syllable_index], "f0_cents",
                                   float(cents.mean()))

        # Mel trên frame gap/lead thật (như `train_v2._mel_split`): model có "im lặng" đúng chỗ?
        gap_mask = _gap_frame_mask_from(item["durations"].tolist(), item["gaps"].tolist(),
                                        lead_frames, frames)
        gap_valid = gap_mask & (np.arange(frames) >= lo)
        gap_z = gap_db = None
        if bool(gap_valid.any()):
            gap_error = (pred_mel[:frames] - target_mel[:frames]).abs().mean(dim=-1)[gap_valid]
            gap_z = float(gap_error.mean())
            gap_db = float((raw_pred[:frames] - raw_target[:frames]).abs().mean(dim=-1)[gap_valid].mean() * 8.686)
            gap_z_errors.append(gap_z)
            gap_db_errors.append(gap_db)

        # Tổng slot model sinh ra so Σ thật (tf_syl: đo sai số B2; tf_seg: ≈ 0).
        true_total = lead_frames + sum(item["durations"].tolist()) + sum(item["gaps"].tolist())
        slot_sum_errors.append(abs(float(outputs["slot_durations"][0].cpu().sum()) - true_total))

        per_clip.append({
            "id": row["id"], "frames": frames,
            "mel_l1_z": round(l1_z, 4), "mel_l1_raw": round(l1_raw, 4),
            "mel_mae_db": round(8.686 * l1_raw, 3),
            "mel_mae_db_speech": round(speech_db, 3) if speech_db is not None else None,
            "speech_frame_frac": round(speech_frac, 3),
            "duration_mae_frames": round(float((predicted_duration - item["durations"]).abs().mean()), 3),
            "slot_sum_err_frames": round(slot_sum_errors[-1], 3),
            "mel_gap_z": round(gap_z, 4) if gap_z is not None else None,
            "mel_gap_db": round(gap_db, 3) if gap_db is not None else None,
        })

    summary = {
        "mode": mode, "arch": "v2_slot5",
        "clips": len(per_clip), "clip_ids": [entry["id"] for entry in per_clip],
        "mel_l1_z": _quantiles([entry["mel_l1_z"] for entry in per_clip]),
        "mel_l1_raw": _quantiles([entry["mel_l1_raw"] for entry in per_clip]),
        "mel_mae_db": _quantiles([entry["mel_mae_db"] for entry in per_clip]),
        "mel_mae_db_speech": _quantiles([entry["mel_mae_db_speech"] for entry in per_clip
                                         if entry["mel_mae_db_speech"] is not None]),
        "speech_frame_frac_mean": round(float(np.mean([entry["speech_frame_frac"]
                                                       for entry in per_clip])), 3),
        "f0_rmse_cents_by_tone": {tone: _quantiles(values) for tone, values in tone_f0.items()},
        "f0_note": V2_F0_NOTE,
        "vuv_accuracy": round(vuv_hits / max(vuv_total, 1), 4),
        "vuv_accuracy_nucleus": round(vuv_nuc_hits / max(vuv_nuc_total, 1), 4),
        "vuv_note": V2_VUV_NOTE,
        "energy_mae": round(float(np.mean(energy_errors)), 4) if energy_errors else None,
        "energy_note": V2_ENERGY_NOTE,
        "duration_mae_frames": _quantiles(duration_errors),
        "duration_mae_ms": round(float(np.mean(duration_errors)) * frame_ms, 2) if duration_errors else None,
        "slot_sum_mae_frames": round(float(np.mean(slot_sum_errors)), 3) if slot_sum_errors else None,
        "mel_gap_z": _quantiles(gap_z_errors),
        "mel_gap_db": _quantiles(gap_db_errors),
        "mel_gap_note": ("L1 mel CHỈ frame gap/lead THẬT (như `train_v2._mel_split`) — "
                         "model được cho gap thật ở chế độ tf nên đây là 'im lặng đúng chỗ'"),
        "per_clip": per_clip,
    }
    if group_by:
        summary["groups"] = _group_report(group_store, group_vuv, group_by)
        summary["groups_by"] = group_by
        summary["groups_note"] = _GROUP_NOTE
        absent = [group for group in sorted(set(groups_skipped)) if group not in summary["groups"]]
        partial = [group for group in sorted(set(groups_skipped)) if group in summary["groups"]]
        if absent:
            summary["groups_skipped"] = absent  # manifest KHÔNG có dữ liệu ⇒ bỏ nhóm, không bịa
        if partial:
            summary["groups_partial"] = partial  # một phần row thiếu ⇒ chỉ gộp row có nhãn
    return summary


def evaluate_free_v2(model, dataset, config: dict, *, device: str = "cpu",
                     limit: int | None = None, speed: float = 1.0,
                     wav_dir: str | Path | None = None, group_by: list[str] | None = None,
                     skip_cer: bool = False) -> dict:
    """Free-run v2 thuần: model tự dự đoán durations/slot (không cho gì ngoài features).

    Mel/F0 đo trên vùng overlap với target thật + `length_ratio`; CER chấm qua
    waveform (tổng hợp text bằng chính model + vocoder dựng từ config — cùng cơ
    chế `Synthesizer.synthesize`/`_transcribe`/`_char_cer` của free v1).
    `cer=None` + `cer_note` khi thiếu `text` trong manifest, row không parse được,
    hoặc thiếu faster-whisper (không crash). RTF chỉ tính forward acoustic.

    Thêm (kiểm toán hiện tượng mù): **nhịp nghỉ** (`gap_sum_mae*`, `pause_f1*`, `mel_gap`) —
    so gap model tự sinh (`pause_gate`/`slot_durations`) với gap thật; **ngữ điệu** (`intonation`)
    — dốc logF0 toàn câu/2 âm tiết cuối + declination thanh NGANG, thật vs model;
    `group_by` → khoá `groups` (metric cấp âm tiết theo nhóm, xem `_group_report`).
    **B1** (sửa thước `?`): `slope_final_*` chỉ khi dấu ở âm tiết cuối row + khoá ghép cặp
    `intonation_paired` / `intonation_by_final` (Δ model−thật, median + CI 95% bootstrap).
    `skip_cer=True` bỏ hẳn phần CER (chạy nhanh; mặc định vẫn chấm).
    """
    import resource
    import tempfile
    import time

    import torch

    device_t = torch.device(device)
    model.to(device_t)
    model.eval()
    mel_stats = json.loads(Path(config["mel"]["normalize_path"]).read_text(encoding="utf-8"))
    mean = torch.tensor(mel_stats["mean"])
    std = torch.tensor(mel_stats["std"]).clamp_min(1e-4)
    hop = int(config["mel"]["hop_length"])
    sample_rate = int(config.get("sample_rate", 24000))
    speed_t = torch.tensor([float(speed)], dtype=torch.float32)
    frame_rate = sample_rate / float(hop)

    group_by = list(group_by or [])
    vocabs = getattr(dataset, "vocabs", None)
    if vocabs is None:
        from vifasttts.frontend.vocab import ComponentVocabs

        vocabs = ComponentVocabs.default()
    punct_codes = {mark: vocabs.punctuation.get(mark) for mark in ("QUESTION", "PERIOD")}
    group_store: dict = {}
    group_vuv: dict = {}
    groups_skipped: list[str] = []

    tone_f0: dict[str, list[float]] = {tone: [] for tone in TONES}
    per_clip: list[dict] = []
    vuv_hits = vuv_total = 0
    energy_errors: list[float] = []
    duration_errors: list[float] = []
    length_ratios: list[float] = []
    total_audio = total_time = 0.0
    gap_sum_errors: list[float] = []
    gap_sum_errors_all: list[float] = []
    gap_pred_frames: list[float] = []
    gap_true_frames: list[float] = []
    mel_gaps: list[float] = []
    pause_masked = [0, 0, 0]  # TP, FP, FN — chỉ vị trí có nhãn (bỏ gap cuối row)
    pause_all = [0, 0, 0]     # gồm cả vị trí gap cuối row khi manifest có giá trị
    slopes: dict[str, dict[str, list[float]]] = {
        metric: {"true": [], "model": []} for metric in _SLOPE_METRICS}
    # B1: cặp (thật, model) CÙNG cửa sổ cho Δ ngữ điệu — tổng thể + tách theo nhãn âm tiết cuối.
    slope_pairs: dict[str, list[list[float]]] = {}
    slope_final_pairs: dict[str, dict[str, dict[str, list[list[float]]]]] = {}
    count = len(dataset) if limit is None else min(limit, len(dataset))

    for index in range(count):
        row = dataset.rows[index]
        item = dataset[index]
        features = item["features"].unsqueeze(0).to(device_t)
        started = time.perf_counter()
        with torch.no_grad():
            outputs = model(features, speed=speed_t.to(device_t))
        elapsed = time.perf_counter() - started

        pred_mel = outputs["mel"][0].cpu()
        target_mel = item["mel"]
        pred_frames, true_frames = pred_mel.size(0), target_mel.size(0)
        total_audio += pred_frames * hop / sample_rate
        total_time += elapsed
        length_ratios.append(pred_frames / max(true_frames, 1))
        overlap = min(pred_frames, true_frames)
        raw_pred = pred_mel[:overlap] * std + mean
        raw_target = target_mel[:overlap] * std + mean
        l1_z = float((pred_mel[:overlap] - target_mel[:overlap]).abs().mean())
        l1_raw = float((raw_pred - raw_target).abs().mean())
        speech = raw_target.mean(dim=1) > -8.0
        speech_db = (float((raw_pred[speech] - raw_target[speech]).abs().mean() * 8.686)
                     if bool(speech.any()) else None)

        predicted_duration = torch.round(
            (torch.exp(outputs["log_duration"][0].cpu()) - 1).clamp_min(1))
        true_durations = item["durations"]
        common = min(predicted_duration.size(0), true_durations.size(0))
        duration_errors.extend(
            (predicted_duration[:common] - true_durations[:common]).abs().tolist())

        active = _v2_structural_active(item["features"][:common])
        active_t = torch.tensor(active)
        pred_vuv_full = (torch.sigmoid(outputs["seg_vuv_logits"][0].cpu()) > 0.5)
        true_vuv_full = item["seg_vuv"] >= 0.5
        pred_vuv, true_vuv = pred_vuv_full[:common], true_vuv_full[:common]
        vuv_hits += int((pred_vuv[active_t] == true_vuv[active_t]).sum())
        vuv_total += int(active_t.sum())
        energy_errors.append(float(
            (outputs["seg_energy"][0].cpu()[:common] - item["seg_energy"][:common]
             ).abs()[active_t].mean()))

        # ── nhịp nghỉ: gap/slot model tự sinh vs gap thật (bỏ vị trí gap cuối row) ──
        true_gaps = item["gaps"].tolist()
        slot = outputs["slot_durations"][0].cpu()
        pred_gaps = _pred_gap_slots(slot)
        pred_windows = _v2_syllable_windows(predicted_duration.tolist(), pred_gaps.tolist(),
                                            int(slot[0]))
        gate = outputs["pause_gate"][0].cpu().numpy()
        labelled = int(min(len(true_gaps), len(pred_gaps), true_durations.size(0) - 1))
        labelled = max(labelled, 0)
        if labelled:
            true_array = np.asarray(true_gaps[:labelled], dtype=np.int64)
            pred_array = pred_gaps[:labelled]
            gap_sum_errors.append(abs(float(pred_array.sum() - true_array.sum())))
            gap_pred_frames.append(float(pred_array.sum()))
            gap_true_frames.append(float(true_array.sum()))
            true_pause = true_array >= PAUSE_GATE_MIN_FRAMES
            pred_pause = (gate[:labelled] > 0.5) | (pred_array >= PAUSE_GATE_MIN_FRAMES)
            hits, false, missed = _pause_counts(pred_pause, true_pause)
            pause_masked[0] += hits
            pause_masked[1] += false
            pause_masked[2] += missed
        positions = int(min(len(true_gaps), len(pred_gaps)))
        if positions:
            true_array = np.asarray(true_gaps[:positions], dtype=np.int64)
            pred_array = pred_gaps[:positions]
            gap_sum_errors_all.append(abs(float(pred_array.sum() - true_array.sum())))
            true_pause = true_array >= PAUSE_GATE_MIN_FRAMES
            pred_pause = (gate[:positions] > 0.5) | (pred_array >= PAUSE_GATE_MIN_FRAMES)
            hits, false, missed = _pause_counts(pred_pause, true_pause)
            pause_all[0] += hits
            pause_all[1] += false
            pause_all[2] += missed

        # Mel trên frame gap/lead THẬT (vị trí, chưa align — xem `gap_note`).
        overlap_gap = _gap_frame_mask_from(item["durations"].tolist(), true_gaps,
                                           int(item["lead"]), true_frames)[:overlap]
        if bool(overlap_gap.any()):
            mel_gaps.append(float((pred_mel[:overlap] - target_mel[:overlap]
                                   ).abs().mean(dim=-1)[torch.tensor(overlap_gap)].mean()))

        pred_log_f0 = outputs["f0"][0].cpu().numpy()
        pred_hz = torch.exp(torch.tensor(pred_log_f0).clamp(-1.0, 8.0))
        windows = _v2_syllable_windows(item["durations"].tolist(), item["gaps"].tolist(),
                                       int(item["lead"]))
        tones = [syllable["abstract"]["tone"] for syllable in row["syllables"]]
        labels: dict[str, list[str]] = {}
        if group_by:
            labels, row_skipped = _group_labels(row, item["features"], vocabs, group_by)
            groups_skipped.extend(row_skipped)
        for syllable_index, (low, high) in enumerate(windows):
            high_c = min(high, overlap)
            if group_by and syllable_index < true_durations.size(0) and high_c > low:
                for group, values in labels.items():
                    label = values[syllable_index]
                    _group_add(group_store, group, label, "mel_z", float(
                        (pred_mel[low:high_c] - target_mel[low:high_c]).abs().mean()))
                    window_speech = speech[low:high_c]
                    if bool(window_speech.any()):
                        _group_add(group_store, group, label, "mel_db", float(
                            (raw_pred[low:high_c][window_speech]
                             - raw_target[low:high_c][window_speech]).abs().mean() * 8.686))
                    _group_add(group_store, group, label, "dur_mae", abs(float(
                        predicted_duration[syllable_index] - true_durations[syllable_index])))
                    active_syllable = torch.tensor(
                        _v2_structural_active(item["features"][syllable_index:syllable_index + 1]))[0]
                    _group_vuv_add(group_vuv, group, label, int((pred_vuv_full[syllable_index]
                                                                 == true_vuv_full[syllable_index]
                                                                 )[active_syllable].sum()),
                                   int(active_syllable.sum()))
            if high_c <= low or syllable_index >= len(tones):
                continue
            voiced = item["vuv"][low:high_c] > 0.5
            if bool(voiced.any()):
                ratio = torch.clamp(pred_hz[low:high_c][voiced], min=1e-6) / torch.clamp(
                    item["f0"][low:high_c][voiced], min=1e-6)
                cents = 1200.0 * torch.abs(torch.log2(torch.clamp(ratio, min=1e-9)))
                tone_f0[tones[syllable_index]].extend(cents.tolist())
                if group_by:
                    for group, values in labels.items():
                        _group_add(group_store, group, values[syllable_index], "f0_cents",
                                   float(cents.mean()))

        # ── ngữ điệu: dốc logF0 thật vs model (timeline của chính nó) ─────────
        true_voiced = (item["vuv"][:true_frames] > 0.5) & (item["f0"][:true_frames] > 1e-3)
        true_positions = np.nonzero(true_voiced.numpy())[0]
        true_series = (true_positions,
                       np.log(np.clip(item["f0"][:true_frames].numpy()[true_positions], 1e-6, None)))
        segment_vuv = torch.sigmoid(outputs["seg_vuv_logits"][0].cpu())
        slot_vuv = torch.cat([torch.zeros(1), torch.cat(  # G_lead = vô thanh, slot G = 0
            [segment_vuv, torch.zeros(segment_vuv.size(0), 1)], dim=1).reshape(-1)]).numpy()
        pred_voiced = (_slot_to_frames(slot_vuv, slot.numpy(), pred_frames) > 0.5) \
            & (pred_log_f0 > 0.5)
        pred_positions = np.nonzero(pred_voiced)[0]
        block = _slope_block(true_series, (pred_positions, pred_log_f0[pred_positions]),
                             windows, pred_windows, tones,
                             [int(value) for value in item["features"][:, 6].tolist()],
                             punct_codes, frame_rate)
        # B1: nhãn âm tiết CUỐI row (cùng máy móc nhóm ghép `coda:tone`) để cắt Δ ngữ điệu theo
        # `punctuation` / `tone` / `punctuation:tone` — đúng tầng ghép cặp mà §43.2 cần.
        final_labels, _ = _group_labels(row, item["features"], vocabs,
                                        list(_SLOPE_FINAL_GROUPS))
        final_labels = {group: values[-1] for group, values in final_labels.items() if values}
        for metric, pair in block.items():
            for tag, value in pair.items():
                if value is not None:
                    slopes[metric][tag].append(float(value))
            if pair.get("true") is None or pair.get("model") is None:
                continue  # chỉ ghép cặp khi CÙNG clip/metric có cả thật lẫn model
            slope_pairs.setdefault(metric, []).append([float(pair["true"]), float(pair["model"])])
            for group, label in final_labels.items():
                bucket = slope_final_pairs.setdefault(group, {}).setdefault(label, {})
                bucket.setdefault(metric, []).append([float(pair["true"]), float(pair["model"])])

        per_clip.append({
            "id": row["id"], "frames": true_frames, "pred_frames": pred_frames,
            "length_ratio": round(length_ratios[-1], 3),
            "mel_l1_z": round(l1_z, 4), "mel_l1_raw": round(l1_raw, 4),
            "mel_mae_db": round(8.686 * l1_raw, 3),
            "mel_mae_db_speech": round(speech_db, 3) if speech_db is not None else None,
        })

    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes
    summary: dict = {
        "mode": "free", "arch": "v2_slot5", "speed": float(speed),
        "clips": len(per_clip), "clip_ids": [entry["id"] for entry in per_clip],
        "device": device, "rtf_backend": "pytorch-cpu",
        "mel_l1_z": _quantiles([entry["mel_l1_z"] for entry in per_clip]),
        "mel_mae_db": _quantiles([entry["mel_mae_db"] for entry in per_clip]),
        "mel_mae_db_speech": _quantiles([entry["mel_mae_db_speech"] for entry in per_clip
                                         if entry["mel_mae_db_speech"] is not None]),
        "length_ratio": _quantiles(length_ratios),
        "f0_rmse_cents_by_tone": {tone: _quantiles(values) for tone, values in tone_f0.items()},
        "f0_note": V2_F0_NOTE,
        "vuv_accuracy": round(vuv_hits / max(vuv_total, 1), 4),
        "vuv_note": V2_VUV_NOTE,
        "energy_mae": round(float(np.mean(energy_errors)), 4) if energy_errors else None,
        "energy_note": V2_ENERGY_NOTE,
        "duration_mae_frames": _quantiles(duration_errors),
        "rtf": round(total_time / max(total_audio, 1e-9), 4),
        "ram_peak_gib": round(rss_gib, 2),
        # ── nhịp nghỉ (free-run): gap model tự sinh vs gap thật ──────────────
        "gap_sum_mae_frames": _quantiles(gap_sum_errors),
        "gap_sum_mae_all_frames": _quantiles(gap_sum_errors_all),
        "gap_sum_pred_frames": _quantiles(gap_pred_frames),
        "gap_sum_true_frames": _quantiles(gap_true_frames),
        "pause_f1": _f1(*pause_masked),
        "pause_precision": (round(pause_masked[0] / (pause_masked[0] + pause_masked[1]), 4)
                            if pause_masked[0] + pause_masked[1] else None),
        "pause_recall": (round(pause_masked[0] / (pause_masked[0] + pause_masked[2]), 4)
                         if pause_masked[0] + pause_masked[2] else None),
        "pause_f1_all": _f1(*pause_all),
        "pause_counts": {"tp": pause_masked[0], "fp": pause_masked[1], "fn": pause_masked[2],
                         "tp_all": pause_all[0], "fp_all": pause_all[1], "fn_all": pause_all[2]},
        "pause_note": _GAP_NOTE,
        "mel_gap": _quantiles(mel_gaps),
        "mel_gap_note": "L1 mel (z) CHỈ trên frame gap/lead THẬT, so theo vị trí (chưa align)",
        # ── ngữ điệu / declination (§35.1 NGANG-drift; B1/§43.2 sửa thước `?`) ──
        "intonation": {metric: {tag: _quantiles(values) for tag, values in pair.items()}
                       for metric, pair in slopes.items()},
        # B1 — Δ GHÉP CẶP (model − thật, median + CI 95% bootstrap): con số phải đọc để kết luận.
        "intonation_paired": _paired_report(slope_pairs),
        "intonation_by_final": _paired_report_nested(slope_final_pairs),
        "intonation_note": _SLOPE_NOTE,
        "per_clip": per_clip,
    }
    if group_by:
        summary["groups"] = _group_report(group_store, group_vuv, group_by)
        summary["groups_by"] = group_by
        summary["groups_note"] = _GROUP_NOTE + " (free-run: cửa sổ âm tiết THẬT, cắt theo overlap)"
        absent = [group for group in sorted(set(groups_skipped)) if group not in summary["groups"]]
        partial = [group for group in sorted(set(groups_skipped)) if group in summary["groups"]]
        if absent:
            summary["groups_skipped"] = absent
        if partial:
            summary["groups_partial"] = partial

    # ── CER qua waveform (sau vòng đo acoustic để không lẫn vào RTF) ──────────
    by_id = {entry["id"]: entry for entry in per_clip}
    text_rows = [dataset.rows[index] for index in range(count)
                 if str(dataset.rows[index].get("text") or "").strip()]
    if skip_cer:
        summary["cer"] = None
        summary["cer_note"] = "bỏ qua CER theo yêu cầu (`--no-cer`)"
    elif not text_rows:
        summary["cer"] = None
        summary["cer_note"] = "bỏ qua CER: manifest thiếu `text`"
    else:
        whisper = _load_whisper()
        if whisper is None:
            summary["cer"] = None
            summary["cer_note"] = "bỏ qua CER: thiếu faster-whisper"
        else:
            if wav_dir is None:
                tmp = tempfile.TemporaryDirectory(prefix="v2_free_wavs_")
                directory = Path(tmp.name)
            else:
                tmp = None
                directory = Path(wav_dir)
            try:
                wav_paths = _synthesize_free_v2_wavs(model, config, text_rows, directory,
                                                     device, speed)
                scored: list[float] = []
                for row, path in zip(text_rows, wav_paths):
                    if path is None:
                        continue  # text không parse ra âm tiết nào
                    cer_value, whisper_text = _score_transcript(
                        row["text"], _transcribe(whisper, path))
                    by_id[row["id"]]["cer"] = cer_value
                    by_id[row["id"]]["whisper"] = whisper_text
                    scored.append(cer_value)
            finally:
                if tmp is not None:
                    tmp.cleanup()
            if scored:
                summary["cer"] = _quantiles(scored)
            else:
                summary["cer"] = None
                summary["cer_note"] = "bỏ qua CER: không có row nào parse ra âm tiết"
    return summary


def run_tf_v2(args, config: dict, *, use_seg: bool) -> dict:
    """Dựng `SlotDataset` (chỉ đọc cache) + model v2 rồi chạy `evaluate_tf_v2`."""
    from vifasttts.data.dataset_v2 import SlotDataset

    f0_stats = resolve_f0_stats(args.checkpoint, config, getattr(args, "f0_stats", None))
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir)
    model, _ = load_eval_model(args.checkpoint, config, f0_stats=f0_stats)
    summary = evaluate_tf_v2(model, dataset, config, use_seg=use_seg,
                             device=args.device, limit=args.clips,
                             group_by=parse_group_by(getattr(args, "group_by", None)))
    summary["checkpoint"] = args.checkpoint
    summary["manifest"] = args.manifest
    return summary


def parse_group_by(value: str | None) -> list[str] | None:
    """`--group-by coda,tone,coda:tone` → `["coda", "tone", "coda:tone"]` (kiểm tra ngay từ CLI).

    Nhóm ghép (`coda:tone`) là tích hai nhóm — dùng để tách hiệu ứng bị lẫn.
    """
    if not value:
        return None
    groups = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [part for group in groups for part in group.split(":") if part not in GROUP_KEYS]
    if unknown:
        raise SystemExit(f"--group-by không hỗ trợ {unknown} (chọn trong {list(GROUP_KEYS)})")
    return groups or None


def run_free_v2(args, config: dict) -> dict:
    """Dựng `SlotDataset` + model v2 rồi chạy `evaluate_free_v2` (acoustic + CER qua waveform)."""
    from vifasttts.data.dataset_v2 import SlotDataset

    f0_stats = resolve_f0_stats(args.checkpoint, config, getattr(args, "f0_stats", None))
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir)
    model, _ = load_eval_model(args.checkpoint, config, f0_stats=f0_stats)
    summary = evaluate_free_v2(model, dataset, config, device=args.device,
                               limit=args.clips, speed=getattr(args, "speed", 1.0),
                               wav_dir=Path(args.out).parent / "free_wavs",
                               group_by=parse_group_by(getattr(args, "group_by", None)),
                               skip_cer=bool(getattr(args, "no_cer", False)))
    summary["checkpoint"] = args.checkpoint
    summary["manifest"] = args.manifest
    return summary


def _load_whisper():
    """Nạp faster-whisper small (int8, CPU); None nếu thiếu thư viện."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("⚠️  Không có faster-whisper — bỏ qua CER")
        return None
    return WhisperModel("small", device="cpu", compute_type="int8")


def _transcribe(whisper, path: Path) -> str:
    """Whisper transcribe một file WAV (tự resample 16 kHz bằng numpy)."""
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32")
    target = round(len(data) * 16000 / rate)
    resampled = np.interp(np.linspace(0, len(data) - 1, target), np.arange(len(data)),
                          data).astype(np.float32)
    segments, _ = whisper.transcribe(resampled, language="vi", beam_size=1)
    return " ".join(segment.text for segment in segments).strip()


def _score_transcript(reference: str, hypothesis: str) -> tuple[float, str]:
    """CER + hypothesis rút gọn cho một clip (dùng chung free v1/v2 — cùng công thức)."""
    return round(_char_cer(reference, hypothesis), 4), hypothesis[:200]


def _synthesize_free_v2_wavs(model, config: dict, rows: list[dict], wav_dir: Path,
                             device: str, speed: float) -> list[Path]:
    """Text → WAV bằng model v2 đã nạp + vocoder dựng từ config (cơ chế như `Synthesizer`).

    Mỗi row cần `text` parse được (row không có âm tiết nào thì bỏ qua, giữ thứ tự
    bằng `None`). Model chạy một forward có `speed` trong graph (khớp `_render` v2).
    """
    import torch

    from vifasttts.frontend.pipeline import parse_text
    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.infer import write_wav
    from vifasttts.models.istftnet import build_generator

    vocabs = ComponentVocabs.default()
    device_t = torch.device(device)
    vocoder = build_generator(config.get("vocoder_training", {}), config["mel"])
    vocoder.to(device_t)
    vocoder.eval()
    model.eval()
    speed_t = torch.tensor([float(speed)], dtype=torch.float32, device=device_t)
    wav_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with torch.no_grad():
        for row in rows:
            items = parse_text(row["text"])
            if not items:
                paths.append(None)  # type: ignore[arg-type]
                continue
            features = torch.tensor([[vocabs.encode(item) for item in items]],
                                    dtype=torch.long, device=device_t)
            mel = model(features, speed=speed_t)["mel"][0]
            waveform = vocoder(mel.unsqueeze(0))[0]
            target = wav_dir / f"{row.get('id', 'clip')}.wav"
            write_wav(target, waveform.detach().cpu(),
                      int(config.get("sample_rate", 24000)))
            paths.append(target)
    return paths


def run_free(args, config: dict) -> dict:
    """Free-run: text → acoustic → vocoder; đo CER (Whisper), RTF, RAM (PyTorch CPU)."""
    import resource
    import time

    from vifasttts.infer import Synthesizer

    rows = _read_rows(args.manifest)[: args.clips]
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    wav_dir = Path(args.out).parent / "free_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    whisper = _load_whisper()

    per_clip: list[dict] = []
    total_audio = total_time = 0.0
    for row in rows:
        target = wav_dir / f"{row['id']}.wav"
        started = time.perf_counter()
        samples, rate = engine.synthesize(row["text"], target)
        elapsed = time.perf_counter() - started
        audio_seconds = samples / rate
        total_audio += audio_seconds
        total_time += elapsed
        hypothesis = _transcribe(whisper, target) if whisper is not None else ""
        cer, whisper_text = _score_transcript(row["text"], hypothesis)
        per_clip.append({
            "id": row["id"], "audio_seconds": round(audio_seconds, 3),
            "synth_seconds": round(elapsed, 3),
            "cer": cer,
            "whisper": whisper_text,
        })
        print(f"  [free] {row['id']} · {audio_seconds:.1f}s · CER {per_clip[-1]['cer']:.3f}")

    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes
    return {
        "mode": "free", "checkpoint": args.checkpoint, "manifest": args.manifest,
        "clips": len(per_clip), "device": args.device, "rtf_backend": "pytorch-cpu",
        "cer": _quantiles([entry["cer"] for entry in per_clip]),
        "rtf": round(total_time / max(total_audio, 1e-9), 4),
        "ram_peak_gib": round(rss_gib, 2),
        "per_clip": per_clip,
    }


def main() -> None:
    """Đọc tham số, chạy mode, ghi JSON + in bảng tóm tắt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["tf_syl", "tf_seg", "free"], default="tf_syl")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clips", type=int, default=100)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--f0-stats", default=None,
                        help="JSON f0_stats v2 (mặc định: state checkpoint → config → data/codelung/f0_stats_v2.json)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="tốc độ free-run v2 (model tự scale durations/gaps)")
    parser.add_argument("--group-by", default=None,
                        help="tách metric cấp âm tiết theo nhóm, vd `coda,tone,status` hoặc nhóm "
                             f"ghép `coda:tone` (chọn trong {','.join(GROUP_KEYS)}); chỉ v2")
    parser.add_argument("--no-cer", action="store_true",
                        help="free mode: bỏ CER (chạy nhanh khi chỉ cần nhịp nghỉ/ngữ điệu)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    arch = _checkpoint_arch(args.checkpoint)
    if arch == "v2_slot5":
        use_seg = args.mode == "tf_seg"
        summary = run_free_v2(args, config) if args.mode == "free" else run_tf_v2(
            args, config, use_seg=use_seg)
    else:
        if args.mode == "tf_seg":
            raise SystemExit("tf_seg chỉ dùng cho v2 (arch=v2_slot5) — v1 không có seg_durations")
        if args.group_by:
            raise SystemExit("--group-by chỉ hỗ trợ v2 (arch=v2_slot5)")
        summary = run_tf_syl(args, config) if args.mode == "tf_syl" else run_free(args, config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {out} (arch={summary.get('arch', 'v1_syl')})")
    _print_summary(summary, args.mode)


def _quantile_line(stats: dict | None) -> str:
    """`mean (median, n)` gọn cho một khối `_quantiles` (n=0 → `—`)."""
    if not stats or not stats.get("n"):
        return "—"
    return f"{stats['mean']:.3f} (median {stats['median']:.3f}, n={stats['n']})"


def _print_summary(summary: dict, mode: str) -> None:
    """In gọn: metric chính + nhóm cấp âm tiết + nhịp nghỉ/ngữ điệu (free)."""
    if mode in ("tf_syl", "tf_seg"):
        print(f"  {summary['clips']} clip · mel L1 z {summary['mel_l1_z']['mean']:.3f} "
              f"· raw {summary['mel_l1_raw']['mean']:.3f} · {summary['mel_mae_db']['mean']:.1f} dB "
              f"· speech {summary['mel_mae_db_speech'].get('mean', float('nan')):.1f} dB "
              f"({summary['speech_frame_frac_mean']:.0%} frame)")
        print(f"  VUV acc {summary['vuv_accuracy']:.3f} · energy MAE {summary['energy_mae']:.3f} "
              f"· duration MAE {summary['duration_mae_ms']:.1f} ms")
        if summary.get("slot_sum_mae_frames") is not None:
            print(f"  slot Σ lệch {summary['slot_sum_mae_frames']:.3f} frame")
        print(f"  mel GAP (frame nghỉ thật): z {_quantile_line(summary['mel_gap_z'])} "
              f"· {_quantile_line(summary['mel_gap_db'])} dB")
        for tone, stats in summary["f0_rmse_cents_by_tone"].items():
            if stats.get("n"):
                print(f"  F0 {tone:6s} median {stats['median']:7.1f} cents (n={stats['n']})")
        _print_groups(summary)
        return

    if summary.get("arch") == "v2_slot5":
        print(f"  {summary['clips']} clip · mel {summary['mel_mae_db']['mean']:.1f} dB "
              f"· length_ratio {summary['length_ratio']['mean']:.3f} "
              + (f"· CER TB {summary['cer']['mean']:.3f}"
                 if summary.get("cer") is not None
                 else f"· CER bỏ qua ({summary['cer_note']})"))
        print(f"  VUV acc {summary['vuv_accuracy']:.3f} "
              f"· duration MAE {summary['duration_mae_frames']['mean']:.1f} frame "
              f"· RTF acoustic {summary['rtf']:.3f} · RAM đỉnh {summary['ram_peak_gib']:.2f} GiB")
        print(f"  NHỊP NGHỈ: gap Σ MAE {_quantile_line(summary['gap_sum_mae_frames'])} frame "
              f"· pause F1 {summary['pause_f1']} (P {summary['pause_precision']} / "
              f"R {summary['pause_recall']}; gồm gap cuối: {summary['pause_f1_all']})")
        print(f"    gap Σ dự đoán {_quantile_line(summary['gap_sum_pred_frames'])} "
              f"vs thật {_quantile_line(summary['gap_sum_true_frames'])} · "
              f"mel_gap {_quantile_line(summary['mel_gap'])}")
        print("  NGỮ ĐIỆU (cent/giây; true → model):")
        for metric, pair in (summary.get("intonation") or {}).items():
            print(f"    {metric:24s} {_quantile_line(pair['true'])} → "
                  f"{_quantile_line(pair['model'])}")
        _print_groups(summary)
        return

    print(f"  {summary['clips']} clip · CER TB {summary['cer']['mean']:.3f} "
          f"· RTF {summary['rtf']:.3f} · RAM đỉnh {summary['ram_peak_gib']:.2f} GiB")


def _print_groups(summary: dict) -> None:
    """In bảng nhóm cấp âm tiết (sắp theo mel_z giảm dần) khi có `--group-by`."""
    groups = summary.get("groups") or {}
    if not groups:
        return
    for group, values in groups.items():
        print(f"  NHÓM `{group}` (mel_z · mel_db hữu thanh · dur MAE · F0 cent · vuv):")
        ordered = sorted(values.items(),
                         key=lambda item: -item[1]["mel_z"].get("mean", float("-inf")))
        for label, metrics in ordered:
            print(f"    {label:9s} n={metrics['n_syllables']:5d} "
                  f"mel_z {metrics['mel_z'].get('mean', float('nan')):.3f} · "
                  f"mel_db {metrics['mel_db'].get('mean', float('nan')):.2f} · "
                  f"dur {metrics['dur_mae'].get('mean', float('nan')):.2f} fr · "
                  f"F0 {metrics['f0_cents'].get('median', float('nan')):.0f} · "
                  f"vuv {metrics['vuv_acc']}")
    if summary.get("groups_skipped"):
        print(f"    (bỏ nhóm: {summary['groups_skipped']} — manifest không có dữ liệu)")
    if summary.get("groups_partial"):
        print(f"    (nhóm thiếu ở một số row: {summary['groups_partial']} — chỉ gộp row có nhãn)")


if __name__ == "__main__":
    main()

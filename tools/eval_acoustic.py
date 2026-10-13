"""tools/eval_acoustic.py — cổng nghiệm thu acoustic (T1.2 / B6).

Chế độ:
  tf_syl — cho `d_syl`/`gaps`/`lead` thật (v1: truyền `spans`); đo mel L1 (z/raw/dB),
           F0 RMSE theo 6 thanh (chỉ frame hữu thanh thật, mask `vuv`), VUV, energy,
           duration (từ `log_duration`).
  tf_seg — v2 (cho thêm `seg_durations`); v1: raise NotImplementedError.
  free   — không cho gì: CER (Whisper), RTF, RAM.

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
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TONES = ["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]
_VI_CHARS = "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ"


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


def evaluate_tf_v2(model, dataset, config: dict, *, use_seg: bool,
                   device: str = "cpu", limit: int | None = None) -> dict:
    """Teacher-forced v2 thuần (test gọi trực tiếp): `use_seg` = cho `seg_durations` (tf_seg).

    Đo đúng key v1 (mel L1 z/raw/dB + speech, F0 cents theo 6 thanh, VUV, energy,
    duration) + `arch` và `slot_sum_mae_frames` (B2 tự chia ở tf_syl).
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

    tone_f0: dict[str, list[float]] = {tone: [] for tone in TONES}
    per_clip: list[dict] = []
    vuv_hits = vuv_total = 0
    vuv_nuc_hits = vuv_nuc_total = 0
    energy_errors: list[float] = []
    duration_errors: list[float] = []
    slot_sum_errors: list[float] = []
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
        for syllable_index, (low, high) in enumerate(windows):
            low_c = max(low, lo)
            high_c = min(high, pred_hz.size(0), true_f0.size(0))
            if high_c <= low_c or syllable_index >= len(tones):
                continue
            voiced = true_vuv_frame[low_c:high_c] > 0.5
            if bool(voiced.any()):
                ratio = torch.clamp(pred_hz[low_c:high_c][voiced], min=1e-6) / torch.clamp(
                    true_f0[low_c:high_c][voiced], min=1e-6)
                cents = 1200.0 * torch.abs(torch.log2(torch.clamp(ratio, min=1e-9)))
                tone_f0[tones[syllable_index]].extend(cents.tolist())

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
        })

    return {
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
        "per_clip": per_clip,
    }


def evaluate_free_v2(model, dataset, config: dict, *, device: str = "cpu",
                     limit: int | None = None, speed: float = 1.0) -> dict:
    """Free-run v2 thuần: model tự dự đoán durations/slot (không cho gì ngoài features).

    Mel/F0 đo trên vùng overlap với target thật + `length_ratio`; CER = None
    (`Synthesizer` chỉ hỗ trợ v1 nên không có waveform để Whisper chấm).
    """
    import resource
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

    tone_f0: dict[str, list[float]] = {tone: [] for tone in TONES}
    per_clip: list[dict] = []
    vuv_hits = vuv_total = 0
    energy_errors: list[float] = []
    duration_errors: list[float] = []
    length_ratios: list[float] = []
    total_audio = total_time = 0.0
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
        pred_vuv = (torch.sigmoid(outputs["seg_vuv_logits"][0].cpu()[:common]) > 0.5)
        true_vuv = item["seg_vuv"][:common] >= 0.5
        vuv_hits += int((pred_vuv[active_t] == true_vuv[active_t]).sum())
        vuv_total += int(active_t.sum())
        energy_errors.append(float(
            (outputs["seg_energy"][0].cpu()[:common] - item["seg_energy"][:common]
             ).abs()[active_t].mean()))

        pred_hz = torch.exp(outputs["f0"][0].cpu().clamp(-1.0, 8.0))
        windows = _v2_syllable_windows(item["durations"].tolist(), item["gaps"].tolist(),
                                       int(item["lead"]))
        tones = [syllable["abstract"]["tone"] for syllable in row["syllables"]]
        for syllable_index, (low, high) in enumerate(windows):
            high_c = min(high, overlap)
            if high_c <= low or syllable_index >= len(tones):
                continue
            voiced = item["vuv"][low:high_c] > 0.5
            if bool(voiced.any()):
                ratio = torch.clamp(pred_hz[low:high_c][voiced], min=1e-6) / torch.clamp(
                    item["f0"][low:high_c][voiced], min=1e-6)
                cents = 1200.0 * torch.abs(torch.log2(torch.clamp(ratio, min=1e-9)))
                tone_f0[tones[syllable_index]].extend(cents.tolist())

        per_clip.append({
            "id": row["id"], "frames": true_frames, "pred_frames": pred_frames,
            "length_ratio": round(length_ratios[-1], 3),
            "mel_l1_z": round(l1_z, 4), "mel_l1_raw": round(l1_raw, 4),
            "mel_mae_db": round(8.686 * l1_raw, 3),
            "mel_mae_db_speech": round(speech_db, 3) if speech_db is not None else None,
        })

    rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes
    return {
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
        "cer": None,
        "cer_note": "bỏ qua CER: Synthesizer/vocoder chỉ hỗ trợ v1 (không có waveform)",
        "rtf": round(total_time / max(total_audio, 1e-9), 4),
        "ram_peak_gib": round(rss_gib, 2),
        "per_clip": per_clip,
    }


def run_tf_v2(args, config: dict, *, use_seg: bool) -> dict:
    """Dựng `SlotDataset` (chỉ đọc cache) + model v2 rồi chạy `evaluate_tf_v2`."""
    from vifasttts.data.dataset_v2 import SlotDataset

    f0_stats = resolve_f0_stats(args.checkpoint, config, getattr(args, "f0_stats", None))
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir)
    model, _ = load_eval_model(args.checkpoint, config, f0_stats=f0_stats)
    summary = evaluate_tf_v2(model, dataset, config, use_seg=use_seg,
                             device=args.device, limit=args.clips)
    summary["checkpoint"] = args.checkpoint
    summary["manifest"] = args.manifest
    return summary


def run_free_v2(args, config: dict) -> dict:
    """Dựng `SlotDataset` + model v2 rồi chạy `evaluate_free_v2` (acoustic, không CER)."""
    from vifasttts.data.dataset_v2 import SlotDataset

    f0_stats = resolve_f0_stats(args.checkpoint, config, getattr(args, "f0_stats", None))
    dataset = SlotDataset(args.manifest, config, f0_stats, cache_dir=args.cache_dir)
    model, _ = load_eval_model(args.checkpoint, config, f0_stats=f0_stats)
    summary = evaluate_free_v2(model, dataset, config, device=args.device,
                               limit=args.clips, speed=getattr(args, "speed", 1.0))
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
        per_clip.append({
            "id": row["id"], "audio_seconds": round(audio_seconds, 3),
            "synth_seconds": round(elapsed, 3),
            "cer": round(_char_cer(row["text"], hypothesis), 4),
            "whisper": hypothesis[:200],
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
        summary = run_tf_syl(args, config) if args.mode == "tf_syl" else run_free(args, config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {out} (arch={summary.get('arch', 'v1_syl')})")
    if args.mode in ("tf_syl", "tf_seg"):
        print(f"  {summary['clips']} clip · mel L1 z {summary['mel_l1_z']['mean']:.3f} "
              f"· raw {summary['mel_l1_raw']['mean']:.3f} · {summary['mel_mae_db']['mean']:.1f} dB "
              f"· speech {summary['mel_mae_db_speech'].get('mean', float('nan')):.1f} dB "
              f"({summary['speech_frame_frac_mean']:.0%} frame)")
        print(f"  VUV acc {summary['vuv_accuracy']:.3f} · energy MAE {summary['energy_mae']:.3f} "
              f"· duration MAE {summary['duration_mae_ms']:.1f} ms")
        if summary.get("slot_sum_mae_frames") is not None:
            print(f"  slot Σ lệch {summary['slot_sum_mae_frames']:.3f} frame")
        for tone, stats in summary["f0_rmse_cents_by_tone"].items():
            if stats.get("n"):
                print(f"  F0 {tone:6s} median {stats['median']:7.1f} cents (n={stats['n']})")
    else:
        if summary.get("arch") == "v2_slot5":
            print(f"  {summary['clips']} clip · mel {summary['mel_mae_db']['mean']:.1f} dB "
                  f"· length_ratio {summary['length_ratio']['mean']:.3f} "
                  f"· CER bỏ qua ({summary['cer_note']})")
            print(f"  VUV acc {summary['vuv_accuracy']:.3f} "
                  f"· duration MAE {summary['duration_mae_frames']['mean']:.1f} frame "
                  f"· RTF acoustic {summary['rtf']:.3f} · RAM đỉnh {summary['ram_peak_gib']:.2f} GiB")
        else:
            print(f"  {summary['clips']} clip · CER TB {summary['cer']['mean']:.3f} "
                  f"· RTF {summary['rtf']:.3f} · RAM đỉnh {summary['ram_peak_gib']:.2f} GiB")


if __name__ == "__main__":
    main()

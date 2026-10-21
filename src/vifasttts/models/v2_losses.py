"""Loss acoustic v2 (docs/ACOUSTIC_V2.md §3, §4, §6) — mirror quy ước `losses.acoustic_losses`.

Thành phần: mel L1 (mask frame, chia số bin) · mel_pre (0.2) · duration MSE(log1p) ·
**ratio CE** (soft target, mask inactive + `d_syl < 3` — B5) · **F0 hệ số hình dạng** L1 trong
không gian z-score (mask Addendum 6 × `f0_k_mask` cho hệ số thiếu) · vuv BCE theo segment ·
energy L1 · phonation L1 (mask voiced ≥ 0,5) · tone CE · **pause** gate BCE (A1/§43.1:
`pos_weight` 3,0 + trọng số theo khối lượng gap cho dương tính) + log_gap MSE
(chỉ khi gate target = 1).

Trọng số mặc định ở `DEFAULT_WEIGHTS`, config `loss_weights` ghi đè từng key.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vifasttts.losses import masked_l1

DEFAULT_WEIGHTS: dict[str, float] = {
    "mel": 1.0, "mel_pre": 0.2, "mel_sharp": 1.0, "duration": 1.0, "ratio": 1.0,
    "f0": 1.0, "f0_delta": 0.5, "f0_prior": 0.1, "tone_disc": 0.0,
    "vuv": 0.5, "energy": 0.5, "phonation": 0.5,
    "tone": 0.1, "pause_gate": 0.5, "pause_gap": 0.5,
}

#: Nhiệt độ của CE phân biệt thanh (§33.4.3): logits = −d²/T với d = khoảng cách tới prior.
TONE_DISC_TEMPERATURE = 1.0

#: §43.1 (A1) — BCE gate pause: dương tính (gap ≥ 3 frame) chỉ chiếm **1,26%** loss train ⇒ model
#: học thuộc rồi bỏ qua (BCE in-sample 0,057 vs ngoài mẫu 1,148). `pos_weight` ≈ (1−p)/p với tỷ lệ
#: dương tính p ≈ 0,25–0,31 (đo trên manifest train). Ước lượng tác động là **dự đoán** (§43.1);
#: nghiệm thu bằng run thật (`configs/small_codelung_v2_base_pause.yaml` + cổng §43.1).
PAUSE_POS_WEIGHT = 1.0  # A1: mặc định 1,0 = hành vi CŨ (config `..._base_pause.yaml` đặt 3,0)
#: Trọng số theo **khối lượng** cho mẫu dương: `w = 1 + clamp(gap_frame, 0, CLIP)/SCALE`.
#: Đòn trúng đúng chỗ thâm hụt: FN trung bình 28 frame/vị trí, FP trung bình ~6,6 frame (§43.1).
PAUSE_MASS_SCALE = 10.0  # chia khối lượng (0 = tắt phạt theo khối lượng)
PAUSE_MASS_CLIP_FRAMES = 40.0

#: Vị trí chuẩn hoá để đo "độ dốc" contour (độc lập độ dài — B1/§6).
DELTA_POSITIONS = (0.25, 0.5, 0.75)


def shape_deltas(shape: torch.Tensor) -> torch.Tensor:
    """Đạo hàm contour theo vị trí chuẩn hoá tại `DELTA_POSITIONS` → `[.., 3]` (T3.2).

    `df/dpos = −√2·π·Σ_{k≥1} k·s_k·sin(πk·pos)` — dùng cho **delta loss** (khớp *hướng đi* của
    cao độ, không phụ thuộc mức trung bình `s0` hay độ dài segment).
    """
    positions = torch.tensor(DELTA_POSITIONS, device=shape.device, dtype=shape.dtype)
    order = torch.arange(shape.size(-1), device=shape.device, dtype=shape.dtype)
    slope = (-math.sqrt(2.0) * math.pi * order.view(-1, 1)
             * torch.sin(math.pi * order.view(-1, 1) * positions.view(1, -1)))
    return shape @ slope


def structural_active(features: torch.Tensor) -> torch.Tensor:
    """[B,N,7] long → [B,N,4] float {0,1} — cùng quy tắc `ViFastTtsModelV2._active_mask`."""
    onset = (features[..., 0] != 0).to(torch.float32)
    medial = (features[..., 1] != 0).to(torch.float32)
    coda = (features[..., 3] != 0).to(torch.float32)
    return torch.stack([onset, medial, torch.ones_like(onset), coda], dim=-1)


def slot_frame_weights(batch: dict[str, torch.Tensor], frames: int, *, onset_boost: float,
                       coda_boost: float) -> torch.Tensor:
    """[B,T] trọng số cho mel loss: 1 + boost ở frame **onset** (đầu) và **coda** (cuối) mỗi âm tiết.

    Vì sao (v3.4): coda chỉ 3–6 frame/row nên L1 gần như bỏ qua ⇒ model đọc NH thành N
    ("xanh"→"xan", đo §53). Tăng trọng số đúng các frame đó buộc model học formant chuyển tiếp.
    Layout khớp `v2_targets`: lead + (syllable: O/M/N/C) + gap, lặp theo âm tiết.
    """
    device = batch["durations"].device
    weight = torch.ones(batch["durations"].size(0), frames, device=device, dtype=torch.float32)
    if onset_boost <= 0 and coda_boost <= 0:
        return weight
    seg4 = batch.get("seg_durations")
    if seg4 is None:
        return weight
    durations = batch["durations"].long()
    gaps = batch.get("gaps")
    lead = batch.get("lead")
    for b in range(durations.size(0)):
        cursor = int(lead[b]) if lead is not None else 0
        for i in range(durations.size(1)):
            dur = int(durations[b, i])
            gap = int(gaps[b, i]) if gaps is not None else 0
            if dur <= 0:
                continue
            o, _m, _n, c = (int(v) for v in seg4[b, i])
            start, end = cursor, cursor + dur
            if onset_boost > 0 and o > 0:
                weight[b, start:min(start + o, frames)] += onset_boost
            if coda_boost > 0 and c > 0:
                weight[b, max(end - c, start):min(end, frames)] += coda_boost
            cursor = end + gap
    return weight


def _speech_frame_weight(batch: dict, frames: int, margin: int = 4) -> torch.Tensor:
    """Trọng số frame: 1 ở vùng tiếng nói (±`margin` frame), 0 ở lặng (A0 v13, v3.5)."""
    vuv = batch["vuv"][:, :frames]
    speech = (vuv >= 0.5).to(torch.float32)
    if speech.size(1) > 0:
        padded = F.pad(speech.unsqueeze(1), (margin, margin), value=0.0)
        speech = F.max_pool1d(padded, kernel_size=2 * margin + 1, stride=1).squeeze(1)
    weight = speech[:, :frames]
    if batch.get("frame_lengths") is not None:
        steps = torch.arange(weight.size(1), device=weight.device).view(1, -1)
        weight = weight * (steps < batch["frame_lengths"].unsqueeze(1)).to(weight.dtype)
    return weight


def slot_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict | None = None,
    *,
    shape_mean: torch.Tensor,
    shape_std: torch.Tensor,
    tone_prior: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Trả (loss tổng đã nhân trọng số, dict từng thành phần để log).

    `shape_mean`/`shape_std` là buffer chuẩn hoá của model (từ `f0_stats_v2`) — target `s_k`
    được z-score cùng thang trước khi so. `tone_prior` [6,4] (z-space, `model.tone_prior_z`)
    bật **prior mềm theo thanh** (T3.2).
    """
    merged = {**DEFAULT_WEIGHTS, **(weights or {})}
    losses: dict[str, torch.Tensor] = {}
    active = structural_active(batch["features"])

    # v3.5: mask frame lặng khỏi mel L1 (giữ ±4 frame quanh biên tiếng nói) — bật bằng
    # `loss_weights.mel_silence_mask: 1.0`; mặc định 0 = hành vi cũ y nguyên.
    silence_mask = float(merged.get("mel_silence_mask", 0.0)) > 0
    coda_boost = float(merged.get("mel_coda_boost", 0.0))
    onset_boost = float(merged.get("mel_onset_boost", 0.0))
    if coda_boost > 0 or onset_boost > 0:
        # v3.4: mel L1 có trọng số theo frame onset/coda (mặc định 0 = hành vi cũ y nguyên).
        mel_error = (outputs["mel"] - batch["mel"]).abs().mean(dim=-1)
        frame_w = slot_frame_weights(batch, mel_error.size(1), onset_boost=onset_boost,
                                     coda_boost=coda_boost)
        if batch.get("frame_lengths") is not None:
            steps = torch.arange(mel_error.size(1), device=mel_error.device).view(1, -1)
            frame_w = frame_w * (steps < batch["frame_lengths"].unsqueeze(1)).to(torch.float32)
        losses["mel"] = (mel_error * frame_w).sum() / frame_w.sum().clamp_min(1e-6)
    else:
        losses["mel"] = masked_l1(outputs["mel"], batch["mel"], batch.get("frame_lengths"),
                                  mean_over_bins=True)
    if "mel_pre" in outputs:
        losses["mel_pre"] = masked_l1(outputs["mel_pre"], batch["mel"], batch.get("frame_lengths"),
                                      mean_over_bins=True)
    if silence_mask and batch.get("vuv") is not None:
        weight = _speech_frame_weight(batch, outputs["mel"].size(1))
        for key, tensor in (("mel", outputs["mel"]), ("mel_pre", outputs.get("mel_pre"))):
            if tensor is None or key not in losses:
                continue
            error = (tensor - batch["mel"]).abs().mean(dim=-1)[:, : weight.size(1)]
            losses[key] = (error * weight).sum() / weight.sum().clamp_min(1e-6)
    # T4.1 (biến thể ổn định): khớp độ **gồ ghề** của mel (chênh lệch liền kề thời gian + tần số).
    # Đo thật: mel model mượt hơn mel thật 62% (thời gian) / 69% (tần số) — đây là "mờ" cần sửa;
    # GAN (adv) ở batch 1 phân kỳ (xem §28) nên chọn loss tất định này làm đòn chính.
    fake_mel = outputs["mel"]
    real_mel = batch["mel"]
    frames = min(fake_mel.size(1), real_mel.size(1))
    fake_mel, real_mel = fake_mel[:, :frames], real_mel[:, :frames]
    frame_mask = torch.ones(fake_mel.size(0), frames, dtype=torch.bool, device=fake_mel.device)
    if batch.get("frame_lengths") is not None:
        steps = torch.arange(frames, device=fake_mel.device).view(1, -1)
        frame_mask = steps < batch["frame_lengths"].unsqueeze(1)
    pair_mask = (frame_mask[:, 1:] & frame_mask[:, :-1]).to(torch.float32)
    delta_time = ((fake_mel[:, 1:] - fake_mel[:, :-1]) - (real_mel[:, 1:] - real_mel[:, :-1])).abs().mean(dim=-1)
    delta_freq = ((fake_mel[:, :, 1:] - fake_mel[:, :, :-1]) - (real_mel[:, :, 1:] - real_mel[:, :, :-1])).abs().mean(dim=-1)
    mask_float = frame_mask.to(torch.float32)
    # E4c (§45): vùng creak — mel dự đoán mượt hơn mel thật 42%/62% ⇒ Vocos render thành "rè".
    # `mel_sharp_creak_boost` > 0 ⇒ w_t = 1 + boost·c_t cho đúng vùng đó (0 = hành vi cũ y nguyên).
    boost = float(merged.get("mel_sharp_creak_boost", 0.0))
    creak_c = batch.get("mel_creak_c")
    if boost > 0 and creak_c is not None:
        creak_c = creak_c[:, :frames].to(fake_mel.dtype)
        weight_time = 1.0 + boost * 0.5 * (creak_c[:, 1:] + creak_c[:, :-1])
        weight_freq = 1.0 + boost * creak_c
        pair_w = pair_mask * weight_time
        freq_w = mask_float * weight_freq
        losses["mel_sharp"] = (
            (delta_time * pair_w).sum() / pair_w.sum().clamp_min(1e-6)
            + (delta_freq * freq_w).sum() / freq_w.sum().clamp_min(1e-6)
        )
    else:
        losses["mel_sharp"] = (
            (delta_time * pair_mask).sum() / pair_mask.sum().clamp_min(1)
            + (delta_freq * mask_float).sum() / mask_float.sum().clamp_min(1)
        )

    losses["duration"] = F.mse_loss(outputs["log_duration"], torch.log1p(batch["durations"].float()))

    ratio_mask = batch["seg_ratio_mask"]
    # mask slot inactive NGAY TRƯỚC softmax (khớp model) — nếu không, logit slot chết làm lệch
    # log p của slot hợp lệ (đã đo bằng test). CE gộp theo âm tiết (như v1), mask B5.
    masked_logits = outputs["seg_ratio_logits"].masked_fill(active < 0.5, -1e9)
    log_prob = F.log_softmax(masked_logits, dim=-1)
    syllable_valid = ratio_mask.any(dim=-1)
    ce = -(batch["seg_ratio"] * log_prob).sum(dim=-1)  # [B,N]
    losses["ratio"] = (ce * syllable_valid).sum() / syllable_valid.sum().clamp_min(1)

    target_z = (batch["f0_shape"] - shape_mean.view(1, 1, 1, -1)) / shape_std.view(1, 1, 1, -1).clamp_min(1e-6)
    f0_mask = batch["f0_weight"].unsqueeze(-1) * batch["f0_k_mask"].to(torch.float32)
    losses["f0"] = ((outputs["seg_shape"] - target_z).abs() * f0_mask).sum() / f0_mask.sum().clamp_min(1)
    # T3.2 — delta loss: khớp *độ dốc contour* (không phụ thuộc s0/độ dài); mask theo trọng số segment
    delta_mask = batch["f0_weight"].unsqueeze(-1)
    delta_error = (shape_deltas(outputs["seg_shape"]) - shape_deltas(target_z)).abs()
    losses["f0_delta"] = (delta_error * delta_mask).sum() / (delta_mask.sum() * len(DELTA_POSITIONS)).clamp_min(1)
    # T3.2 — prior mềm theo thanh: kéo nucleus về hình dạng trung bình của thanh (z-score)
    if tone_prior is not None:
        tone = batch["features"][..., 4]
        prior = tone_prior[tone]  # [B,N,4]
        nucleus_weight = batch["f0_weight"][..., 2]
        prior_error = ((outputs["seg_shape"][..., 2, :] - prior) ** 2).sum(dim=-1)
        losses["f0_prior"] = (prior_error * nucleus_weight).sum() / nucleus_weight.sum().clamp_min(1)
        # T6.1 (§33.4.3) — CE *phân biệt* thanh từ chính contour dự đoán: logits = −d²/T với
        # d = khoảng cách (bình quân trên hệ số hợp lệ) tới prior của 6 thanh. Khác `f0_prior`
        # (chỉ *kéo về* prior đúng): số hạng này *đẩy ra* khỏi prior sai ⇒ thưởng biên phân biệt
        # (NGÃ vs SẮC/NẶNG — đo §33.3: model sai 1,5–2 nửa cung dù basis chỉ cần 13–65 cent).
        temperature = float(merged.get("tone_disc_temperature", TONE_DISC_TEMPERATURE))
        nucleus = outputs["seg_shape"][..., 2, :]                      # [B,N,4] — chỉ slot N
        kmask_n = batch["f0_k_mask"][..., 2, :].to(torch.float32)      # [B,N,4] hệ số hợp lệ
        distance = (((nucleus.unsqueeze(-2) - tone_prior.view(1, 1, -1, 4)) ** 2)
                    * kmask_n.unsqueeze(-2)).sum(-1) / kmask_n.sum(-1, keepdim=True).clamp_min(1.0)
        disc_mask = (nucleus_weight > 0) & kmask_n.any(dim=-1)
        if disc_mask.any():
            logits = -distance / max(temperature, 1e-3)
            losses["tone_disc"] = F.cross_entropy(logits[disc_mask], tone[disc_mask])
        else:
            losses["tone_disc"] = distance.sum() * 0.0

    losses["vuv"] = (
        F.binary_cross_entropy_with_logits(outputs["seg_vuv_logits"],
                                           (batch["seg_vuv"] >= 0.5).to(torch.float32),
                                           reduction="none") * active
    ).sum() / active.sum().clamp_min(1)
    losses["energy"] = ((outputs["seg_energy"] - batch["seg_energy"]).abs() * active).sum() / active.sum().clamp_min(1)

    if "seg_phonation" in outputs and "seg_phonation" in batch:
        pho_mask = batch["seg_phonation_mask"].unsqueeze(-1).to(torch.float32)
        denominator = pho_mask.sum() * outputs["seg_phonation"].size(-1)
        losses["phonation"] = (
            (outputs["seg_phonation"] - batch["seg_phonation"]).abs() * pho_mask
        ).sum() / denominator.clamp_min(1)

    losses["tone"] = F.cross_entropy(outputs["tone_logits"].transpose(1, 2), batch["features"][..., 4])

    # A1/§43.1 — gate pause: `pos_weight` (chống "học thuộc" toàn âm tính) + trọng số **khối lượng**
    # cho dương tính (`gap_frames = expm1(pause_log_gap)` vì target ghi `log1p(gap)` — v2_targets).
    pause_mask = batch["pause_mask"].to(torch.float32)
    gate_target = batch["pause_gate"]
    positive = (gate_target >= 0.5).to(torch.float32)
    gate_bce = F.binary_cross_entropy_with_logits(
        outputs["pause_logits"], gate_target, reduction="none",
        pos_weight=torch.as_tensor(float(merged.get("pause_pos_weight", PAUSE_POS_WEIGHT)),
                                dtype=torch.float32,
                                   device=gate_target.device))
    if "pause_log_gap" in batch:
        gap_frames = torch.expm1(batch["pause_log_gap"].to(torch.float32).clamp_min(0.0))
        mass_scale = float(merged.get("pause_mass_scale", PAUSE_MASS_SCALE))
        mass = (1.0 + gap_frames.clamp(0.0, PAUSE_MASS_CLIP_FRAMES) / mass_scale
                if mass_scale > 0 else torch.ones_like(gap_frames))
        gate_weight = 1.0 + positive * (mass - 1.0)  # âm tính giữ trọng số 1
    else:  # batch không có target gap (caller cũ) → không đổi hành vi
        gate_weight = torch.ones_like(gate_bce)
    losses["pause_gate"] = (gate_bce * gate_weight * pause_mask).sum() / pause_mask.sum().clamp_min(1)
    gap_mask = pause_mask * positive
    losses["pause_gap"] = (
        (outputs["pause_log_gap"] - batch["pause_log_gap"]) ** 2 * gap_mask
    ).sum() / gap_mask.sum().clamp_min(1)

    total = torch.zeros((), device=next(iter(losses.values())).device)
    for key, value in losses.items():
        total = total + value * float(merged.get(key, 1.0))
    return total, losses

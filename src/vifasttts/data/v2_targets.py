"""Mục tiêu train v2 — nền tảng T2.2 (docs/ACOUSTIC_V2.md §3, §6, §19.2; Addendum 6).

Nhận `row` (manifest `_slot4`) + feature cấp frame **đã cắt theo row** (cùng quy ước
`TtsDataset`: gốc 0 = `frame_range[0]` với row segment, = 0 với clip đầy đủ) và trả mọi
tensor target cấp segment cho một âm tiết:

- `seg_ratio` [N,4] + `seg_ratio_mask` [N,4] — target ratio cho loss CE/KL (B2/B5: mask
  inactive + `d_syl < 3`).
- `f0_shape` [N,4,4] (`s_k = c_k/√n` trên frame hữu thanh, z-score f0_stats_v2) +
  `f0_weight` [N,4] — đúng mask Addendum 6 (O cần onset hữu thanh ∧ ratio ≥ 0,5;
  M/C ratio ≥ 0,5; N luôn, trọng số = tỷ lệ hữu thanh) + `voiced_ratio` [N,4].
- `seg_energy` [N,4] (mean cửa sổ), `seg_vuv` [N,4].
- `pause_gate` [N], `pause_log_gap` [N], `pause_mask` [N] — gate = `gap ≥ 3` frame (§4).
  A2 (§43.1): **giữ nhãn cho vị trí cuối row** — manifest mới ghi đủ `gaps` (gap sau âm tiết
  cuối = pause ở mép cắt/lặng đuôi) và `frame_range` phủ các frame đó ⇒ bất biến
  `T == Σdur + Σgap`; row cũ (`len(gaps) == N-1`) tự động rơi vào `index >= len(gaps)` như trước.

Chạy trên CPU/long; chỉ dùng tensor op (đưa thẳng vào collate). Không phụ thuộc cache.
"""

from __future__ import annotations

import math

import torch

from vifasttts.data.features import dct4_frames
from vifasttts.frontend.segments import F0_VOICED_ONSETS

ROLES = ("O", "M", "N", "C")
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
NUCLEUS_INDEX = ROLE_INDEX["N"]
GATE_MIN_FRAMES = 3  # §4: gate BCE target `gap ≥ 3 frame`


def segment_windows(row: dict, *, relative: bool = False) -> list[list[tuple[str, int, int]]]:
    """Cửa sổ frame `[(role, start, end)]` cho từng âm tiết từ `seg_durations`.

    `relative=False` → toạ độ tuyệt đối trong clip (như T2.1/R5b); `relative=True` →
    toạ độ trong feature **đã cắt theo row** (segment: bắt đầu 0; clip đầy đủ: `lead_frames`).
    """
    count = len(row["durations"])
    gaps = list(row.get("gaps") or [])
    if len(gaps) == count - 1:
        gaps = gaps + [0]
    if relative:
        cursor = 0 if "frame_range" in row else int(row.get("lead_frames") or 0)
    else:
        cursor = int(row["frame_range"][0]) if "frame_range" in row else int(row.get("lead_frames") or 0)
    seg_durations = row.get("seg_durations")
    windows: list[list[tuple[str, int, int]]] = []
    for index in range(count):
        segment = seg_durations[index] if seg_durations is not None else None
        inner = cursor
        syllable: list[tuple[str, int, int]] = []
        for role_index, role in enumerate(ROLES):
            length = int(segment[role_index]) if segment is not None else 0
            if length > 0:
                syllable.append((role, inner, inner + length))
            inner += length
        windows.append(syllable)
        cursor += int(row["durations"][index]) + (int(gaps[index]) if index < len(gaps) else 0)
    return windows


def structural_active(syllable: dict) -> list[bool]:
    """Slot active theo **cấu trúc abstract** (không theo seg_durations) — mask chuẩn của model."""
    abstract = syllable["abstract"]
    return [abstract["onset"] != "NONE", abstract["medial"] != "NONE", True,
            abstract["coda"] != "NONE"]


def ratio_targets(row: dict, *, min_frames: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """(target ratio [N,4], mask [N,4] bool) — mask inactive + `d_syl < min_frames` (B5)."""
    count = len(row["durations"])
    target = torch.zeros(count, 4, dtype=torch.float32)
    mask = torch.zeros(count, 4, dtype=torch.bool)
    for index in range(count):
        duration = float(row["durations"][index])
        if duration <= 0:
            continue
        segment = row["seg_durations"][index]
        target[index] = torch.tensor([value / duration for value in segment], dtype=torch.float32)
        if duration >= min_frames:
            mask[index] = torch.tensor(structural_active(row["syllables"][index]), dtype=torch.bool)
    return target, mask


def pause_targets(row: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(gate [N], log_gap [N], mask [N]) — **mọi** vị trí có gap trong row đều có nhãn (A2/§43.1).

    Trước đây bỏ `index ≥ count-1` (rìa row bị cắt) ⇒ 46,5% khối lượng pause không có nhãn.
    Manifest A2 ghi đủ `gaps` + `frame_range` phủ pause đuôi (bất biến `T == Σdur + Σgap`)
    nên vị trí cuối là nhãn **thật**; row cũ (`len(gaps) == N-1`) vẫn bị bỏ ở vị trí cuối.
    """
    count = len(row["durations"])
    gaps = list(row.get("gaps") or [])
    gate = torch.zeros(count, dtype=torch.float32)
    log_gap = torch.zeros(count, dtype=torch.float32)
    mask = torch.zeros(count, dtype=torch.bool)
    for index in range(count):
        if index >= len(gaps):
            continue  # row không ghi gap sau âm tiết này ⇒ không bịa nhãn
        gap = int(gaps[index])
        gate[index] = float(gap >= GATE_MIN_FRAMES)
        log_gap[index] = math.log1p(max(0, gap))
        mask[index] = True
    return gate, log_gap, mask


def _pool(values: torch.Tensor, row: dict, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Mean (qua `valid`) của `values` [T] theo cửa sổ từng role → [N,4]."""
    values = values.float()
    output = torch.zeros(len(row["durations"]), 4, dtype=torch.float32)
    for index, windows in enumerate(segment_windows(row, relative=True)):
        for role, low, high in windows:
            low = max(low, 0)
            high = min(high, values.size(0))
            if high <= low:
                continue
            window = values[low:high]
            if valid is not None:
                window = window[valid[low:high]]
            if window.numel():
                output[index, ROLE_INDEX[role]] = window.mean()
    return output


def segment_pools(row: dict, energy: torch.Tensor, vuv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(energy [N,4] mean cửa sổ, voiced_ratio [N,4])."""
    energy_pool = _pool(energy, row)
    ratio = _pool((vuv.float() > 0.5).float(), row)
    return energy_pool, ratio


def f0_targets(row: dict, f0: torch.Tensor, vuv: torch.Tensor, *, f0_mean: float, f0_std: float,
               keep: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(s_k [N,4,keep], weight [N,4], voiced_ratio [N,4], k_mask [N,4,keep] bool) — mask Addendum 6 + F4.

    - `s_k = c_k/√n` trên frame hữu thanh (`vuv > 0,5` và `f0 > 0`), `n ≥ 2`;
    - `k_mask`: hệ số nào **có thật** (k < n) — loss phải mask phần thiếu (B1);
    - trọng số: N = tỷ lệ hữu thanh (luôn); O = 1 nếu onset ∈ `F0_VOICED_ONSETS` ∧ ratio ≥ 0,5;
      M/C = 1 nếu ratio ≥ 0,5; còn lại 0 (không tính loss).
    """
    count = len(row["durations"])
    shape = torch.zeros(count, 4, keep, dtype=torch.float32)
    weight = torch.zeros(count, 4, dtype=torch.float32)
    voiced_ratio = torch.zeros(count, 4, dtype=torch.float32)
    k_mask = torch.zeros(count, 4, keep, dtype=torch.bool)
    for index, windows in enumerate(segment_windows(row, relative=True)):
        syllable = row["syllables"][index]
        onset = syllable["abstract"]["onset"]
        for role, low, high in windows:
            low = max(low, 0)
            high = min(high, f0.size(0))
            if high <= low:
                continue
            window_f0 = f0[low:high].float()
            voiced = (vuv[low:high].float() > 0.5) & (window_f0 > 1e-3)
            voiced_count = int(voiced.sum())
            ratio = float(voiced.float().mean())
            column = ROLE_INDEX[role]
            voiced_ratio[index, column] = ratio
            if voiced_count < 2:
                continue
            z = (torch.log(window_f0[voiced]) - f0_mean) / max(f0_std, 1e-6)
            coefficients = dct4_frames(z, keep=keep) / math.sqrt(voiced_count)
            shape[index, column, :coefficients.numel()] = coefficients.float()
            k_mask[index, column, :coefficients.numel()] = True
            if role == "N":
                weight[index, column] = ratio
            elif role == "M":
                weight[index, column] = 1.0 if ratio >= 0.5 else 0.0
            elif role == "O":
                weight[index, column] = 1.0 if (onset in F0_VOICED_ONSETS and ratio >= 0.5) else 0.0
            else:  # C — §6 giữ nguyên, coda tắc không bung hơi (Addendum 6)
                weight[index, column] = 1.0 if ratio >= 0.5 else 0.0
    return shape, weight, voiced_ratio, k_mask


PHONATION_CHANNELS = ("hnr", "glottalization", "jitter")


def phonation_targets(row: dict, features: dict, vuv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(seg_phonation [N,4,3], mask [N,4]) — mean kênh phonation trên frame hữu thanh.

    Mask = (≥ 2 frame hữu thanh) ∧ (tỷ lệ hữu thanh ≥ 0,5) — cùng triết lý mask F0 (§19.2).
    Thiếu kênh trong `features` → trả zeros/mask 0 (dataset cũ không có phonation vẫn chạy).
    """
    count = len(row["durations"])
    output = torch.zeros(count, 4, len(PHONATION_CHANNELS), dtype=torch.float32)
    mask = torch.zeros(count, 4, dtype=torch.bool)
    channels = [features.get(name) for name in PHONATION_CHANNELS]
    if any(channel is None for channel in channels):
        return output, mask
    for index, windows in enumerate(segment_windows(row, relative=True)):
        for role, low, high in windows:
            low = max(low, 0)
            high = min(high, vuv.size(0))
            if high <= low:
                continue
            voiced = vuv[low:high].float() > 0.5
            voiced_count = int(voiced.sum())
            if voiced_count < 2 or float(voiced.float().mean()) < 0.5:
                continue
            column = ROLE_INDEX[role]
            for channel_index, channel in enumerate(channels):
                output[index, column, channel_index] = channel[low:high][voiced].float().mean()
            mask[index, column] = True
    return output, mask


def segment_targets(row: dict, features: dict, f0_stats: dict) -> dict[str, torch.Tensor]:
    """Gộp mọi target cấp segment của một row (feature đã cắt theo row).

    Keys: `seg_ratio`/`seg_ratio_mask` · `f0_shape`/`f0_weight`/`f0_k_mask` ·
    `seg_energy`/`seg_vuv` · `seg_phonation`/`seg_phonation_mask` (khi features có kênh) ·
    `pause_gate`/`pause_log_gap`/`pause_mask`.
    """
    ratio, ratio_mask = ratio_targets(row)
    gate, log_gap, pause_mask = pause_targets(row)
    energy, voiced_ratio = segment_pools(row, features["energy"], features["vuv"])
    shape, f0_weight, _, k_mask = f0_targets(row, features["f0"], features["vuv"],
                                             f0_mean=float(f0_stats["f0_log_mean"]),
                                             f0_std=float(f0_stats["f0_log_std"]))
    phonation, phonation_mask = phonation_targets(row, features, features["vuv"])
    return {
        "seg_ratio": ratio,
        "seg_ratio_mask": ratio_mask,
        "f0_shape": shape,
        "f0_weight": f0_weight,
        "f0_k_mask": k_mask,
        "seg_energy": energy,
        "seg_vuv": voiced_ratio,
        "seg_phonation": phonation,
        "seg_phonation_mask": phonation_mask,
        "pause_gate": gate,
        "pause_log_gap": log_gap,
        "pause_mask": pause_mask,
    }

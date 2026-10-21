"""Chuẩn hoá độ to theo TỪNG CHUNK trên log-mel (A0 v10, Việc 4/6d).

Vì sao: model tự chọn mức to tuyệt đối cho mỗi chunk, nên mức đó trôi theo số bước GAN.
Cách sửa rẻ nhất: cộng một hằng số vào log-mel của chunk để mức hữu thanh về mức mục tiêu
(trung vị của giọng thật, đo bằng `tools/measure_voiced_level.py`), giới hạn ±4 dB.

Quy ước (khớp `tools/measure_voiced_level.py`):
  level_frame = ln(sum_b exp(2·logmel_b))   → dB = level · 10/ln(10)
  log-mel ở đây là log biên độ (natural log), nên dịch Δln làm năng lượng × e^{2Δln}:
  ΔdB = 8,686·Δln  ⇒  Δln = ΔdB / 8,686.
"""

from __future__ import annotations

import numpy as np

DB_PER_LN = 10.0 / np.log(10.0)  # 4,3429: ln(năng lượng) → dB
LN_PER_DB = np.log(10.0) / 20.0  # 0,115129: dB biên độ → ln biên độ (1/8,686)


def mel_level_db(mel_z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Mức dB từng frame từ log-mel đã z-norm."""
    mel = np.asarray(mel_z, dtype=np.float64)
    linear = np.exp(2.0 * (mel * std + mean))
    return np.log(np.sum(linear, axis=-1) + 1e-12) * DB_PER_LN


def voiced_frames(level_db: np.ndarray, floor_db: float = 30.0) -> np.ndarray:
    """Frame 'hữu thanh' = trên đỉnh − 30 dB (cùng quy ước `premise_drift`)."""
    return level_db > (float(np.max(level_db)) - floor_db)


def normalize_chunk(
    mel_z: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    target_db: float,
    *,
    max_shift_db: float = 4.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Dịch đều log-mel của chunk để mức hữu thanh về `target_db` (giới hạn ±max_shift_db)."""
    mel = np.asarray(mel_z, dtype=np.float32)
    level = mel_level_db(mel, mean, std)
    if level.size == 0:
        return mel, {"level_db": float("nan"), "shift_db": 0.0}
    frames = voiced_frames(level)
    if not frames.any():
        return mel, {"level_db": float("nan"), "shift_db": 0.0}
    current = float(np.median(level[frames]))
    shift_db = float(np.clip(target_db - current, -max_shift_db, max_shift_db))
    # log-mel = z*std + mean ⇒ cộng Δln vào log-mel = cộng Δln/std vào z (std khác từng bin).
    shifted = mel + np.float32(shift_db * LN_PER_DB) / np.asarray(std, dtype=np.float32)
    return shifted, {"level_db": round(current, 2), "shift_db": round(shift_db, 2)}

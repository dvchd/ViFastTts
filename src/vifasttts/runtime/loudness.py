"""Chuẩn hoá độ to CẢ FILE theo ITU-R BS.1770 (A0 v10, Việc 2 / 6f).

Vì sao: đầu ra hiện ~−26 dBFS còn VieNeu ~−16 LUFS, tai nghe nhỏ bằng một nửa — đó là mức của
dữ liệu chứ không phải lỗi model. Nhân MỘT hệ số cho cả file (không nén động) để về `target_lufs`,
sau đó hạ thêm nếu true-peak vượt `peak_dbtp` (limiter tĩnh, không méo, tất định).

Đo loudness: lọc K-weighting chuẩn 48 kHz trên bản resample 2× (24→48 kHz), block 400 ms chồng 75 %,
gate tuyệt đối −70 LUFS + gate tương đối −10 LU. True-peak: oversample 4× rồi lấy đỉnh.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

# Hệ số K-weighting chuẩn BS.1770-4 tại 48 kHz.
_SHELF_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
_SHELF_A = np.array([1.0, -1.69065929318241, 0.73248077421585])
_HIPASS_B = np.array([1.0, -2.0, 1.0])
_HIPASS_A = np.array([1.0, -1.99004745483398, 0.99007225036621])


def _biquad(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Lọc biquad trực tiếp dạng II (a[0] = 1), numpy thuần — tất định."""
    y = np.empty_like(x)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for i in range(x.size):
        xi = float(x[i])
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return y


def measure_lufs(waveform: np.ndarray, sample_rate: int) -> float:
    """Integrated loudness (LUFS) theo BS.1770-4. Trả −inf nếu file toàn lặng."""
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("-inf")
    if sample_rate != 48000:
        x48 = resample_poly(x, 48000, sample_rate)
    else:
        x48 = x
    y = _biquad(_biquad(x48, _SHELF_B, _SHELF_A), _HIPASS_B, _HIPASS_A)
    block = int(0.4 * 48000)
    step = int(0.1 * 48000)
    if y.size < block:
        z = np.array([np.mean(y ** 2)]) if y.size else np.array([])
    else:
        z = np.array([np.mean(y[i:i + block] ** 2) for i in range(0, y.size - block + 1, step)])
    if z.size == 0:
        return float("-inf")
    loud = -0.691 + 10.0 * np.log10(z + 1e-30)
    keep = loud > -70.0
    if not keep.any():
        return float("-inf")
    rel = -0.691 + 10.0 * np.log10(np.mean(z[keep]) + 1e-30) - 10.0
    keep &= loud > rel
    if not keep.any():
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(np.mean(z[keep]) + 1e-30))


def true_peak_dbtp(waveform: np.ndarray, sample_rate: int, oversample: int = 4) -> float:
    """True-peak xấp xỉ bằng oversample 4× (dBTP)."""
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return float("-inf")
    up = resample_poly(x, oversample, 1)
    peak = float(np.max(np.abs(up))) if up.size else 0.0
    return float(20.0 * np.log10(peak + 1e-30))


def _limit_true_peak(y: np.ndarray, sample_rate: int, *, ceiling_db: float = -1.0,
                     lookahead_ms: float = 5.0, release_ms: float = 80.0) -> np.ndarray:
    """Limiter look-ahead (tất định): chỉ hạ đúng vùng đỉnh, không hạ cả file.

    Vì sao: đỉnh −1 dBTP của giọng nói thường chỉ là một xung vài ms (bật hơi/plosive).
    Hạ tĩnh cả file sẽ làm LUFS rơi 4–5 dB và mỗi câu một mức to khác nhau.
    """
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    ceil = 10.0 ** (ceiling_db / 20.0)
    up = resample_poly(x, 4, 1)
    if up.size == 0 or float(np.max(np.abs(up))) <= ceil:
        return np.asarray(y, dtype=np.float32)

    gain = np.minimum(1.0, ceil / np.maximum(np.abs(up), 1e-12))
    # Cửa sổ look-ahead: gain tại i = min(gain[i : i+L]) để hạ TRƯỚC khi tới đỉnh.
    window = max(1, int(4 * sample_rate * lookahead_ms / 1000.0))
    from scipy.ndimage import minimum_filter1d

    padded = np.concatenate([gain, np.ones(window, dtype=np.float64)])
    ahead = minimum_filter1d(padded, size=window, mode="nearest")[: gain.size]
    # Nhả gain chậm (release), hạ thì tức thì.
    release = float(np.exp(-1.0 / max(1.0, 4 * sample_rate * release_ms / 1000.0)))
    smoothed = np.empty_like(ahead)
    previous = 1.0
    for i in range(ahead.size):
        value = ahead[i]
        previous = value if value < previous else release * previous + (1.0 - release) * value
        smoothed[i] = previous
    # Về lại 24 kHz: lấy gain nhỏ nhất trong 4 mẫu (an toàn cho true-peak).
    trimmed = smoothed[: (smoothed.size // 4) * 4].reshape(-1, 4).min(axis=1)
    if trimmed.size < x.size:
        trimmed = np.concatenate([trimmed, np.full(x.size - trimmed.size, trimmed[-1])])
    return (x * trimmed[: x.size]).astype(np.float32)


def normalize_loudness(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    target_lufs: float = -16.0,
    peak_dbtp: float = -1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Nhân MỘT hệ số cho cả file về `target_lufs`, rồi limiter giữ true-peak ≤ `peak_dbtp`.

    Trả (waveform mới, thông tin đo). Không nén toàn dải ⇒ không đổi hình bao, tất định.
    """
    x = np.asarray(waveform, dtype=np.float32).reshape(-1).copy()
    lufs_in = measure_lufs(x, sample_rate)
    tp_in = true_peak_dbtp(x, sample_rate)
    if not np.isfinite(lufs_in) or target_lufs is None or target_lufs == 0:
        return x, {"lufs_in": lufs_in, "lufs_out": lufs_in, "gain_db": 0.0,
                   "tp_in": tp_in, "tp_out": tp_in}
    gain_db = float(target_lufs - lufs_in)
    y = x
    for _ in range(4):  # lặp: bù lại phần limiter đã lấy, để LUFS về đúng target
        y = (x * np.float32(10.0 ** (gain_db / 20.0))).astype(np.float32)
        for _ in range(3):
            if true_peak_dbtp(y, sample_rate) <= peak_dbtp:
                break
            y = _limit_true_peak(y, sample_rate, ceiling_db=peak_dbtp)
        achieved = measure_lufs(y, sample_rate)
        if not np.isfinite(achieved) or abs(achieved - target_lufs) < 0.1:
            break
        gain_db += float(target_lufs - achieved)
    return y, {
        "lufs_in": round(lufs_in, 2),
        "lufs_out": round(measure_lufs(y, sample_rate), 2),
        "gain_db": round(gain_db, 2),
        "tp_in": round(tp_in, 2),
        "tp_out": round(true_peak_dbtp(y, sample_rate), 2),
    }

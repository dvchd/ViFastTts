"""tools/tone_ruler.py — "thước đo thanh điệu" (6 thanh) đo trên **audio THẬT**.

Lý do tồn tại: bộ chấm cũ `tools/eval_tone_pairs.py` (harvest F0 → DCT-4 → gần prior
nhất) chỉ đạt **0,46** accuracy trên chính audio thật (xem `artifacts/tone_path/diagnosis.json`)
— quá yếu để xác nhận cải tiến. Tool này **học** bộ phân loại từ ~220k âm tiết audio thật
(train manifest) rồi chấm val (khác clip cha → không rò rỉ), làm **trần tham chiếu** cho
bản tổng hợp (model → Vocos) sau này.

Đường đi (một code path duy nhất cho train + chấm):

1. `raw_record(...)` — mỗi âm tiết: cửa sổ **nucleus** (slot N) và/hoặc **syllable** (O..C);
   trên các frame hữu thanh (`vuv > 0,5`): log-F0 → z-score bằng `f0_log_mean/std`
   (`data/codelung/f0_stats_v2.json`) → **nội suy theo thời gian** về `CONTOUR_MAX=16` điểm;
   thêm thống kê (độ dài, tỉ lệ hữu thanh, phương sai), phonation (hnr/jitter/glottalization
   — toàn bộ + đoạn cuối `final_frac`), energy.
2. `assemble_features(...)` — chọn nhóm đặc trưng theo `spec` → ma trận `[n, F]`.
3. MLP nhỏ (torch, **CPU** — MPS đang bận train) + class weight cho NGÃ/NẶNG ít.

Cảnh báo trung thực: đây là thước đo **thanh điệu** trên F0/creak, không phải "thước đo
đúng chữ"; nó chấm âm tiết đã biết vị trí. Cửa sổ nucleus lấy từ `seg_durations` của
manifest `_slot4` (T2.1, ưu tiên hybrid) — nếu segmentation lệch thì contour lệch.

KẾT QUẢ ĐO (val manifest 41.521 âm tiết → giữ 35.572 = 85,7%; `tone_ruler.json` có số đầy đủ):

  | bộ chấm                                    | accuracy |
  |--------------------------------------------|----------|
  | chance (6 thanh)                           |   0,167  |
  | thước CŨ (`eval_tone_pairs`: DCT-4 → prior)|   0,406  |
  | chỉ độ dài (MLP, 6 cột)                    |   0,244  |
  | chỉ contour F0 (MLP, 62 cột)               |   0,626  |
  | **tone_ruler, đủ đặc trưng (ensemble 3)**  | **0,762**|

  Per-thanh: NGANG 0,752 · HUYEN 0,756 · SAC 0,744 · HOI 0,847 · NGA 0,773 · NANG 0,780.
  Kiến trúc: MLP 384×3 lớp, dropout 0,15, 60 epoch, 3 seed soft-vote (single seed ≈ 0,757).

  **Kiểm chứng holdout** (842 clip CHƯA từng dùng, `holdout_clean_v2_dur_slot4.jsonl`,
  29.959 âm tiết): **0,7618** — khớp val 0,7623 (chênh 0,0005) → con số không phải do
  chỉnh trên val (`score_real_holdout.json`).

  Cảnh báo (đọc trước khi tin con số):
  - Nhãn là **weak label từ chính tả** (R2 ~98,3%): phần "sai" gồm cả ca phát âm lệch chuẩn
    tắc (phương ngữ, nói nhanh) → 0,762 là **trần thực tế của thước**, không phải lỗi model.
  - Val được dùng cho early-stop mềm (giữ epoch tốt nhất) → số báo cáo lạc quan nhẹ.
  - `phonation` (hnr/jitter/glottalization) là nhóm đặc trưng đắt giá nhất (bỏ → −2,2 điểm);
    chấm audio **tổng hợp** phải trích đủ kênh (`features_from_wav`), thiếu → cảnh báo + giảm điểm.
  - `span`/`voiced_ratio` tương quan cấu trúc vần (âm tiết kết thúc tắc ngắn hơn) — text-free
    nhưng không thuần "thanh"; may là nhóm chỉ-độ-dài chỉ đạt 0,200 (gần chance).

Ví dụ:
  python tools/tone_ruler.py --train
  python tools/tone_ruler.py --score-real            # mặc định manifest val
  python tools/tone_ruler.py --score-real --manifest data/codelung/val_clean_v2_dur_seg_slot4.jsonl

API cho tool khác (chấm audio tổng hợp):
  from tone_ruler import ToneRuler
  ruler = ToneRuler.load("artifacts/tone_path/tone_ruler.pt")
  ruler.score_items([item])  # item: f0/vuv/(hnr/jitter/glottalization)/seg_durations hoặc slot_durations
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Thứ tự chuẩn (khớp `ComponentVocabs.default().tone` và `eval_tone_pairs.TONES`).
TONES = ["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]
TONE_TO_ID = {tone: index for index, tone in enumerate(TONES)}
CHANCE = 1.0 / len(TONES)
CONTOUR_MAX = 16
DEFAULT_F0_STATS = "data/codelung/f0_stats_v2.json"
DEFAULT_CACHE_DIR = "cache/codelung_v2"
DEFAULT_CONFIG = "configs/small_codelung_v2.yaml"
DEFAULT_OUT_DIR = "artifacts/tone_path"
DEFAULT_MODEL = "artifacts/tone_path/tone_ruler.pt"
DEFAULT_TRAIN_MANIFEST = "data/codelung/train_clean_v2_dur_seg_slot4.jsonl"
DEFAULT_VAL_MANIFEST = "data/codelung/val_clean_v2_dur_seg_slot4.jsonl"

#: Các kênh phonation (thứ tự khớp `v2_targets.PHONATION_CHANNELS`) — thiếu thì pad 0 + cờ 0.
PHONATION_KEYS = ("hnr", "glottalization", "jitter")

#: Trường scalar của một `raw_record` (mọi trường luôn có; thiếu dữ liệu → 0).
#: Nhóm "ngữ cảnh" (`dur_rel`/`z_mean_rel`/`energy_z`/`dprev`/`dnext`/`pos_rel`) được tính ở
#: cấp item (câu/segment) rồi **ghi thẳng vào record** — nhờ vậy train và chấm dùng đúng một
#: cách chuẩn hoá (không lệch scaler giữa hai đường).
RECORD_SCALARS = (
    "span", "span_syl", "n_voiced", "voiced_ratio", "voiced_ratio_syl", "voiced_start_rel",
    "voiced_end_rel", "z_mean", "z_std", "z_min", "z_max", "z_mean_syl", "z_std_syl",
    "hnr_all", "jitter_all", "glott_all", "hnr_last", "jitter_last", "glott_last",
    "hnr_syl_last", "jitter_syl_last", "glott_syl_last", "phon_ok", "phon_ok_syl",
    "energy_mean", "energy_slope", "rate_first", "rate_last", "rate_total",
    "dur_rel", "z_mean_rel", "z_mean_rel_n", "range_z_n", "energy_z", "dprev", "dnext",
    "pos_rel",
)

#: Spec mặc định — chốt sau thực nghiệm (xem `artifacts/tone_path/tone_ruler.json`).
FEATURE_SPEC: dict = {
    "version": "tone-ruler-v1",
    "window": "nucleus",          # cửa sổ chính để đo contour: nucleus | syllable
    "contour_source": "both",     # contour đưa vào vector: nucleus | syllable | both
    "contour_points": 16,
    "final_frac": 0.4,            # tỉ lệ frame hữu thanh cuối dùng cho thống kê creak
    "groups": ("basic", "syl", "level", "contour", "contour_rel", "dct", "slope", "rate",
               "phonation", "context", "energy"),
    "min_span": 4,                # bỏ âm tiết quá ngắn (frame)
    "min_voiced": 2,              # bỏ âm tiết quá ít frame hữu thanh
}


def _np(values) -> np.ndarray:
    """Tensor (CPU) / list / ndarray → numpy float64."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def load_f0_stats(path: str | Path = DEFAULT_F0_STATS) -> dict:
    """Đọc `f0_stats_v2.json` (chỉ cần `f0_log_mean`/`f0_log_std`)."""
    stats = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("f0_log_mean", "f0_log_std"):
        if key not in stats:
            raise ValueError(f"f0_stats thiếu {key}: {path}")
    return stats


def seg_durations_from_slot_durations(slot_durations) -> tuple[int, np.ndarray, np.ndarray]:
    """Layout phẳng `[lead, (O,M,N,C,G)×N]` (model sinh) → `(lead, seg [N,4], gaps [N])`.

    Dùng cho tool chấm audio tổng hợp (`eval_tone_pairs` cũng dùng layout này): O/M/N/C → `seg`,
    G (gap sau mỗi âm tiết) → `gaps` — **phải giữ gap** vì cửa sổ các âm tiết sau phụ thuộc gap.
    """
    flat = [int(v) for v in _np(slot_durations).round().astype(np.int64).tolist()]
    if len(flat) < 6:
        raise ValueError(f"slot_durations quá ngắn: {flat}")
    lead = flat[0]
    body = flat[1:]
    n_syl = len(body) // 5
    seg = np.zeros((n_syl, 4), dtype=np.int64)
    gaps = np.zeros(n_syl, dtype=np.int64)
    for index in range(n_syl):
        seg[index] = body[5 * index:5 * index + 4]
        gaps[index] = body[5 * index + 4]
    return lead, seg, gaps


def seg_layout(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Chuẩn hoá item → `(seg [N,4], durations [N], gaps [N], lead)`.

    Nhận `seg_durations` (thứ tự O,M,N,C) hoặc `slot_durations` phẳng `[lead,(O,M,N,C,G)×N]`.
    `durations`/`gaps` thiếu thì suy từ `seg_durations` (gap = 0).
    """
    if item.get("seg_durations") is not None:
        seg = _np(item["seg_durations"]).astype(np.int64).reshape(-1, 4)
        lead = int(item.get("lead", 0) or 0)
        gaps = item.get("gaps")
        gaps = _np(gaps).astype(np.int64) if gaps is not None else None
    elif item.get("slot_durations") is not None:
        lead, seg, gaps = seg_durations_from_slot_durations(item["slot_durations"])
    else:
        raise ValueError("item thiếu cả `seg_durations` lẫn `slot_durations`")
    count = seg.shape[0]
    durations = item.get("durations")
    durations = (_np(durations).astype(np.int64) if durations is not None
                 else seg.sum(axis=1))
    if gaps is not None and gaps.size == count - 1:
        gaps = np.concatenate([gaps, np.zeros(1, dtype=np.int64)])
    if gaps is None or gaps.size < count:
        pad = np.zeros(count - (0 if gaps is None else gaps.size), dtype=np.int64)
        gaps = pad if gaps is None else np.concatenate([gaps, pad])
    return seg, durations[:count], gaps[:count], lead


def seg_windows(item: dict, mode: str = "nucleus") -> list[tuple[int, int]]:
    """Cửa sổ `(start, end)` row-relative cho từng âm tiết (khớp `v2_targets.segment_windows`).

    `mode="nucleus"`: chỉ slot N (`start = lead + Σtrước + O + M`); `"syllable"`: cả O..C.
    """
    seg, durations, gaps, lead = seg_layout(item)
    starts = int(lead) + np.concatenate(
        [[0], np.cumsum((durations + gaps)[:-1])]).astype(np.int64)
    windows: list[tuple[int, int]] = []
    for index in range(seg.shape[0]):
        if mode == "nucleus":
            start = int(starts[index]) + int(seg[index, 0]) + int(seg[index, 1])
            end = start + int(seg[index, 2])
        else:
            start = int(starts[index])
            end = start + int(seg[index].sum())
        windows.append((start, max(start, end)))
    return windows


def resample_contour(values: np.ndarray, positions: np.ndarray, points: int) -> np.ndarray:
    """Nội suy tuyến tính contour về `points` điểm (theo **thời gian** trong cửa sổ).

    `values`: giá trị tại các frame hữu thanh; `positions`: vị trí chuẩn hoá [0,1] của chúng.
    Thiếu frame → trả vector 0 (caller tự mask bằng `n_voiced`).
    """
    out = np.zeros(int(points), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return out
    if values.size == 1:
        return np.full(int(points), float(values[0]), dtype=np.float64)
    grid = np.linspace(0.0, 1.0, int(points))
    return np.interp(grid, np.asarray(positions, dtype=np.float64), values)


def _window_stats(f0_hz: np.ndarray, vuv: np.ndarray, z: np.ndarray, window: tuple[int, int],
                  points: int) -> dict:
    """Thống kê một cửa sổ: contour z-log-F0 + vị trí hữu thanh (row-relative frame)."""
    total = f0_hz.size
    start = int(min(max(window[0], 0), total))
    end = int(min(max(window[1], 0), total))
    result = {"span": float(max(0, end - start)), "n_voiced": 0.0, "voiced_ratio": 0.0,
              "voiced_start_rel": 0.0, "voiced_end_rel": 0.0, "z_mean": 0.0, "z_std": 0.0,
              "z_min": 0.0, "z_max": 0.0, "contour": np.zeros(points)}
    if end <= start:
        return result
    span = end - start
    voiced = vuv[start:end] > 0.5
    count = int(voiced.sum())
    result["span"] = float(span)
    result["n_voiced"] = float(count)
    result["voiced_ratio"] = float(count / span)
    if count == 0:
        return result
    index = np.flatnonzero(voiced)
    relative = (index + 0.5) / span
    result["voiced_start_rel"] = float(relative[0])
    result["voiced_end_rel"] = float(relative[-1])
    values = z[start:end][voiced]
    result["z_mean"] = float(values.mean())
    result["z_std"] = float(values.std())
    result["z_min"] = float(values.min())
    result["z_max"] = float(values.max())
    result["contour"] = resample_contour(values, relative, points)
    return result


def raw_record(
    f0_hz, vuv, window: tuple[int, int], *,
    hnr=None, jitter=None, glottalization=None, energy=None,
    syllable_window: tuple[int, int] | None = None,
    f0_log_mean: float = 4.803942, f0_log_std: float = 0.167907,
    final_frac: float = 0.4, points: int = CONTOUR_MAX,
) -> dict:
    """Đặc trưng "thô" một âm tiết: contour + thống kê (chưa chuẩn hoá theo ngữ cảnh).

    `window` = cửa sổ chính (nucleus); `syllable_window` = cửa sổ cả âm tiết O..C (tuỳ chọn).
    `hnr`/`jitter`/`glottalization`/`energy` là kênh cấp frame (row-relative) — thiếu thì 0.
    """
    f0 = np.maximum(_np(f0_hz).reshape(-1), 1e-6)
    vuv_arr = _np(vuv).reshape(-1)
    z = (np.log(f0) - float(f0_log_mean)) / max(float(f0_log_std), 1e-6)
    base = _window_stats(f0, vuv_arr, z, window, points)
    record = {key: float(base[key]) for key in
              ("span", "n_voiced", "voiced_ratio", "voiced_start_rel", "voiced_end_rel",
               "z_mean", "z_std", "z_min", "z_max")}
    record["contour"] = base["contour"]
    other = None
    if syllable_window is None or tuple(syllable_window) == tuple(window):
        other = base
    else:
        other = _window_stats(f0, vuv_arr, z, syllable_window, points)
    record["span_syl"] = float(other["span"])
    record["voiced_ratio_syl"] = float(other["voiced_ratio"])
    record["z_mean_syl"] = float(other["z_mean"])
    record["z_std_syl"] = float(other["z_std"])
    record["contour_syl"] = other["contour"]

    # Tốc độ đổi cao độ trên **thời gian thật** (frame) — contour đã chuẩn hoá thời gian nên
    # tốc độ tuyệt đối là thông tin riêng (SAC/HUYỀN/NẶNG khác nhau ở độ dốc nhanh/chậm).
    record["rate_first"] = 0.0
    record["rate_last"] = 0.0
    record["rate_total"] = 0.0
    start, end = int(window[0]), int(window[1])
    start = min(max(start, 0), vuv_arr.size)
    end = min(max(end, 0), vuv_arr.size)
    voiced = np.flatnonzero(vuv_arr[start:end] > 0.5) if end > start else np.zeros(0, np.int64)
    if voiced.size >= 2:
        values = z[start:end][voiced]
        split = max(1, values.size // 3)
        record["rate_total"] = float((values[-1] - values[0]) / max(values.size - 1, 1))
        record["rate_first"] = float((values[split] - values[0]) / max(split, 1))
        record["rate_last"] = float((values[-1] - values[-1 - split]) / max(split, 1))

    for name in ("hnr", "jitter", "glott"):
        record[f"{name}_all"] = 0.0
        record[f"{name}_last"] = 0.0
        record[f"{name}_syl_last"] = 0.0
    record["phon_ok"] = 0.0
    record["phon_ok_syl"] = 0.0
    channels = [_np(value).reshape(-1) if value is not None else None
                for value in (hnr, jitter, glottalization)]
    if all(channel is not None for channel in channels):
        if voiced.size:
            take_all = max(1, math.ceil(float(final_frac) * voiced.size))
            for channel, name in zip(channels, ("hnr", "jitter", "glott"), strict=True):
                record[f"{name}_all"] = float(channel[start:end][voiced].mean())
                record[f"{name}_last"] = float(channel[start:end][voiced[-take_all:]].mean())
            record["phon_ok"] = 1.0
        if syllable_window is not None:
            # Creak ở **cuối cả âm tiết** (kể cả coda): NGÃ/NẶNG hay kẹt thanh ở đây.
            syl_start = min(max(int(syllable_window[0]), 0), vuv_arr.size)
            syl_end = min(max(int(syllable_window[1]), 0), vuv_arr.size)
            syl_voiced = (np.flatnonzero(vuv_arr[syl_start:syl_end] > 0.5)
                          if syl_end > syl_start else np.zeros(0, np.int64))
            if syl_voiced.size:
                take = max(1, math.ceil(float(final_frac) * syl_voiced.size))
                for channel, name in zip(channels, ("hnr", "jitter", "glott"), strict=True):
                    record[f"{name}_syl_last"] = float(
                        channel[syl_start:syl_end][syl_voiced[-take:]].mean())
                record["phon_ok_syl"] = 1.0

    record["energy_mean"] = 0.0
    record["energy_slope"] = 0.0
    if energy is not None:
        values = _np(energy).reshape(-1)
        start, end = int(window[0]), int(window[1])
        start = min(max(start, 0), values.size)
        end = min(max(end, 0), values.size)
        if end > start:
            window_values = values[start:end]
            record["energy_mean"] = float(window_values.mean())
            half = max(1, window_values.size // 2)
            record["energy_slope"] = float(window_values[-half:].mean()
                                           - window_values[:half].mean())
    for key in ("dur_rel", "z_mean_rel", "z_mean_rel_n", "range_z_n", "energy_z", "dprev",
                "dnext", "pos_rel"):
        record[key] = 0.0
    return record


def apply_context(records: list[dict], context: dict | None) -> None:
    """Ghi các trường **phụ thuộc ngữ cảnh** vào record tại chỗ (`dur_rel`, `z_mean_rel`,
    `z_mean_rel_n`, `range_z_n`, `energy_z`).

    Ngữ cảnh = thống kê của cả item (câu/segment): level trung vị z-log-F0 các frame hữu thanh
    (± độ lệch chuẩn), độ dài âm tiết trung bình, energy mean/std. Nhờ vậy contour được so với
    **chính câu đó** (khử khác biệt speaker/ghi âm) mà vẫn giữ giá trị tuyệt đối ở nhóm `level`.
    """
    if not context:
        return
    dur = math.log1p(float(context.get("dur_mean", 0.0) or 0.0))
    level = float(context.get("level", 0.0) or 0.0)
    level_std = max(float(context.get("level_std", 0.0) or 0.0), 1e-6)
    energy_mean = float(context.get("energy_mean", 0.0) or 0.0)
    energy_std = float(context.get("energy_std", 0.0) or 0.0)
    for record in records:
        record["dur_rel"] = math.log1p(record["span"]) - dur if dur > 0 else 0.0
        record["z_mean_rel"] = record["z_mean"] - level
        record["z_mean_rel_n"] = record["z_mean_rel"] / level_std
        record["range_z_n"] = (record["z_max"] - record["z_min"]) / level_std
        record["energy_z"] = ((record["energy_mean"] - energy_mean) / energy_std
                              if energy_std > 0 else 0.0)


def item_records(item: dict, *, spec: dict = FEATURE_SPEC, f0_stats: dict | None = None
                 ) -> tuple[dict, list[tuple[int, int]], dict]:
    """Item → `(records, windows, info)`: raw record mọi âm tiết + ngữ cảnh cấp item.

    Ngữ cảnh (dùng cho `dur_rel`/`z_mean_rel`/`energy_z`/`dprev`/`dnext`): lấy trên **toàn item**
    (câu/segment) — với audio tổng hợp thì truyền cả câu cần chấm.
    """
    stats = f0_stats or load_f0_stats()
    f0 = _np(item["f0"]).reshape(-1)
    vuv = _np(item["vuv"]).reshape(-1)
    mode = str(spec.get("window", "nucleus"))
    windows = seg_windows(item, mode)
    syllable_windows = seg_windows(item, "syllable")
    channels = {name: item.get(name) for name in PHONATION_KEYS}
    energy = item.get("energy")
    records = []
    for window, syllable_window in zip(windows, syllable_windows, strict=True):
        records.append(raw_record(
            f0, vuv, window, hnr=channels["hnr"], jitter=channels["jitter"],
            glottalization=channels["glottalization"], energy=energy,
            syllable_window=syllable_window if mode != "syllable" else None,
            f0_log_mean=float(stats["f0_log_mean"]), f0_log_std=float(stats["f0_log_std"]),
            final_frac=float(spec.get("final_frac", 0.4)), points=CONTOUR_MAX))

    z = (np.log(np.maximum(f0, 1e-6)) - float(stats["f0_log_mean"])) / max(
        float(stats["f0_log_std"]), 1e-6)
    voiced = vuv > 0.5
    level = float(np.median(z[voiced])) if voiced.any() else 0.0
    spans = np.array([record["span"] for record in records]) if records else np.zeros(0)
    means = [record["z_mean"] for record in records]
    count = len(records)
    energy_values = None if energy is None else _np(energy).reshape(-1)
    context = {
        "level": level,
        "level_std": float(z[voiced].std()) if voiced.any() else 0.0,
        "dur_mean": float(spans.mean()) if spans.size else 0.0,
        "energy_mean": float(energy_values.mean()) if energy_values is not None
        and energy_values.size else 0.0,
        "energy_std": float(energy_values.std()) if energy_values is not None
        and energy_values.size else 0.0,
        "n_syllables": count,
    }
    apply_context(records, context)
    for index, record in enumerate(records):
        record["dprev"] = float(means[index] - means[index - 1]) if index > 0 else 0.0
        record["dnext"] = float(means[index] - means[index + 1]) if index + 1 < count else 0.0
        record["pos_rel"] = float(index / (count - 1)) if count > 1 else 0.0
    return records_matrix(records), windows, context


def records_matrix(records: list[dict]) -> dict:
    """List raw record → dict array: scalar `[n]` + `contour`/`contour_syl` `[n,16]`."""
    count = len(records)
    matrix = {key: np.array([float(record.get(key, 0.0)) for record in records], dtype=np.float64)
              for key in RECORD_SCALARS}
    for key in ("contour", "contour_syl"):
        matrix[key] = np.stack([np.asarray(record.get(key, np.zeros(CONTOUR_MAX)),
                                           dtype=np.float64) for record in records]) \
            if count else np.zeros((0, CONTOUR_MAX))
    return matrix


def features_from_spec(spec: dict) -> list[str]:
    """Tên đặc trưng theo thứ tự ghép (đi qua `feature_pairs` → không lệch cột)."""
    return [name for name, _values in feature_pairs(records_matrix([{}]), spec=spec)]


def feature_pairs(records: dict, *, spec: dict = FEATURE_SPEC) -> list[tuple[str, np.ndarray]]:
    """Danh sách `(tên, cột [n,1])` theo thứ tự ghép — nguồn duy nhất cho tên + vector.

    `assemble_features` và `features_from_spec` đều đi qua hàm này → tên không bao giờ lệch
    thứ tự với cột (baseline chọn cột theo tên).
    """
    count = len(records.get("span", []))
    groups = list(spec.get("groups", FEATURE_SPEC["groups"]))
    points = int(spec.get("contour_points", 12))
    source = str(spec.get("contour_source", "both"))
    pairs: list[tuple[str, np.ndarray]] = []

    def scalars(key: str) -> np.ndarray:
        return np.asarray(records.get(key, np.zeros(count)), dtype=np.float64)

    def cur(key: str) -> np.ndarray:
        values = records.get(key)
        if values is None or len(values) != count:
            return np.zeros((count, CONTOUR_MAX))
        return np.asarray(values, dtype=np.float64).reshape(count, -1)

    def column(name: str, values: np.ndarray) -> None:
        matrix = np.asarray(values, dtype=np.float64).reshape(count, -1)
        for index in range(matrix.shape[1]):
            pairs.append((f"{name}[{index}]" if matrix.shape[1] > 1 else name,
                          matrix[:, index:index + 1]))

    def primary_contour() -> np.ndarray:
        key = "contour_syl" if source == "syllable" else "contour"
        return _resample_matrix(cur(key), points)

    if "basic" in groups:
        column("log_span", np.log1p(scalars("span")))
        column("log_voiced", np.log1p(scalars("n_voiced")))
        column("voiced_ratio", scalars("voiced_ratio"))
        column("voiced_start_rel", scalars("voiced_start_rel"))
        column("voiced_end_rel", scalars("voiced_end_rel"))
        column("dur_rel", scalars("dur_rel"))
    if "syl" in groups:
        column("log_span_syl", np.log1p(scalars("span_syl")))
        column("voiced_ratio_syl", scalars("voiced_ratio_syl"))
        column("z_mean_syl_rel", scalars("z_mean_syl") - scalars("z_mean"))
        column("z_std_syl", scalars("z_std_syl"))
    if "level" in groups:
        column("z_mean", scalars("z_mean"))
        column("z_std", scalars("z_std"))
        column("z_mean_rel", scalars("z_mean_rel"))
        column("z_mean_rel_n", scalars("z_mean_rel_n"))
    if "contour" in groups:
        if source in ("nucleus", "both"):
            column("contour_nuc", _resample_matrix(cur("contour"), points))
        if source in ("syllable", "both"):
            column("contour_syl", _resample_matrix(cur("contour_syl"), points))
    if "contour_rel" in groups:
        contour = primary_contour()
        column("shape", contour - contour.mean(axis=1, keepdims=True))
    if "dct" in groups:
        shape = primary_contour()
        shape = shape - shape.mean(axis=1, keepdims=True)
        grid = np.linspace(0.0, 1.0, points)
        for order in range(1, 5):
            basis = np.cos(np.pi * order * (grid + 0.5 / points))
            column(f"dct{order}", (shape * basis).sum(axis=1, keepdims=True)
                   * math.sqrt(2.0 / points))
    if "slope" in groups:
        shape = primary_contour()
        shape = shape - shape.mean(axis=1, keepdims=True)
        grid = np.linspace(0.0, 1.0, points)
        centered = grid - grid.mean()
        third = max(1, points // 3)
        column("slope_first_last", shape[:, -1] - shape[:, 0])
        column("slope_linear", (shape * centered).sum(axis=1) / max(
            float((centered ** 2).sum()), 1e-9))
        column("range_rel", shape.max(axis=1) - shape.min(axis=1))
        column("range_z_n", scalars("range_z_n"))
        column("argmin_rel", shape.argmin(axis=1) / max(points - 1, 1))
        column("argmax_rel", shape.argmax(axis=1) / max(points - 1, 1))
        column("slope_tail", shape[:, -third:].mean(axis=1) - shape[:, :third].mean(axis=1))
    if "rate" in groups:
        column("rate_first", scalars("rate_first"))
        column("rate_last", scalars("rate_last"))
        column("rate_total", scalars("rate_total"))
    if "phonation" in groups:
        column("hnr_all", scalars("hnr_all"))
        column("jitter_all", scalars("jitter_all"))
        column("glott_all", scalars("glott_all"))
        column("hnr_last", scalars("hnr_last"))
        column("jitter_last", scalars("jitter_last"))
        column("glott_last", scalars("glott_last"))
        column("hnr_syl_last", scalars("hnr_syl_last"))
        column("jitter_syl_last", scalars("jitter_syl_last"))
        column("glott_syl_last", scalars("glott_syl_last"))
        column("phon_ok", scalars("phon_ok"))
    if "context" in groups:
        column("delta_prev", scalars("dprev"))
        column("delta_next", scalars("dnext"))
        column("pos_rel", scalars("pos_rel"))
    if "energy" in groups:
        column("energy_z", scalars("energy_z"))
        column("energy_slope", scalars("energy_slope"))
    return pairs


def assemble_features(records: dict, *, spec: dict = FEATURE_SPEC) -> np.ndarray:
    """Raw records (dict array) + spec → ma trận `[n, F]` float32.

    Một code path cho cả train lẫn chấm (train: records gom từ manifest; chấm: từ item).
    Các trường ngữ cảnh (`dur_rel`/`z_mean_rel`/`energy_z`/`dprev`/`dnext`/`pos_rel`) đã
    được ghi sẵn vào record bởi `apply_context`/`item_records`.
    """
    count = len(records.get("span", []))
    pairs = feature_pairs(records, spec=spec)
    if not pairs:
        return np.zeros((count, 0), dtype=np.float32)
    matrix = np.concatenate([values for _name, values in pairs], axis=1)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _resample_matrix(values: np.ndarray, points: int) -> np.ndarray:
    """Resample `[n, CONTOUR_MAX]` → `[n, points]` (mặc định giữ nguyên nếu bằng nhau)."""
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    if values.shape[1] == points:
        return values
    grid = np.linspace(0.0, 1.0, values.shape[1])
    target = np.linspace(0.0, 1.0, int(points))
    out = np.empty((values.shape[0], int(points)), dtype=np.float64)
    for index in range(values.shape[0]):
        out[index] = np.interp(target, grid, values[index])
    return out


def valid_mask(records: dict, *, spec: dict = FEATURE_SPEC) -> tuple[np.ndarray, dict]:
    """Mask âm tiết đủ điều kiện (`span ≥ min_span` ∧ `n_voiced ≥ min_voiced`) + lý do bỏ."""
    span = np.asarray(records["span"], dtype=np.float64)
    voiced = np.asarray(records["n_voiced"], dtype=np.float64)
    short = span < float(spec.get("min_span", 4))
    quiet = voiced < float(spec.get("min_voiced", 2))
    mask = ~(short | quiet)
    return mask, {"short": int(short.sum()), "few_voiced": int(quiet.sum()),
                  "kept": int(mask.sum()), "total": int(span.size)}


def features_from_wav(wav, config: dict, *, with_mel: bool = False) -> dict:
    """WAV (tensor/numpy 1 kênh) → kênh cấp frame `{f0, vuv, energy[, hnr, jitter, glottalization]}`.

    Dùng cho **audio tổng hợp**: cùng hàm trích với cache v3 (`extract_features`) nhưng nhận
    tensor thay vì đường dẫn — tiện chấm `Vocos(model)` mà không cần ghi WAV tạm. `config`
    là config YAML của repo (`mel.hop_length`, `f0.*`, `phonation.enabled`).
    """
    import torch

    from vifasttts.data.features import (
        canonical_mel_config,
        estimate_f0_pyworld,
        extract_mel_energy,
    )
    from vifasttts.data.phonation import extract_phonation_frames

    sample_rate = int(config.get("sample_rate", 24000))
    mel_config = canonical_mel_config(config)
    hop_length = int(mel_config["hop_length"])
    waveform = torch.as_tensor(np.asarray(_np(wav)), dtype=torch.float32).reshape(-1)
    mel, energy = extract_mel_energy(waveform, mel_config)
    frames = int(mel.size(0))
    f0_config = dict(config.get("f0", {}) or {})
    f0, vuv = estimate_f0_pyworld(waveform, sample_rate, hop_length,
                                  f_min=float(f0_config.get("f_min", 70.0)),
                                  f_max=float(f0_config.get("f_max", 400.0)),
                                  interpolate=bool(f0_config.get("interpolate", True)))

    def fit(values) -> np.ndarray:
        values = _np(values).reshape(-1)
        if values.size == frames:
            return values
        return np.interp(np.linspace(0.0, values.size - 1.0, frames),
                         np.arange(values.size), values) if values.size else np.zeros(frames)

    output = {"f0": fit(f0), "vuv": fit(vuv), "energy": fit(energy)}
    if config.get("phonation", {}).get("enabled", False):
        phonation = extract_phonation_frames(waveform, config,
                                             f0=torch.from_numpy(output["f0"]).float(),
                                             vuv=torch.from_numpy(output["vuv"]).float())
        for name, values in phonation.items():
            output[name] = fit(values)
    if with_mel:
        output["mel"] = _np(mel)
    return output


def item_from_synthesis(item: dict, wav, config: dict, *, tones: list[str] | None = None) -> dict:
    """Ghép item chấm được cho **audio tổng hợp**: kênh trích từ WAV + cửa sổ từ model.

    `item` cần `slot_durations` (layout model `[lead,(O,M,N,C,G)×N]`) và/hoặc `seg_durations`.
    Trả item mới (giữ `tones` nếu có) để đưa thẳng vào `ToneRuler.score_items([item])`.
    """
    frames = features_from_wav(wav, config)
    merged = dict(item)
    merged.update(frames)
    if tones is not None:
        merged["tones"] = list(tones)
    return merged


def featurize(f0_hz, vuv, window: tuple[int, int], *, hnr=None, jitter=None, glottalization=None,
              energy=None, syllable_window: tuple[int, int] | None = None,
              context: dict | None = None, spec: dict = FEATURE_SPEC,
              f0_stats: dict | None = None) -> np.ndarray:
    """API công khai: một âm tiết → vector đặc trưng `[F]` (cùng spec với lúc train).

    `window` = `(start, end)` frame của nucleus (hoặc cả âm tiết nếu `spec["window"]="syllable"`);
    các kênh `f0_hz`/`vuv`/`hnr`/`jitter`/`glottalization`/`energy` phải **cùng toạ độ frame**
    (row-relative). Thiếu phonation → cột phonation = 0 và cờ `phon_ok = 0`. Âm tiết không đủ
    frame hữu thanh vẫn trả vector (caller kiểm bằng `n_voiced`).

    `context`: thống kê cả câu/segment (`{"level", "dur_mean", "energy_mean", "energy_std"}`)
    — nên truyền khi chấm để khớp cách train (xem `item_context`).
    """
    stats = f0_stats or load_f0_stats()
    record = raw_record(f0_hz, vuv, window, hnr=hnr, jitter=jitter,
                        glottalization=glottalization, energy=energy,
                        syllable_window=syllable_window,
                        f0_log_mean=float(stats["f0_log_mean"]),
                        f0_log_std=float(stats["f0_log_std"]),
                        final_frac=float(spec.get("final_frac", 0.4)), points=CONTOUR_MAX)
    apply_context([record], context)
    matrix = records_matrix([record])
    return assemble_features(matrix, spec=spec)[0]


# ----------------------------------------------------------------------------------
# Dữ liệu train/val: quét manifest + cache, gom raw record cho mọi âm tiết
# ----------------------------------------------------------------------------------


def syllable_tones(row: dict) -> list[str]:
    """Nhãn thanh theo `syllables[i]["abstract"]["tone"]` ('' nếu không phải tiếng)."""
    tones = []
    for syllable in row.get("syllables", []):
        tone = str((syllable.get("abstract") or {}).get("tone", "")).upper()
        tones.append(tone if tone in TONE_TO_ID else "")
    return tones


def _cut_channel(values, start: int, frames: int | None):
    """Cắt kênh cấp frame theo `frame_range` của row segment (không cắt nếu row là clip đủ)."""
    if frames is None:
        return values
    return values[start:start + frames]


def extract_manifest(manifest: str | Path, *, cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
                     config_path: str | Path = DEFAULT_CONFIG, f0_stats: dict | None = None,
                     spec: dict = FEATURE_SPEC, limit: int | None = None
                     ) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """Manifest `_slot4` → `(records, tone_ids, valid, info)` (mọi âm tiết, kể cả âm tiết bỏ).

    `tone_ids[i] = -1` với âm tiết không phải tiếng (tone rỗng). `valid` = mask theo
    `valid_mask` (độ dài/hữu thanh). Cache `cache_dir=None` → trích trực tiếp từ audio.
    """
    import yaml

    from vifasttts.data.cache import FeatureCache
    from vifasttts.data.dataset import ManifestDataset

    stats = f0_stats or load_f0_stats()
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cache = FeatureCache(cache_dir, config) if cache_dir is not None else None
    rows = ManifestDataset(manifest).rows
    if limit is not None:
        rows = rows[: int(limit)]

    all_records: list[dict] = []
    tone_ids: list[int] = []
    row_info: list[tuple[str, int]] = []
    missing_cache = 0
    started = time.time()
    for position, row in enumerate(rows):
        extracted = cache.get(row) if cache is not None else None
        if extracted is None:
            missing_cache += 1
            from vifasttts.data.features import extract_features

            extracted = extract_features(row["audio_path"], config)
        frame_range = row.get("frame_range")
        start = int(frame_range[0]) if frame_range else 0
        frames = (int(frame_range[1]) - start) if frame_range else None
        item = {
            "f0": _cut_channel(extracted["f0"], start, frames),
            "vuv": _cut_channel(extracted["vuv"], start, frames),
            "energy": _cut_channel(extracted["energy"], start, frames),
            "seg_durations": row["seg_durations"], "durations": row["durations"],
            "gaps": row.get("gaps"), "lead": row.get("lead_frames") or 0,
        }
        for name in PHONATION_KEYS:
            if name in extracted:
                item[name] = _cut_channel(extracted[name], start, frames)
        records, _windows, _context = item_records(item, spec=spec, f0_stats=stats)
        tones = syllable_tones(row)
        count = len(records["span"])
        row_info.extend([(str(row.get("id", position)), index) for index in range(count)])
        all_records.append(records)
        tone_ids.extend(TONE_TO_ID.get(tone, -1) for tone in tones[:count])
        if count > len(tones):
            tone_ids.extend([-1] * (count - len(tones)))
        if (position + 1) % 2000 == 0:
            print(f"  ... {position + 1}/{len(rows)} row, {(time.time() - started):.1f}s",
                  flush=True)
    merged = _merge_records(all_records)
    tone_array = np.asarray(tone_ids, dtype=np.int64)
    mask, reasons = valid_mask(merged, spec=spec)
    info = {
        "manifest": str(manifest), "rows": len(rows), "syllables": int(tone_array.size),
        "tone_missing": int((tone_array < 0).sum()), "missing_cache": int(missing_cache),
        "drop": reasons, "seconds": round(time.time() - started, 1),
        "row_info": row_info,
    }
    return merged, tone_array, mask, info


def _merge_records(chunks: list[dict]) -> dict:
    """Gộp list `records_matrix` theo trục âm tiết."""
    if not chunks:
        return records_matrix([])
    merged: dict[str, np.ndarray] = {}
    for key in RECORD_SCALARS:
        merged[key] = np.concatenate([np.asarray(chunk.get(
            key, np.zeros(len(chunk.get("span", [])), dtype=np.float64)), dtype=np.float64)
            for chunk in chunks], axis=0)
    for key in ("contour", "contour_syl"):
        merged[key] = np.concatenate([np.asarray(chunk.get(
            key, np.zeros((len(chunk.get("span", [])), CONTOUR_MAX))), dtype=np.float64)
            for chunk in chunks], axis=0)
    return merged


def filter_records(records: dict, tone_ids: np.ndarray, mask: np.ndarray
                   ) -> tuple[dict, np.ndarray]:
    """Lọc `records` theo `mask` + tone hợp lệ → `(records_đã_lọc, y)`."""
    keep = mask & (np.asarray(tone_ids) >= 0)
    selected = {key: np.asarray(values)[keep] for key, values in records.items()}
    return selected, np.asarray(tone_ids)[keep].astype(np.int64)


def dataset_matrix(records: dict, tone_ids: np.ndarray, mask: np.ndarray, *,
                   spec: dict = FEATURE_SPEC) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) đã lọc theo mask + tone hợp lệ — dùng cho train/eval."""
    selected, labels = filter_records(records, tone_ids, mask)
    return assemble_features(selected, spec=spec), labels


# ----------------------------------------------------------------------------------
# Model: MLP nhỏ trên CPU + class weight (NGÃ/NẶNG ít)
# ----------------------------------------------------------------------------------


class ToneMLP:
    """MLP 2 lớp ẩn cho đặc trưng âm tiết; chuẩn hoá feature **trong** module (buffer).

    `state_dict` chứa cả `feat_mean`/`feat_std` → load là chấm được ngay, không cần scaler ngoài.
    """

    def __init__(self, in_dim: int, *, hidden: int = 128, layers: int = 2, dropout: float = 0.1,
                 n_classes: int = len(TONES)) -> None:
        import torch

        class _Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                blocks: list[torch.nn.Module] = []
                size = in_dim
                for _ in range(max(1, layers)):
                    blocks += [torch.nn.Linear(size, hidden), torch.nn.ReLU(),
                               torch.nn.Dropout(dropout)]
                    size = hidden
                blocks.append(torch.nn.Linear(size, n_classes))
                self.net = torch.nn.Sequential(*blocks)
                self.register_buffer("feat_mean", torch.zeros(in_dim))
                self.register_buffer("feat_std", torch.ones(in_dim))

            def forward(self, values):
                normalized = (values - self.feat_mean) / self.feat_std.clamp_min(1e-6)
                return self.net(normalized)

        self.module = _Net()
        self.in_dim = int(in_dim)

    def state_dict(self):
        """State dict (gồm buffer chuẩn hoá)."""
        return self.module.state_dict()

    def load_state_dict(self, state):
        """Nạp state dict."""
        self.module.load_state_dict(state)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        """Xác suất `[n,6]` (softmax, eval mode, CPU)."""
        import torch

        self.module.eval()
        with torch.no_grad():
            logits = self.module(torch.from_numpy(np.asarray(matrix, dtype=np.float32)))
            return torch.softmax(logits, dim=-1).numpy()

    def set_normalizer(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Gán mean/std chuẩn hoá feature (tính trên train)."""
        import torch

        self.module.feat_mean.copy_(torch.from_numpy(np.asarray(mean, dtype=np.float32)))
        self.module.feat_std.copy_(torch.from_numpy(np.asarray(std, dtype=np.float32)))


def class_weights(tone_ids: np.ndarray, *, power: float = 1.0) -> np.ndarray:
    """Trọng số lớp nghịch tần suất (chuẩn hoá mean = 1) — NGÃ/NẶNG rất ít."""
    counts = np.bincount(tone_ids, minlength=len(TONES)).astype(np.float64)
    weights = np.zeros(len(TONES), dtype=np.float64)
    present = counts > 0
    weights[present] = (counts[present].sum() / (present.sum() * counts[present])) ** float(power)
    weights[present] /= weights[present].mean()
    return weights


def train_mlp(X: np.ndarray, y: np.ndarray, *, X_val: np.ndarray | None = None,
              y_val: np.ndarray | None = None, hidden: int = 128, layers: int = 2,
              dropout: float = 0.1, epochs: int = 30, batch_size: int = 1024,
              lr: float = 1e-3, max_lr: float | None = None, weight_decay: float = 1e-4,
              seed: int = 0, weighted: bool = True, verbose: bool = True,
              restore_best: bool = True
              ) -> tuple[ToneMLP, dict]:
    """Train MLP trên CPU; trả `(model, history)` — history có loss/acc train + val mỗi epoch.

    LR: warm-up tuyến tính 2 epoch từ `lr/10` → `lr` (hoặc `max_lr`) rồi cosine về `lr/10`
    (train lâu không bị "hết LR sớm" như lr cố định). `restore_best=True` + có val → giữ
    trạng thái **epoch val tốt nhất** (early-stop mềm, báo cáo đúng con số đạt được).
    """
    import torch

    generator = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))
    model = ToneMLP(X.shape[1], hidden=hidden, layers=layers, dropout=dropout)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    model.set_normalizer(mean, std)
    peak = float(max_lr or lr)
    floor = peak / 10.0
    optimizer = torch.optim.Adam(model.module.parameters(), lr=floor,
                                 weight_decay=float(weight_decay))
    weights = torch.from_numpy(class_weights(y).astype(np.float32)) if weighted else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.int64))
    history: dict = {"loss": [], "acc": [], "val_acc": [], "val_loss": [], "lr": [],
                     "best_epoch": 0}
    count = X_t.shape[0]
    warmup_epochs = max(1, min(2, int(epochs) // 5))
    best_state: dict | None = None
    best_accuracy = -1.0
    for epoch in range(int(epochs)):
        if epoch < warmup_epochs:
            current_lr = floor + (peak - floor) * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, int(epochs) - warmup_epochs - 1)
            current_lr = floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for group in optimizer.param_groups:
            group["lr"] = float(current_lr)
        model.module.train()
        order = generator.permutation(count)
        total_loss, total = 0.0, 0
        for start in range(0, count, int(batch_size)):
            index = torch.from_numpy(order[start:start + int(batch_size)].copy())
            logits = model.module(X_t[index])
            loss = criterion(logits, y_t[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * index.numel()
            total += index.numel()
        history["loss"].append(total_loss / max(total, 1))
        history["lr"].append(float(current_lr))
        history["acc"].append(float((model.predict_proba(X).argmax(axis=1) == y).mean()))
        if X_val is not None and y_val is not None:
            probabilities = model.predict_proba(X_val)
            accuracy = float((probabilities.argmax(axis=1) == y_val).mean())
            history["val_acc"].append(accuracy)
            history["val_loss"].append(float(
                torch.nn.functional.cross_entropy(
                    torch.log(torch.from_numpy(probabilities).clamp_min(1e-9)),
                    torch.from_numpy(np.asarray(y_val, dtype=np.int64))).item()))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                history["best_epoch"] = epoch + 1
                best_state = {key: value.detach().clone()
                              for key, value in model.state_dict().items()}
        if verbose:
            message = (f"  epoch {epoch + 1:3d}/{epochs} loss {history['loss'][-1]:.4f} "
                       f"acc {history['acc'][-1]:.4f}")
            if history["val_acc"]:
                message += f" val_acc {history['val_acc'][-1]:.4f}"
            print(message, flush=True)
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        history["restored_best"] = True
    return model, history


def train_ensemble(X: np.ndarray, y: np.ndarray, *, X_val: np.ndarray | None = None,
                   y_val: np.ndarray | None = None, seeds: tuple[int, ...] = (0, 1, 2),
                   verbose: bool = True, **kwargs) -> list[tuple[ToneMLP, dict]]:
    """Train `len(seeds)` model cùng cấu hình, khác seed → ensemble soft-vote.

    Ensemble 3 seed đo thật: +0,5–0,7 điểm accuracy so single (xem `tone_ruler.json`).
    """
    trained = []
    for seed in seeds:
        model, history = train_mlp(X, y, X_val=X_val, y_val=y_val, seed=int(seed),
                                   verbose=verbose, **kwargs)
        trained.append((model, history))
        if verbose:
            best = history["best_epoch"]
            print(f"  seed {seed}: val_acc {history['val_acc'][best - 1]:.4f} (epoch {best})"
                  if history["val_acc"] else f"  seed {seed}: xong", flush=True)
    return trained


def ensemble_probabilities(models: list[ToneMLP], X: np.ndarray) -> np.ndarray:
    """Trung bình xác suất các model (soft vote)."""
    return np.mean([model.predict_proba(X) for model in models], axis=0)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Ma trận nhầm `[6,6]` (hàng = thật, cột = đoán)."""
    matrix = np.zeros((len(TONES), len(TONES)), dtype=np.int64)
    for true, pred in zip(np.asarray(y_true), np.asarray(y_pred), strict=True):
        if 0 <= int(true) < len(TONES) and 0 <= int(pred) < len(TONES):
            matrix[int(true), int(pred)] += 1
    return matrix


def report(y_true: np.ndarray, y_pred: np.ndarray, *, label: str = "") -> dict:
    """Accuracy tổng + per-thanh + ma trận nhầm (dict JSON-able)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    total = int(y_true.size)
    hits = int((y_true == y_pred).sum()) if total else 0
    per_tone = {}
    for index, tone in enumerate(TONES):
        rows = int(matrix[index].sum())
        per_tone[tone] = {
            "n": rows,
            "correct": int(matrix[index, index]),
            "accuracy": round(float(matrix[index, index] / rows), 4) if rows else None,
        }
    return {"label": label, "total": total, "correct": hits,
            "accuracy": round(hits / total, 4) if total else None,
            "chance": round(CHANCE, 4), "per_tone": per_tone,
            "confusion": matrix.tolist()}


def print_report(summary: dict) -> None:
    """In bảng ngắn: accuracy + per-thanh + ma trận nhầm."""
    tag = f"[{summary['label']}] " if summary.get("label") else ""
    print(f"{tag}{summary['correct']}/{summary['total']} = {summary['accuracy']:.4f} "
          f"(chance {summary['chance']:.4f})")
    for tone in TONES:
        entry = summary["per_tone"][tone]
        accuracy = f"{entry['accuracy']:.4f}" if entry["accuracy"] is not None else "n/a"
        print(f"  {tone:6s} {entry['correct']:5d}/{entry['n']:5d} = {accuracy}")
    print("  Ma trận nhầm (hàng = thật, cột = đoán):")
    print("         " + " ".join(f"{tone[:5]:>6s}" for tone in TONES))
    for index, tone in enumerate(TONES):
        row = summary["confusion"][index]
        print(f"  {tone:6s} " + " ".join(f"{value:>6d}" for value in row))


def save_model(models: list[ToneMLP], spec: dict, *, path: str | Path, metrics: dict,
               f0_stats: dict) -> None:
    """Lưu `tone_ruler.pt`: `state_dicts` (ensemble) + spec + tên đặc trưng + metrics + f0_stats."""
    import torch

    payload = {
        "version": spec.get("version", "tone-ruler-v1"),
        "state_dicts": [model.state_dict() for model in models],
        "in_dim": int(models[0].in_dim),
        "ensemble": len(models),
        "spec": {key: (list(value) if isinstance(value, tuple) else value)
                 for key, value in spec.items()},
        "feature_names": features_from_spec(spec),
        "f0_stats": {"f0_log_mean": float(f0_stats["f0_log_mean"]),
                     "f0_log_std": float(f0_stats["f0_log_std"])},
        "metrics": metrics,
        "tones": TONES,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _model_from_state(state: dict, in_dim: int) -> ToneMLP:
    """Dựng `ToneMLP` đúng kiến trúc từ state_dict đã lưu (hidden/layers suy từ shape)."""
    weights = [key for key in state if key.startswith("net.") and key.endswith(".weight")]
    hidden = int(state["net.0.weight"].shape[0])
    layers = max(1, len(weights) - 1)
    model = ToneMLP(int(in_dim), hidden=hidden, layers=layers, dropout=0.0)
    model.load_state_dict(state)
    return model


class ToneRuler:
    """Bọc model đã train: chấm item (audio thật hoặc tổng hợp) → thanh dự đoán + xác suất."""

    def __init__(self, models: list[ToneMLP], spec: dict, f0_stats: dict,
                 metrics: dict | None = None):
        """`spec`/`f0_stats` phải khớp lúc train (đọc từ `tone_ruler.pt`)."""
        self.models = list(models)
        self.spec = dict(spec)
        self.spec["groups"] = list(spec.get("groups", FEATURE_SPEC["groups"]))
        self.f0_stats = dict(f0_stats)
        self.metrics = dict(metrics or {})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL) -> ToneRuler:
        """Nạp `tone_ruler.pt` (CPU); ensemble nhiều model → trung bình xác suất."""
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        states = payload.get("state_dicts") or [payload["state_dict"]]
        models = [_model_from_state(state, int(payload["in_dim"])) for state in states]
        spec = dict(payload["spec"])
        spec["groups"] = tuple(spec.get("groups", FEATURE_SPEC["groups"]))
        return cls(models, spec, payload["f0_stats"], payload.get("metrics"))

    def score_records(self, records: dict) -> np.ndarray:
        """Chấm raw records (dict array, đã ghi trường ngữ cảnh) → xác suất `[n,6]`."""
        matrix = assemble_features(records, spec=self.spec)
        if matrix.shape[1] != self.models[0].in_dim:
            raise ValueError(f"spec lệch model: {matrix.shape[1]} != {self.models[0].in_dim}")
        return ensemble_probabilities(self.models, matrix)

    def score_items(self, items: list[dict]) -> list[list[dict]]:
        """Chấm danh sách item (mỗi item = câu/segment) → list kết quả từng âm tiết.

        Mỗi kết quả: `{index, start, end, span, n_voiced, valid, tone, tone_id, probs, reason}`;
        `valid=False` (âm tiết quá ngắn/ít hữu thanh) → `tone=None`.
        """
        output = []
        warned = False
        for item in items:
            records, windows, _context = item_records(item, spec=self.spec,
                                                      f0_stats=self.f0_stats)
            mask, _reasons = valid_mask(records, spec=self.spec)
            probabilities = self.score_records(records)
            if ("phonation" in list(self.spec.get("groups", ())) and not warned
                    and float(np.mean(records["phon_ok"])) < 0.5):
                print("CẢNH BÁO: item thiếu kênh phonation (hnr/jitter/glottalization) — "
                      "dùng `tone_ruler.features_from_wav` hoặc đọc cache v3 để có đủ; "
                      "thiếu kênh làm accuracy giảm (đo thật: ~2 điểm).", file=sys.stderr)
                warned = True
            tones = item.get("tones")
            rows = []
            for index, (start, end) in enumerate(windows):
                probability = probabilities[index]
                predicted = int(probability.argmax())
                reason = ""
                if not mask[index]:
                    reason = ("ngắn" if records["span"][index] < float(self.spec.get("min_span", 4))
                              else "ít hữu thanh")
                rows.append({
                    "index": int(index), "start": int(start), "end": int(end),
                    "span": float(records["span"][index]),
                    "n_voiced": float(records["n_voiced"][index]),
                    "valid": bool(mask[index]),
                    "tone": TONES[predicted] if mask[index] else None,
                    "tone_id": predicted if mask[index] else None,
                    "probs": {tone: round(float(probability[i]), 4)
                              for i, tone in enumerate(TONES)},
                    "tone_true": (str(tones[index]).upper()
                                  if tones is not None and index < len(tones) else None),
                    "reason": reason,
                })
            output.append(rows)
        return output

    def classify_syllables(self, items: list[dict]) -> list[list[dict]]:
        """Alias `score_items` (tên API theo yêu cầu "classify_syllables")."""
        return self.score_items(items)


def classify_syllables(items: list[dict], *, model_path: str | Path = DEFAULT_MODEL
                       ) -> list[list[dict]]:
    """Tiện ích: nạp model mặc định rồi chấm `items` (xem `ToneRuler.score_items`)."""
    return ToneRuler.load(model_path).score_items(items)


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------


def dct4_shape(contour: np.ndarray, keep: int = 4) -> np.ndarray:
    """`s_k = c_k/√n` trên contour (numpy, khớp `data/features.dct4_frames` + §6).

    Dùng để tái lập **thước cũ** (`eval_tone_pairs`: DCT-4 → gần prior nhất) trên cùng cửa sổ
    âm tiết — cột mốc so sánh trực tiếp (0,46 trên audio thật theo `artifacts/tone_path/`).
    """
    values = np.asarray(contour, dtype=np.float64).reshape(len(contour), -1)
    count = values.shape[1]
    coefficients = np.zeros((values.shape[0], keep), dtype=np.float64)
    if count == 0:
        return coefficients
    index = np.arange(keep)[:, None]
    frames = np.arange(count)[None, :]
    basis = np.cos(np.pi * index * (frames + 0.5) / count)
    basis[0] /= math.sqrt(2.0)
    return (basis @ values.T).T * math.sqrt(2.0 / count) / math.sqrt(count)


def prior_dct4_predictions(records: dict, f0_stats: dict) -> np.ndarray:
    """Thước **cũ**: contour nucleus → `s_k`/√n → z theo `shape_mean/std` → gần `tone_prior` nhất."""
    prior = f0_stats.get("shape_tone_prior") or {}
    if not prior:
        raise ValueError("f0_stats thiếu shape_tone_prior (cần cho baseline thước cũ)")
    shape_mean = np.asarray(f0_stats["shape_mean"], dtype=np.float64)
    shape_std = np.maximum(np.asarray(f0_stats["shape_std"], dtype=np.float64), 1e-6)
    table = np.zeros((len(TONES), 4), dtype=np.float64)
    for tone_index, tone in enumerate(TONES):
        values = (prior.get(tone) or {}).get("mean") or []
        if len(values) == 4:
            table[tone_index] = (np.asarray(values, dtype=np.float64) - shape_mean) / shape_std
    coefficients = dct4_shape(records["contour"], keep=4)
    observed = (coefficients - shape_mean) / shape_std
    distances = ((table[:, None, :] - observed[None, :, :]) ** 2).sum(axis=-1)
    return distances.argmin(axis=0)


def build_eval_baselines(X: np.ndarray, y: np.ndarray, mask_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray, spec: dict, *,
                         epochs: int = 12, seed: int = 0,
                         val_records: dict | None = None, f0_stats: dict | None = None) -> dict:
    """Baseline: "chỉ độ dài" · "chỉ F0 contour" (cùng MLP) · thước cũ `prior_dct4` (không train)."""
    names = features_from_spec(spec)
    index = {name: position for position, name in enumerate(names)}
    duration_columns = [index[name] for name in
                        ("log_span", "log_voiced", "voiced_ratio", "dur_rel", "log_span_syl",
                         "voiced_ratio_syl")
                        if name in index]
    contour_columns = [position for name, position in index.items()
                       if name.startswith(("contour", "shape", "slope", "dct", "rate",
                                           "argmin", "argmax", "range"))]
    results = {}
    for label, columns in (("duration_only", duration_columns), ("f0_contour_only", contour_columns)):
        if not columns:
            continue
        model, _history = train_mlp(X[mask_train][:, columns], y[mask_train], epochs=epochs,
                                    seed=seed, verbose=False)
        probabilities = model.predict_proba(X_val[:, columns])
        summary = report(y_val, probabilities.argmax(axis=1), label=label)
        results[label] = {"columns": len(columns), "accuracy": summary["accuracy"],
                          "per_tone": summary["per_tone"]}
        print(f"  baseline {label:16s} ({len(columns)} cột): acc {summary['accuracy']:.4f}")
    if val_records is not None and f0_stats is not None:
        # Thước CŨ (eval_tone_pairs): DCT-4 → gần `tone_prior` nhất, trên đúng tập âm tiết này.
        old = prior_dct4_predictions(val_records, f0_stats)
        summary = report(y_val, old, label="prior_dct4_cu")
        results["prior_dct4_cu"] = {"columns": 4, "accuracy": summary["accuracy"],
                                    "per_tone": summary["per_tone"]}
        print(f"  baseline {'prior_dct4_cu':16s} (4 hệ số, thước cũ): "
              f"acc {summary['accuracy']:.4f}")
    return results


def command_train(args: argparse.Namespace) -> None:
    """`--train`: trích đặc trưng audio thật (train + val) → train MLP → báo cáo + lưu model."""
    spec = dict(FEATURE_SPEC)
    if args.contour_points:
        spec["contour_points"] = int(args.contour_points)
    if args.window:
        spec["window"] = str(args.window)
    if args.contour_source:
        spec["contour_source"] = str(args.contour_source)
    if args.groups:
        spec["groups"] = tuple(args.groups)
    stats = load_f0_stats(args.f0_stats)

    print(f"[1/4] Trích đặc trưng train: {args.train_manifest}")
    records, tone_ids, mask, info = extract_manifest(
        args.train_manifest, cache_dir=args.cache_dir, config_path=args.config,
        f0_stats=stats, spec=spec, limit=args.limit)
    print(f"  âm tiết {info['syllables']} · bỏ {info['drop']['short']} ngắn + "
          f"{info['drop']['few_voiced']} ít hữu thanh · tone thiếu {info['tone_missing']} "
          f"· {info['seconds']}s")
    X_train, y_train = dataset_matrix(records, tone_ids, mask, spec=spec)
    n_voiced = records["n_voiced"][mask & (tone_ids >= 0)]
    print(f"  giữ {X_train.shape[0]} âm tiết · {X_train.shape[1]} đặc trưng · "
          f"frame hữu thanh/cửa sổ: mean {n_voiced.mean():.2f} · "
          f"≥2 frame: {float((n_voiced >= 2).mean()):.3f}")
    print("  phân bố thanh (train): " + json.dumps(
        {tone: int((y_train == index).sum()) for index, tone in enumerate(TONES)}))

    print(f"[2/4] Trích đặc trưng val: {args.val_manifest}")
    val_records, val_tones, val_mask, val_info = extract_manifest(
        args.val_manifest, cache_dir=args.cache_dir, config_path=args.config,
        f0_stats=stats, spec=spec, limit=args.val_limit)
    X_val, y_val = dataset_matrix(val_records, val_tones, val_mask, spec=spec)
    n_voiced_val = val_records["n_voiced"][val_mask & (val_tones >= 0)]
    print(f"  giữ {X_val.shape[0]} âm tiết · frame hữu thanh: mean {n_voiced_val.mean():.2f} "
          f"· ≥2 frame {float((n_voiced_val >= 2).mean()):.3f}")

    if args.records_cache:
        target = Path(args.records_cache)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"{target}_train.npz", **records, tone_ids=tone_ids, mask=mask)
        np.savez_compressed(f"{target}_val.npz", **val_records, tone_ids=val_tones,
                            mask=val_mask)
        print(f"  đã lưu raw records: {target}_train.npz · {target}_val.npz")

    print(f"[3/4] Train MLP ({args.epochs} epoch, hidden {args.hidden}×{args.layers}, "
          f"{len(args.seeds)} seed, CPU)")
    trained = train_ensemble(X_train, y_train, X_val=X_val, y_val=y_val,
                             seeds=tuple(args.seeds), hidden=args.hidden, layers=args.layers,
                             dropout=args.dropout, epochs=args.epochs,
                             batch_size=args.batch_size, lr=args.lr,
                             weighted=not args.no_weight, verbose=True)
    models = [model for model, _history in trained]
    histories = {str(seed): history for seed, (_model, history) in zip(args.seeds, trained)}
    probabilities = ensemble_probabilities(models, X_val)
    summary = report(y_val, probabilities.argmax(axis=1), label="val (audio thật)")
    if len(models) > 1:
        single = [report(y_val, model.predict_proba(X_val).argmax(axis=1),
                         label="val single")["accuracy"] for model in models]
        print(f"  single seed: {single} → ensemble acc {summary['accuracy']:.4f}")
    print_report(summary)

    print("[4/4] Baseline (cùng MLP, ít epoch)")
    train_mask = np.ones(X_train.shape[0], dtype=bool)
    val_kept, _val_labels = filter_records(val_records, val_tones, val_mask)
    baselines = build_eval_baselines(X_train, y_train, train_mask, X_val, y_val, spec,
                                     epochs=max(3, args.epochs // 3), seed=args.seed,
                                     val_records=val_kept, f0_stats=stats)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model or (out_dir / "tone_ruler.pt"))
    metrics = {
        "val": summary, "baselines": baselines,
        "train": {"syllables": int(X_train.shape[0]), "features": int(X_train.shape[1]),
                  "per_tone": {tone: int((y_train == index).sum())
                               for index, tone in enumerate(TONES)}},
        "val_info": {key: value for key, value in val_info.items() if key != "row_info"},
        "train_info": {key: value for key, value in info.items() if key != "row_info"},
        "history": histories, "ensemble": len(models), "seeds": list(args.seeds),
        "architecture": {"hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
                         "epochs": args.epochs},
    }
    save_model(models, spec, path=model_path, metrics=metrics, f0_stats=stats)
    description = {
        "tool": "tools/tone_ruler.py", "version": spec["version"], "spec": spec,
        "feature_names": features_from_spec(spec), "tones": TONES,
        "f0_stats": {"path": str(args.f0_stats), "f0_log_mean": float(stats["f0_log_mean"]),
                     "f0_log_std": float(stats["f0_log_std"])},
        "model": str(model_path), "metrics": metrics,
    }
    json_path = out_dir / "tone_ruler.json"
    json_path.write_text(json.dumps(description, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã lưu {model_path} và {json_path}")


def command_score_real(args: argparse.Namespace) -> None:
    """`--score-real`: chấm lại audio thật theo manifest (mặc định val) → per-thanh."""
    ruler = ToneRuler.load(args.model or DEFAULT_MODEL)
    records, tone_ids, mask, info = extract_manifest(
        args.manifest, cache_dir=args.cache_dir, config_path=args.config,
        f0_stats=ruler.f0_stats, spec=ruler.spec, limit=args.limit)
    X, y = dataset_matrix(records, tone_ids, mask, spec=ruler.spec)
    probabilities = ensemble_probabilities(ruler.models, X)
    summary = report(y, probabilities.argmax(axis=1), label=f"real · {Path(args.manifest).name}")
    print(f"Âm tiết: tổng {info['syllables']} · bỏ {info['drop']['short']} ngắn + "
          f"{info['drop']['few_voiced']} ít hữu thanh · không phải tiếng {info['tone_missing']}")
    print_report(summary)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"info": {k: v for k, v in info.items() if k != "row_info"},
                                   "report": summary}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"Đã ghi {out}")


def main() -> None:
    """Đọc tham số CLI: `--train` (mặc định) hoặc `--score-real`."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Trích đặc trưng + train + báo cáo")
    mode.add_argument("--score-real", action="store_true", help="Chấm audio thật bằng model đã lưu")
    parser.add_argument("--train-manifest", default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--manifest", default=DEFAULT_VAL_MANIFEST,
                        help="Manifest cho --score-real")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--f0-stats", default=DEFAULT_F0_STATS)
    parser.add_argument("--model", default=None, help=f"Mặc định {DEFAULT_MODEL}")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--out", default=None, help="JSON kết quả cho --score-real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seed cho ensemble (nhiều model → trung bình xác suất)")
    parser.add_argument("--no-weight", action="store_true", help="Tắt class weight")
    parser.add_argument("--contour-points", type=int, default=None)
    parser.add_argument("--window", choices=["nucleus", "syllable"], default=None)
    parser.add_argument("--contour-source", choices=["nucleus", "syllable", "both"], default=None)
    parser.add_argument("--groups", nargs="+", default=None)
    parser.add_argument("--records-cache", default=None,
                        help="Tiền tố lưu raw records (.npz) — để thử biến thể nhanh")
    args = parser.parse_args()
    if args.score_real:
        command_score_real(args)
    else:
        command_train(args)


if __name__ == "__main__":
    main()

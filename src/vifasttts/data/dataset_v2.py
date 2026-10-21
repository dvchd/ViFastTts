"""Dataset v2 cấp slot (docs/ACOUSTIC_V2.md §3, §23; T2.2).

`SlotDataset` đọc manifest `_slot4` + cache feature, cắt theo `frame_range` rồi dựng
mọi target cấp segment qua `data/v2_targets.segment_targets`. Không fallback khi thiếu
`seg_durations`/`frame_range` (F3: raise `ValueError`).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from vifasttts.data.cache import FeatureCache
from vifasttts.data.dataset import ManifestDataset, encode_syllables
from vifasttts.data.features import extract_features
from vifasttts.data.v2_targets import segment_targets
from vifasttts.frontend.vocab import ComponentVocabs

#: Key cấp frame cần pad theo trục thời gian trong `collate_slot_batch`.
FRAME_KEYS = ("mel", "f0", "vuv", "energy")


class SlotDataset(Dataset):
    """Dataset train acoustic v2: một mẫu = một segment với target cấp slot.

    Mỗi `__getitem__` trả dict tensor (giữ tensor, không numpy):
    `id` (str), `features` [N,7] long, `durations`/`gaps` [N] long, `lead` scalar long,
    `mel` [T,20], `energy`/`f0`/`vuv` [T], `seg_durations` [N,4] long (thứ tự O,M,N,C),
    cộng 9 key target từ `segment_targets` (`seg_ratio`, `seg_ratio_mask`, `f0_shape`,
    `f0_weight`, `seg_energy`, `seg_vuv`, `pause_gate`, `pause_log_gap`, `pause_mask`).

    Cache chỉ đọc (`FeatureCache.get`); miss → `extract_features` trực tiếp, KHÔNG ghi
    cache (ghi cache là việc của `cache-features`, tránh race khi nhiều worker).
    """

    def __init__(
        self,
        manifest: str | Path,
        config: dict[str, Any],
        f0_stats: dict[str, Any],
        cache_dir: str | Path | None = None,
        *,
        min_syllables: int = 1,
    ) -> None:
        """Nạp manifest + config feature + stats F0; `cache_dir` bật đọc cache.

        `min_syllables=2` lọc row 1 âm tiết — MPS abort ở conv backward khi N=1
        (xem `docs/ACOUSTIC_V2.md` §25); val/eval để mặc định vì chỉ forward.
        """
        self.rows = [row for row in ManifestDataset(manifest).rows
                     if len(row.get("durations") or []) >= min_syllables]
        self.config = config
        self.f0_stats = f0_stats
        self.vocabs = ComponentVocabs.default()
        self.cache = FeatureCache(cache_dir, config) if cache_dir is not None else None
        # v3.5 (A0 v17 A3): cắt ngẫu nhiên 1–5 âm tiết đầu/cuối, độc lập p=0,3 mỗi phía.
        # Mục đích: model không học vẹt "âm đầu câu/âm cuối câu" của nhãn segment.
        training = config.get("training", {}) if isinstance(config, dict) else {}
        self.augment_edges = bool(training.get("augment_edges", False))
        self.augment_prob = float(training.get("augment_edge_prob", 0.3))
        self.augment_seed = int(training.get("augment_seed", 0))
        self.min_keep = int(training.get("augment_min_syllables", 4))
        # Kế hoạch cắt TÍNH TRƯỚC theo index (tất định) — sampler cần biết N sau cắt để gộp batch.
        self._plans: dict[int, tuple[int, int]] = {}
        if self.augment_edges:
            for index, row in enumerate(self.rows):
                self._plans[index] = self._sample_plan(row, index)

    def __len__(self) -> int:
        """Số row trong manifest."""
        return len(self.rows)

    def _sample_plan(self, row: dict[str, Any], index: int) -> tuple[int, int]:
        """Chọn `(prefix, suffix)` cắt biên — tất định theo index, tính TRƯỚC khi lấy mẫu."""
        count = len(row.get("durations") or [])
        if "frame_range" not in row or count < self.min_keep + 2:
            return 0, 0
        rng = random.Random(self.augment_seed * 1_000_003 + index)
        budget = count - self.min_keep
        prefix = rng.randint(1, min(5, budget)) if rng.random() < self.augment_prob else 0
        budget -= prefix
        suffix = (rng.randint(1, min(5, budget))
                  if budget > 0 and rng.random() < self.augment_prob else 0)
        return prefix, suffix

    def effective_lengths(self) -> list[int]:
        """N thật sau augmentation của từng row — dùng cho `SameNBatchSampler`."""
        return [len(row.get("durations") or []) - sum(self._plans.get(index, (0, 0)))
                for index, row in enumerate(self.rows)]

    def _augment_edges(self, row: dict[str, Any], start: int, frames: int | None,
                       index: int) -> tuple[dict[str, Any], int, int | None]:
        """Cắt 1–5 âm tiết đầu/cuối (xác suất riêng mỗi phía) — giữ feature biên GỐC.

        Chỉ áp dụng cho row segment (`frame_range`, toạ độ tương đối gốc 0); trả row mới +
        `(start, frames)` mới. Kế hoạch đã tính sẵn trong `self._plans`.
        """
        if frames is None or "frame_range" not in row:
            return row, start, frames
        count = len(row["durations"])
        prefix, suffix = self._plans.get(index, (0, 0))
        if prefix == 0 and suffix == 0:
            return row, start, frames
        durations = [int(v) for v in row["durations"]]
        gaps = [int(v) for v in (row.get("gaps") or [])]
        if len(gaps) == count - 1:
            gaps = gaps + [0]
        lead = int(row.get("lead_frames") or 0)
        offsets, cursor = [], lead
        for position in range(count):
            offsets.append(cursor)
            cursor += durations[position] + gaps[position]
        first, last = prefix, count - 1 - suffix
        new_start = offsets[first]
        new_end = offsets[last] + durations[last] + gaps[last]
        new_row = dict(row)
        new_row["durations"] = durations[first:last + 1]
        new_row["gaps"] = gaps[first:last + 1]
        new_row["seg_durations"] = [list(v) for v in row["seg_durations"][first:last + 1]]
        new_row["syllables"] = list(row["syllables"])[first:last + 1]
        new_row["lead_frames"] = 0
        origin = int(row["frame_range"][0])
        new_row["frame_range"] = [origin + new_start, origin + new_end]
        return new_row, start + new_start, new_end - new_start

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Mẫu thứ index: feature cấp âm tiết + frame đã cắt + target cấp segment.

        Row segment (`frame_range`) → cắt `[start:start+frames]`; **raise** nếu cache ngắn hơn
        (không fallback im lặng). Row clip đầy đủ (không `frame_range`, vd holdout) → dùng trọn tensor.
        """
        row = self.rows[index]
        if "seg_durations" not in row or row["seg_durations"] is None:
            raise ValueError(f"F3: row {row.get('id', index)} thiếu seg_durations")
        if self.cache is not None:
            extracted = self.cache.get(row)
            if extracted is None:
                extracted = extract_features(row["audio_path"], self.config)
        else:
            extracted = extract_features(row["audio_path"], self.config)

        frame_range = row.get("frame_range")
        start = int(frame_range[0]) if frame_range else 0
        frames = (int(frame_range[1]) - start) if frame_range else None
        if self.augment_edges:
            row, start, frames = self._augment_edges(row, start, frames, index)

        def _cut(name: str) -> torch.Tensor:
            values = extracted[name]
            if frames is None:
                return values
            if values.size(0) < start + frames:
                raise ValueError(
                    f"cache {name} ngắn hơn frame_range cho row {row.get('id', index)}: "
                    f"{values.size(0)} < {start + frames}"
                )
            return values[start:start + frames]

        mel = _cut("mel")
        energy = _cut("energy")
        f0 = _cut("f0")
        vuv = _cut("vuv")
        slices = {"f0": f0, "vuv": vuv, "energy": energy}
        for name in ("hnr", "glottalization", "jitter"):  # cache v3 (R6) — bỏ qua nếu thiếu
            if name in extracted:
                slices[name] = _cut(name)

        features = torch.tensor(encode_syllables(row["syllables"], self.vocabs), dtype=torch.long)
        count = features.size(0)
        durations = torch.tensor(row["durations"], dtype=torch.long)
        gaps_list = list(row.get("gaps") or [])
        if len(gaps_list) == count - 1:
            gaps_list = gaps_list + [0]
        gaps_list = (gaps_list + [0] * count)[:count]
        gaps = torch.tensor(gaps_list, dtype=torch.long)
        lead = torch.tensor(int(row.get("lead_frames") or 0), dtype=torch.long)
        seg_durations = torch.tensor(row["seg_durations"], dtype=torch.long)

        targets = segment_targets(row, slices, self.f0_stats)
        return {
            "id": row.get("id", str(index)),
            "features": features,
            "durations": durations,
            "gaps": gaps,
            "lead": lead,
            "mel": mel,
            "energy": energy,
            "f0": f0,
            "vuv": vuv,
            "seg_durations": seg_durations,
            **targets,
        }


def collate_slot_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Gộp batch v2: pad trục frame, stack thẳng trục âm tiết.

    Quyết định: **yêu cầu mọi mẫu có cùng N** (số âm tiết); khác N → raise `ValueError`.
    Lý do: model v2 mở rộng slot theo từng âm tiết đồng bộ, pad N sẽ làm lệch mask
    `seg_ratio`/trọng số F0; train v2 dùng batch 1 hoặc gom mẫu cùng N ở trainer.
    Trục frame (mel/f0/vuv/energy) pad 0 tới Tmax → [B,Tmax,...] + `frame_lengths` [B].
    Mọi tensor cấp âm tiết stack thẳng (đã cùng N); `lead` → [B] long; `id` giữ list[str].
    """
    if not items:
        raise ValueError("collate_slot_batch cần ít nhất một mẫu")
    counts = {item["features"].size(0) for item in items}
    if len(counts) != 1:
        raise ValueError(f"batch v2 yêu cầu cùng N, nhận được {sorted(counts)}")
    batch = len(items)
    frame_max = max(item["mel"].size(0) for item in items)
    mel_dim = items[0]["mel"].size(1)
    output: dict[str, Any] = {
        "id": [item["id"] for item in items],
        "frame_lengths": torch.tensor([item["mel"].size(0) for item in items], dtype=torch.long),
    }
    frame_keys = ["mel", "f0", "vuv", "energy"]
    if all("mel_creak_c" in item for item in items):  # E4c: [T] theo frame (dataset mới)
        frame_keys.append("mel_creak_c")
    for key in frame_keys:
        first = items[0][key]
        padded = torch.zeros((batch, frame_max, *first.shape[1:]), dtype=first.dtype)
        for i, item in enumerate(items):
            length = item[key].size(0)
            padded[i, :length] = item[key]
        output[key] = padded
    assert output["mel"].shape[1:] == (frame_max, mel_dim)
    for key in items[0]:
        if key in ("id", "mel", "f0", "vuv", "energy", "mel_creak_c", "frame_lengths"):
            continue
        output[key] = torch.stack([item[key] for item in items])
    return output


def rare_group_weights(rows: list[dict[str, Any]], *, nga: float = 2.0, nang: float = 1.5,
                       checked_coda: float = 1.5) -> list[float]:
    """Trọng số lấy mẫu theo nhóm HIẾM (V11S3 — phân tích bao phủ train fixtail):

    NGÃ chỉ **4,5%** âm tiết, NẶNG 9,7%, coda tắc 1–5%; ô `M:NGA` chỉ **2 lần** toàn tập
    (222.737 âm tiết) — đúng nhóm mà model yếu nhất (rè ngã + SẮC-tắc §42.1). Nhân trọng số
    để nhóm hiếm được học nhiều hơn mà **không thêm dữ liệu giả**. Hệ số 1,0 = tắt.
    """
    weights: list[float] = []
    for row in rows:
        value = 1.0
        for syllable in row.get("syllables") or []:
            abstract = syllable["abstract"]
            if abstract["tone"] == "NGA":
                value *= max(nga, 1e-6)
            elif abstract["tone"] == "NANG":
                value *= max(nang, 1e-6)
            if abstract["coda"] in ("P", "T", "C", "CH"):
                value *= max(checked_coda, 1e-6)
        weights.append(value)
    return weights


class SameNBatchSampler:
    """Sampler gộp index **theo cùng N** (số âm tiết) — để batch > 1 vẫn hợp lệ.

    `collate_slot_batch` yêu cầu cùng N trong batch (không pad trục âm tiết); sampler này
    chia index thành nhóm cùng N rồi cắt thành batch `batch_size`, xáo trộn trong nhóm và
    xáo trộn thứ tự batch. Nhờ vậy dùng được batch > 1 mà không đổi dữ liệu/model.
    """

    def __init__(self, rows: list[dict[str, Any]], batch_size: int, *,
                 shuffle: bool = True, seed: int = 42, drop_last: bool = False,
                 weights: list[float] | None = None,
                 lengths: list[int] | None = None) -> None:
        """`rows`: manifest rows (cần `durations` để biết N).

        `weights` (V11S3): trọng số lấy mẫu theo ROW — nhóm hiếm (NGÃ/NẶNG + coda tắc)
        được lấy **có hoàn lại** nhiều hơn, cùng cỡ nhóm ⇒ số batch không đổi.
        """
        import math

        self.rows = rows
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.weights = list(weights) if weights is not None else None
        if self.weights is not None and len(self.weights) != len(rows):
            raise ValueError(f"weights dài {len(self.weights)} ≠ số row {len(rows)}")
        if lengths is not None and len(lengths) != len(rows):
            raise ValueError(f"lengths dài {len(lengths)} ≠ số row {len(rows)}")
        self.epoch = 0
        self._groups: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            count = int(lengths[index]) if lengths is not None else len(row.get("durations") or [])
            self._groups.setdefault(count, []).append(index)
        self._count = sum(
            (len(group) // self.batch_size if self.drop_last
             else math.ceil(len(group) / self.batch_size))
            for group in self._groups.values()
        )

    def __len__(self) -> int:
        """Tổng số batch mỗi epoch."""
        return self._count

    def __iter__(self):
        """Yield từng batch (list index cùng N)."""
        import random

        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        batches: list[list[int]] = []
        for group in self._groups.values():
            if self.weights is None:
                indices = list(group)
            else:
                # Lấy có hoàn lại theo trọng số (cùng cỡ nhóm) — hàng hiếm xuất hiện nhiều hơn.
                indices = rng.choices(group, weights=[self.weights[i] for i in group], k=len(group))
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start:start + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

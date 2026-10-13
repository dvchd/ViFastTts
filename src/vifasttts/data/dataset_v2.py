"""Dataset v2 cấp slot (docs/ACOUSTIC_V2.md §3, §23; T2.2).

`SlotDataset` đọc manifest `_slot4` + cache feature, cắt theo `frame_range` rồi dựng
mọi target cấp segment qua `data/v2_targets.segment_targets`. Không fallback khi thiếu
`seg_durations`/`frame_range` (F3: raise `ValueError`).
"""

from __future__ import annotations

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
    ) -> None:
        """Nạp manifest + config feature + stats F0; `cache_dir` bật đọc cache."""
        self.rows = ManifestDataset(manifest).rows
        self.config = config
        self.f0_stats = f0_stats
        self.vocabs = ComponentVocabs.default()
        self.cache = FeatureCache(cache_dir, config) if cache_dir is not None else None

    def __len__(self) -> int:
        """Số row trong manifest."""
        return len(self.rows)

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
    for key in ("mel", "f0", "vuv", "energy"):
        first = items[0][key]
        padded = torch.zeros((batch, frame_max, *first.shape[1:]), dtype=first.dtype)
        for i, item in enumerate(items):
            length = item[key].size(0)
            padded[i, :length] = item[key]
        output[key] = padded
    assert output["mel"].shape[1:] == (frame_max, mel_dim)
    for key in items[0]:
        if key in ("id", "mel", "f0", "vuv", "energy", "frame_lengths"):
            continue
        output[key] = torch.stack([item[key] for item in items])
    return output

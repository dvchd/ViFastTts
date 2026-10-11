from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from vifasttts.data.cache import FeatureCache
from vifasttts.data.features import extract_features
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.types import (
    AbstractSyllable,
    Boundaries,
    Orthography,
    ParsedSyllable,
    ParseStatus,
    Tone,
)


class ManifestDataset:
    def __init__(self, manifest: str | Path) -> None:
        with Path(manifest).open("r", encoding="utf-8") as f:
            self.rows: list[dict[str, Any]] = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def parsed_from_dict(row: dict[str, Any]) -> ParsedSyllable:
    """Dựng lại ParsedSyllable từ dict trong manifest."""
    abstract = row["abstract"]
    return ParsedSyllable(
        raw=row["raw"],
        normalized=row["normalized"],
        orthography=Orthography(**row["orthography"]),
        abstract=AbstractSyllable(
            onset=abstract["onset"], medial=abstract["medial"], nucleus=abstract["nucleus"],
            coda=abstract["coda"], tone=Tone(abstract["tone"]),
        ),
        boundaries=Boundaries(**row.get("boundaries", {})),
        status=ParseStatus(row.get("status", "VALID")),
        confidence=float(row.get("confidence", 1.0)),
        punctuation=row.get("punctuation"),
    )


def encode_syllables(rows: list[dict[str, Any]], vocabs: ComponentVocabs) -> list[list[int]]:
    return [vocabs.encode(parsed_from_dict(row)) for row in rows]


def allocate_durations(frames: int, syllables: int) -> list[int]:
    """Chia đều frame cho các âm tiết khi chưa có forced alignment.

    Placeholder có tính deterministic: tổng đúng bằng số frame, mỗi âm tiết
    tối thiểu 1 frame khi đủ frame. Thay bằng duration do aligner bên ngoài
    khi có (trường "durations" trong manifest).
    """
    if syllables <= 0:
        return []
    if frames <= 0:
        return [0] * syllables
    if frames < syllables:
        return [1] * frames + [0] * (syllables - frames)
    base = frames // syllables
    durations = [base] * syllables
    for index in range(frames - base * syllables):
        durations[index] += 1
    return durations


class TtsDataset(Dataset):
    """Dataset huấn luyện acoustic model từ manifest JSONL.

    Mỗi mẫu trả về tensor cấp âm tiết (features, durations, f0, vuv, energy)
    và mel cấp frame. `durations` lấy từ trường "durations" trong manifest
    (nếu có, ví dụ từ forced alignment); nếu không, chia đều theo số frame.

    `cache_dir` bật cache feature mel/energy/f0/vuv theo hash(audio, config)
    để bỏ qua decode/STFT ở epoch sau (xem `data/cache.py`).
    """

    def __init__(self, manifest: str | Path, config: dict[str, Any], cache_dir: str | Path | None = None) -> None:
        self.rows = ManifestDataset(manifest).rows
        self.config = config
        self.vocabs = ComponentVocabs.default()
        self.sample_rate = int(config.get("sample_rate", 24000))
        self.cache = FeatureCache(cache_dir, config) if cache_dir is not None else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        features = torch.tensor(encode_syllables(row["syllables"], self.vocabs), dtype=torch.long)
        syllable_count = features.size(0)
        if self.cache is not None:
            extracted = self.cache.get(row)
            if extracted is None:
                extracted = extract_features(row["audio_path"], self.config)
                self.cache.put(row, extracted)
        else:
            extracted = extract_features(row["audio_path"], self.config)
        mel = extracted["mel"]
        frames = mel.size(0)
        durations_list = row.get("durations")
        if durations_list and sum(durations_list) == frames and len(durations_list) == syllable_count:
            durations = torch.tensor(durations_list, dtype=torch.long)
        else:
            durations = torch.tensor(allocate_durations(frames, syllable_count), dtype=torch.long)

        f0 = torch.zeros(syllable_count)
        vuv = torch.zeros(syllable_count)
        energy = torch.zeros(syllable_count)
        start = 0
        frame_f0 = extracted["f0"]
        frame_vuv = extracted["vuv"]
        frame_energy = extracted["energy"]
        for i, duration in enumerate(durations.tolist()):
            end = min(frames, start + duration)
            if end > start:
                f0[i] = frame_f0[start:end].mean()
                vuv[i] = (frame_vuv[start:end].mean() > 0.5).float()
                energy[i] = frame_energy[start:end].mean()
            start = end
        return {
            "features": features,
            "durations": durations,
            "f0": f0,
            "vuv": vuv,
            "energy": energy,
            "mel": mel,
        }


def collate_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad batch: features [B, Nmax, 7], mel [B, Tmax, M] + length masks."""
    batch = len(items)
    syllable_max = max(item["features"].size(0) for item in items)
    frame_max = max(item["mel"].size(0) for item in items)
    mel_dim = items[0]["mel"].size(1)
    features = torch.zeros(batch, syllable_max, 7, dtype=torch.long)
    durations = torch.zeros(batch, syllable_max, dtype=torch.long)
    f0 = torch.zeros(batch, syllable_max)
    vuv = torch.zeros(batch, syllable_max)
    energy = torch.zeros(batch, syllable_max)
    mel = torch.zeros(batch, frame_max, mel_dim)
    syllable_lengths = torch.zeros(batch, dtype=torch.long)
    mel_lengths = torch.zeros(batch, dtype=torch.long)
    for i, item in enumerate(items):
        n, t = item["features"].size(0), item["mel"].size(0)
        features[i, :n] = item["features"]
        durations[i, :n] = item["durations"]
        f0[i, :n] = item["f0"]
        vuv[i, :n] = item["vuv"]
        energy[i, :n] = item["energy"]
        mel[i, :t] = item["mel"]
        syllable_lengths[i] = n
        mel_lengths[i] = t
    return {
        "features": features, "durations": durations, "f0": f0, "vuv": vuv,
        "energy": energy, "mel": mel,
        "syllable_lengths": syllable_lengths, "mel_lengths": mel_lengths,
    }

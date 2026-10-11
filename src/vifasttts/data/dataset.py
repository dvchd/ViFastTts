"""Dataset cho training: acoustic (TtsDataset) và vocoder (VocoderDataset) + collate.

Mỗi mẫu acoustic trả tensor cấp âm tiết + mel cấp frame; chi tiết: `docs/MODEL_FLOW.md` §4.1.
"""

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
    """Đọc manifest JSONL thành danh sách dict (mỗi dòng một mẫu)."""

    def __init__(self, manifest: str | Path) -> None:
        """Đọc toàn bộ manifest vào bộ nhớ (manifest thường nhỏ, chỉ chứa metadata)."""
        with Path(manifest).open("r", encoding="utf-8") as f:
            self.rows: list[dict[str, Any]] = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        """Số mẫu trong manifest."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Dict của mẫu thứ `index`."""
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
    """Mã hoá danh sách âm tiết (dict trong manifest) thành list 7 ID."""
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
        """Nạp manifest + config; `cache_dir` bật cache mel/f0/vuv giữa các epoch."""
        self.rows = ManifestDataset(manifest).rows
        self.config = config
        self.vocabs = ComponentVocabs.default()
        self.sample_rate = int(config.get("sample_rate", 24000))
        self.cache = FeatureCache(cache_dir, config) if cache_dir is not None else None

    def __len__(self) -> int:
        """Số mẫu trong dataset."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Mẫu thứ index: features/durations/f0/vuv/energy/mel (+ phonation nếu bật)."""
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
        frame_phonation = None
        if "hnr" in extracted:
            frame_phonation = torch.stack(
                [extracted["hnr"], extracted["tilt"], extracted["glottalization"]], dim=-1
            )
        phonation = torch.zeros(syllable_count, 3)
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
                if frame_phonation is not None:
                    phonation[i] = frame_phonation[start:end].mean(dim=0)
            start = end
        item = {
            "features": features,
            "durations": durations,
            "f0": f0,
            "vuv": vuv,
            "energy": energy,
            "mel": mel,
        }
        if frame_phonation is not None:
            item["phonation"] = phonation
        return item


def collate_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad batch: features [B, Nmax, 7], mel [B, Tmax, M] + length masks.

    Kèm `phonation` [B, Nmax, 3] khi dataset bật kênh phonation.
    """
    batch = len(items)
    syllable_max = max(item["features"].size(0) for item in items)
    frame_max = max(item["mel"].size(0) for item in items)
    mel_dim = items[0]["mel"].size(1)
    has_phonation = "phonation" in items[0]
    features = torch.zeros(batch, syllable_max, 7, dtype=torch.long)
    durations = torch.zeros(batch, syllable_max, dtype=torch.long)
    f0 = torch.zeros(batch, syllable_max)
    vuv = torch.zeros(batch, syllable_max)
    energy = torch.zeros(batch, syllable_max)
    phonation = torch.zeros(batch, syllable_max, 3)
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
        if has_phonation:
            phonation[i, :n] = item["phonation"]
        mel[i, :t] = item["mel"]
        syllable_lengths[i] = n
        mel_lengths[i] = t
    output = {
        "features": features, "durations": durations, "f0": f0, "vuv": vuv,
        "energy": energy, "mel": mel,
        "syllable_lengths": syllable_lengths, "mel_lengths": mel_lengths,
    }
    if has_phonation:
        output["phonation"] = phonation
    return output


class VocoderDataset(Dataset):
    """Mel thật + waveform thật cho huấn luyện vocoder.

    Dùng cùng cache feature như `TtsDataset`; waveform được cắt về đúng số
    mẫu `mel_frames * hop` (center=True) để khớp với đầu ra generator.
    """

    def __init__(self, manifest: str | Path, config: dict[str, Any], cache_dir: str | Path | None = None) -> None:
        """Nạp manifest + config cho huấn luyện vocoder."""
        self.rows = ManifestDataset(manifest).rows
        self.config = config
        self.hop_length = int(config["mel"]["hop_length"])
        self.sample_rate = int(config.get("sample_rate", 24000))
        self.cache = FeatureCache(cache_dir, config) if cache_dir is not None else None

    def __len__(self) -> int:
        """Số mẫu trong dataset."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Mẫu thứ index: mel [T, 100] + waveform thật [T*hop] (đã pad/khớp)."""
        from vifasttts.data.features import load_wav

        row = self.rows[index]
        if self.cache is not None:
            extracted = self.cache.get(row)
            if extracted is None:
                extracted = extract_features(row["audio_path"], self.config)
                self.cache.put(row, extracted)
        else:
            extracted = extract_features(row["audio_path"], self.config)
        mel = extracted["mel"]
        wav = load_wav(row["audio_path"], self.sample_rate)
        target_length = mel.size(0) * self.hop_length
        if wav.numel() < target_length:
            wav = torch.nn.functional.pad(wav, (0, target_length - wav.numel()))
        return {"mel": mel, "wav": wav[:target_length]}


def collate_vocoder_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad batch vocoder: mel [B, Tmax, M], wav [B, Tmax*hop]."""
    batch = len(items)
    frame_max = max(item["mel"].size(0) for item in items)
    mel_dim = items[0]["mel"].size(1)
    hop_length = items[0]["wav"].numel() // max(1, items[0]["mel"].size(0))
    mel = torch.zeros(batch, frame_max, mel_dim)
    wav = torch.zeros(batch, frame_max * hop_length)
    mel_lengths = torch.zeros(batch, dtype=torch.long)
    for i, item in enumerate(items):
        frames = item["mel"].size(0)
        mel[i, :frames] = item["mel"]
        wav[i, : item["wav"].numel()] = item["wav"]
        mel_lengths[i] = frames
    return {"mel": mel, "wav": wav, "mel_lengths": mel_lengths}

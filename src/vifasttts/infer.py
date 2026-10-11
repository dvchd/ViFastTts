from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from vifasttts.config import validate_config
from vifasttts.frontend.chunking import chunk_text, pause_seconds
from vifasttts.frontend.pipeline import parse_text


def write_wav(path: str | Path, waveform, sample_rate: int = 24000) -> None:
    """Ghi WAV PCM 16-bit mono; nhận cả tensor torch lẫn mảng numpy."""
    if hasattr(waveform, "detach"):
        array = waveform.detach().cpu().numpy()  # type: ignore[union-attr]
    else:
        array = np.asarray(waveform, dtype=np.float32)
    pcm = (np.clip(array, -1.0, 1.0) * 32767.0).round().astype("<i2")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def _render_chunk(model, vocoder, vocabs, text: str, dialect: str, speed: float, options) -> np.ndarray | None:
    """Một đoạn văn bản → waveform numpy (None nếu đoạn không có âm tiết)."""
    import torch

    items = parse_text(text, dialect=dialect, options=options)
    if not items:
        return None
    features = torch.tensor([[vocabs.encode(item) for item in items]], dtype=torch.long)
    with torch.no_grad():
        outputs = model(features)
        if speed != 1.0:
            durations = torch.round(outputs["durations"].float() / max(speed, 1e-3)).clamp_min(1).long()
            outputs = model(features, durations=durations)
        mel = outputs["mel"][0]
        waveform = vocoder(mel.unsqueeze(0))[0]
    return np.asarray(waveform.cpu().numpy(), dtype=np.float32)


def synthesize(
    config: dict,
    checkpoint: str,
    text: str,
    output: str,
    *,
    dialect: str = "north",
    speed: float = 1.0,
    vocoder_checkpoint: str | None = None,
    overrides: dict[str, str] | None = None,
    peak_normalize: bool = False,
    max_chars: int = 400,
) -> tuple[int, int]:
    """text -> WAV. Trả về (số mẫu, sample_rate).

    - Văn bản dài được chunk theo câu (xem `frontend/chunking.py`) rồi ghép
      với khoảng nghỉ theo dấu câu.
    - `overrides`: từ điển "mặt chữ → cách đọc" (khớp hoa thường) áp trước mọi luật.
    - `peak_normalize`: chuẩn hoá đỉnh về -3 dBFS (mặc định chỉ fade đầu/cuối).
    """
    import torch

    from vifasttts.frontend.normalizer import NormalizerOptions
    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.models.istftnet import ISTFTNetLite
    from vifasttts.models.vifasttts import ViFastTtsModel
    from vifasttts.runtime.postprocess import postprocess

    validate_config(config, "acoustic")
    chunks = chunk_text(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("Không có âm tiết nào để tổng hợp")
    options = NormalizerOptions(dialect=dialect, exact_overrides=dict(overrides or {}))  # type: ignore[arg-type]

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda), len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    model = ViFastTtsModel(config, sizes)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval()

    mel_config = config["mel"]
    sample_rate = int(config.get("sample_rate", 24000))
    vocoder_training = config.get("vocoder_training", {})
    # Cùng quy ước với train-vocoder: hidden/upsample_ratio mặc định 256/4.
    vocoder = ISTFTNetLite(
        n_mels=int(mel_config["n_mels"]),
        hidden=int(vocoder_training.get("hidden", 256)),
        n_fft=int(mel_config["n_fft"]),
        hop_length=int(mel_config["hop_length"]),
        upsample_ratio=int(vocoder_training.get("upsample_ratio", 4)),
    )
    if vocoder_checkpoint and Path(vocoder_checkpoint).exists():
        vocoder_state = torch.load(vocoder_checkpoint, map_location="cpu")
        vocoder.load_state_dict(vocoder_state.get("vocoder", vocoder_state))
    elif isinstance(state, dict) and "vocoder" in state:
        vocoder.load_state_dict(state["vocoder"])
    else:
        print("Cảnh báo: chưa có trọng số vocoder (random weights). Huấn luyện vocoder theo docs/TRAINING.md.")
    vocoder.eval()

    pieces: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        rendered = _render_chunk(model, vocoder, vocabs, chunk, dialect, speed, options)
        if rendered is None:
            continue
        pieces.append(rendered)
        if index < len(chunks) - 1:
            pause = int(sample_rate * pause_seconds(chunk))
            if pause:
                pieces.append(np.zeros(pause, dtype=np.float32))
    if not pieces:
        raise ValueError("Không có âm tiết nào để tổng hợp")

    waveform = postprocess(np.concatenate(pieces), sample_rate, target_db=-3.0 if peak_normalize else None)
    write_wav(output, waveform, sample_rate)
    return int(waveform.size), sample_rate


def load_overrides(path: str | Path) -> dict[str, str]:
    """Đọc file JSON {"mặt chữ": "cách đọc"} cho normalizer."""
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
        raise ValueError("File overrides phải là JSON object chuỗi→chuỗi")
    return dict(payload)

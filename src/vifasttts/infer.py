from __future__ import annotations

import wave
from pathlib import Path


def write_wav(path: str | Path, waveform, sample_rate: int = 24000) -> None:
    """Ghi WAV PCM 16-bit mono bằng stdlib (không cần soundfile)."""
    import torch

    pcm = waveform.detach().clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16).cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


def synthesize(
    config: dict,
    checkpoint: str,
    text: str,
    output: str,
    *,
    dialect: str = "north",
    speed: float = 1.0,
    vocoder_checkpoint: str | None = None,
) -> tuple[int, int]:
    """text -> WAV. Trả về (số mẫu, sample_rate)."""
    import torch

    from vifasttts.frontend.pipeline import parse_text
    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.models.istftnet import ISTFTNetLite
    from vifasttts.models.vifasttts import ViFastTtsModel

    items = parse_text(text, dialect=dialect)
    if not items:
        raise ValueError("Không có âm tiết nào để tổng hợp")
    vocabs = ComponentVocabs.default()
    features = torch.tensor([[vocabs.encode(item) for item in items]], dtype=torch.long)
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda), len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]

    model = ViFastTtsModel(config, sizes)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        if speed != 1.0:
            durations = torch.round(outputs["durations"].float() / max(speed, 1e-3)).clamp_min(1).long()
            outputs = model(features, durations=durations)
    mel = outputs["mel"][0]

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
    with torch.no_grad():
        waveform = vocoder(mel.unsqueeze(0))[0]
    write_wav(output, waveform, sample_rate)
    return int(waveform.numel()), sample_rate

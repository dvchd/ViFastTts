"""Chạy suy luận ONNX (acoustic + vocoder) không cần PyTorch.

- `OnnxAcousticModel`: 7 ID → mel/durations/log_duration/f0...
- `OnnxVocoder`: mel → phổ (real/imag) → waveform bằng iSTFT numpy.
- `synthesize_onnx`: text → WAV, tự chunk câu dài và ghép khoảng nghỉ.

Metadata của file ONNX (do `export.py` ghi) cung cấp sample_rate/n_fft/hop_length;
nếu thiếu thì truyền tường minh khi khởi tạo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vifasttts.frontend.chunking import chunk_text, pause_seconds
from vifasttts.frontend.normalizer import NormalizerOptions
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.runtime.istft import istft
from vifasttts.runtime.postprocess import postprocess


def _ort_session(path: str | Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError("Cần cài extra 'onnx' (onnxruntime) để chạy ONNX") from exc
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def read_metadata(path: str | Path) -> dict[str, str]:
    """Đọc custom metadata của file ONNX qua onnxruntime."""
    session = _ort_session(path)
    return dict(session.get_modelmeta().custom_metadata_map or {})


class OnnxAcousticModel:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.session = _ort_session(path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map or {})

    def features_from_text(
        self,
        text: str,
        *,
        dialect: str = "north",
        options: NormalizerOptions | None = None,
    ) -> np.ndarray:
        items = parse_text(text, dialect=dialect, options=options)
        if not items:
            raise ValueError("Không có âm tiết nào để tổng hợp")
        vocabs = ComponentVocabs.default()
        return np.array([[vocabs.encode(item) for item in items]], dtype=np.int64)

    def run(self, features: np.ndarray) -> dict[str, np.ndarray]:
        outputs = self.session.run(None, {self.input_name: features.astype(np.int64)})
        return dict(zip(self.output_names, outputs))


class OnnxVocoder:
    def __init__(
        self,
        path: str | Path,
        *,
        n_fft: int | None = None,
        hop_length: int | None = None,
        sample_rate: int | None = None,
    ) -> None:
        self.path = str(path)
        self.session = _ort_session(path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        metadata = dict(self.session.get_modelmeta().custom_metadata_map or {})
        self.n_fft = int(n_fft if n_fft is not None else metadata.get("n_fft", 0))
        self.hop_length = int(hop_length if hop_length is not None else metadata.get("hop_length", 0))
        self.sample_rate = int(sample_rate if sample_rate is not None else metadata.get("sample_rate", 0))
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError(
                "Thiếu n_fft/hop_length: file ONNX không có metadata, hãy truyền tường minh"
            )

    def spectrogram(self, mel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        outputs = self.session.run(None, {self.input_name: mel.astype(np.float32)})
        named = dict(zip(self.output_names, outputs))
        real = named.get("spec_real", outputs[0])
        imag = named.get("spec_imag", outputs[1] if len(outputs) > 1 else outputs[0] * 0)
        return np.asarray(real)[0], np.asarray(imag)[0]

    def waveform(self, mel: np.ndarray) -> np.ndarray:
        real, imag = self.spectrogram(mel)
        return istft(real, imag, n_fft=self.n_fft, hop_length=self.hop_length)


def _stretch_mel(mel: np.ndarray, speed: float) -> np.ndarray:
    """Co giãn mel theo tốc độ (xấp xỉ nearest-frame; chỉ dùng cho runtime ONNX)."""
    length = mel.shape[1]
    target = max(1, round(length / max(speed, 1e-3)))
    index = np.clip((np.arange(target) * speed).astype(np.int64), 0, length - 1)
    return mel[:, index]


def synthesize_onnx(
    text: str,
    acoustic_path: str | Path,
    output: str | Path,
    *,
    vocoder_path: str | Path,
    dialect: str = "north",
    speed: float = 1.0,
    overrides: dict[str, str] | None = None,
    peak_normalize: bool = False,
    max_chars: int = 400,
    sample_rate: int | None = None,
) -> tuple[int, int]:
    """text → WAV bằng ONNX Runtime + numpy (không cần PyTorch)."""
    from vifasttts.infer import write_wav

    acoustic = OnnxAcousticModel(acoustic_path)
    vocoder = OnnxVocoder(vocoder_path, sample_rate=sample_rate)
    rate = sample_rate or vocoder.sample_rate or int(acoustic.metadata.get("sample_rate", 24000))
    options = NormalizerOptions(dialect=dialect, exact_overrides=dict(overrides or {}))  # type: ignore[arg-type]

    chunks = chunk_text(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("Không có âm tiết nào để tổng hợp")
    waveforms: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        features = acoustic.features_from_text(chunk, dialect=dialect, options=options)
        outputs = acoustic.run(features)
        mel = np.asarray(outputs["mel"])
        if speed != 1.0:
            mel = _stretch_mel(mel, speed)
        waveforms.append(vocoder.waveform(mel))
        if index < len(chunks) - 1:
            pause = int(rate * pause_seconds(chunk))
            if pause:
                waveforms.append(np.zeros(pause, dtype=np.float32))
    waveform = np.concatenate(waveforms)
    waveform = postprocess(waveform, rate, target_db=-3.0 if peak_normalize else None)
    write_wav(output, waveform, rate)
    return int(waveform.size), rate

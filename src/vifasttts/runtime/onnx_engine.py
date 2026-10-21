"""Chạy suy luận ONNX (acoustic + vocoder) không cần PyTorch.

- `OnnxAcousticModel`: 7 ID → mel/durations/log_duration/f0...
- `OnnxVocoder`: mel → phổ (real/imag) → waveform bằng iSTFT numpy.
- `synthesize_onnx`: text → WAV, tự chunk câu dài và ghép khoảng nghỉ.

Metadata của file ONNX (do `export.py` ghi) cung cấp sample_rate/n_fft/hop_length;
nếu thiếu thì truyền tường minh khi khởi tạo.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from vifasttts.frontend.chunking import (
    STEADY_PAUSE_COMMA,
    STEADY_PAUSE_DEFAULT,
    STEADY_PAUSE_PERIOD,
    chunk_text,
    pause_seconds,
    split_clauses,
    steady_pause_seconds,
)
from vifasttts.frontend.normalizer import NormalizerOptions
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.runtime.istft import istft
from vifasttts.runtime.postprocess import postprocess


def _import_ort():
    """Import onnxruntime với thông báo lỗi thân thiện nếu thiếu extra 'onnx'."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError("Cần cài extra 'onnx' (onnxruntime) để chạy ONNX") from exc
    return ort


def _ort_session(path: str | Path, provider: str = "auto"):
    """Mở InferenceSession với danh sách provider đã giải.

    `VIFASTT_INTRA_THREADS` (tuỳ chọn) đặt số luồng intra-op — dùng để kiểm tất định
    khi đổi 1/2/4 luồng và để chốt RTF trên máy đích.
    """
    import os

    ort = _import_ort()
    options = ort.SessionOptions()
    # A0 v11 §0.3: ghim 2 luồng làm mặc định production để md5 ổn định tuyệt đối
    # (đổi số luồng làm lệch ≤ 1 LSB ở 0,5 % mẫu; xem báo cáo Việc 2.2).
    options.intra_op_num_threads = int(os.environ.get("VIFASTT_INTRA_THREADS", "2"))
    return ort.InferenceSession(
        str(path),
        providers=resolve_providers(provider),
        sess_options=options,
    )


def resolve_providers(provider: str = "auto") -> list[Any]:
    """Danh sách execution provider cho ONNX Runtime theo nền tảng.

    - `auto` (mặc định): CUDA nếu có, ngược lại CPU. **Không tự chọn CoreML**
      vì đo trên Mac cho thấy CoreML EP chậm hơn CPU ở kích thước model này
      (xem `docs/APPLE_SILICON.md`).
    - `cpu` | `cuda` | `tensorrt` | `coreml` (CoreML bật `MLComputeUnits=ALL` để
      có thể dùng GPU/Neural Engine; yêu cầu mạng tĩnh nên dùng
      `RequireStaticInputShapes=1`).
    - Provider không khả dụng → ValueError rõ ràng.
    """
    ort = _import_ort()

    available = set(ort.get_available_providers())
    choice = (provider or "auto").strip().lower()
    if choice == "auto":
        choice = "cuda" if "CUDAExecutionProvider" in available else "cpu"
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise ValueError("Yêu cầu provider=cuda nhưng onnxruntime này không có CUDAExecutionProvider")
        return [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    if choice == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise ValueError("Yêu cầu provider=tensorrt nhưng không có TensorrtExecutionProvider")
        return [("TensorrtExecutionProvider", {}), "CPUExecutionProvider"]
    if choice == "coreml":
        if "CoreMLExecutionProvider" not in available:
            raise ValueError("Yêu cầu provider=coreml nhưng onnxruntime này không có CoreMLExecutionProvider")
        import os

        cache_dir = Path(os.environ.get("VIFASTTTS_COREML_CACHE", Path.home() / ".cache" / "vifasttts" / "coreml"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        return [
            ("CoreMLExecutionProvider", {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",  # CPU + GPU + Neural Engine
                # Ổn định hơn với model có shape động (vd LengthRegulator).
                "RequireStaticInputShapes": "1",
                # ORT biên dịch subgraph sang Core ML mỗi session; cache lại để mở nhanh.
                "ModelCacheDirectory": str(cache_dir),
            }),
            "CPUExecutionProvider",
        ]
    raise ValueError(f"provider không hợp lệ: {provider!r} (auto|cpu|cuda|coreml|tensorrt)")


def read_metadata(path: str | Path) -> dict[str, str]:
    """Đọc custom metadata của file ONNX qua onnxruntime."""
    session = _ort_session(path)
    return dict(session.get_modelmeta().custom_metadata_map or {})


class OnnxAcousticModel:
    """Bọc session ONNX của model acoustic: text → features → dict đầu ra."""

    def __init__(self, path: str | Path, provider: str = "auto") -> None:
        """`provider`: auto|cpu|cuda|coreml|tensorrt (xem `resolve_providers`)."""
        self.path = str(path)
        self.session = _ort_session(path, provider)
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.input_name = self.input_names[0]
        self.speed_input_name = self.input_names[1] if len(self.input_names) > 1 else None
        # V11.A7: graph mới có input thứ ba `gate_threshold` (cuối); graph cũ 2 input vẫn chạy
        self.gate_input_name = self.input_names[2] if len(self.input_names) > 2 else None
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map or {})
        self.is_v2 = self.metadata.get("arch") == "v2_slot5" or self.speed_input_name is not None

    def features_from_text(
        self,
        text: str,
        *,
        dialect: str = "north",
        options: NormalizerOptions | None = None,
    ) -> np.ndarray:
        """Parse text → mảng int64 [1, N, 7] cho session ONNX."""
        items = parse_text(text, dialect=dialect, options=options)
        if not items:
            raise ValueError("Không có âm tiết nào để tổng hợp")
        vocabs = ComponentVocabs.default()
        return np.array([[vocabs.encode(item) for item in items]], dtype=np.int64)

    def run(self, features: np.ndarray, *, speed: float = 1.0,
            gate_threshold: float = 0.5) -> dict[str, np.ndarray]:
        """Chạy session; v2 truyền thêm `speed` (+ `gate_threshold` nếu graph có input thứ ba)."""
        if self.is_v2 and self.speed_input_name is not None:
            feeds = {
                self.input_name: features.astype(np.int64),
                self.speed_input_name: np.array([speed], dtype=np.float32),
            }
            gate_name = getattr(self, "gate_input_name", None)
            if gate_name is not None:
                feeds[gate_name] = np.array([gate_threshold], dtype=np.float32)
            outputs = self.session.run(None, feeds)
        else:
            outputs = self.session.run(None, {self.input_name: features.astype(np.int64)})
        return dict(zip(self.output_names, outputs))


class OnnxVocoder:
    """Bọc session ONNX của vocoder: mel → phổ (real/imag) → waveform (iSTFT numpy)."""

    def __init__(
        self,
        path: str | Path,
        *,
        n_fft: int | None = None,
        hop_length: int | None = None,
        sample_rate: int | None = None,
        provider: str = "auto",
    ) -> None:
        """Thông số iSTFT lấy từ metadata ONNX; truyền tường minh nếu file thiếu metadata."""
        self.path = str(path)
        self.session = _ort_session(path, provider)
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
        """mel [1, T, M] → (real, imag) [bins, T*ratio]."""
        outputs = self.session.run(None, {self.input_name: mel.astype(np.float32)})
        named = dict(zip(self.output_names, outputs))
        real = named.get("spec_real", outputs[0])
        imag = named.get("spec_imag", outputs[1] if len(outputs) > 1 else outputs[0] * 0)
        return np.asarray(real)[0], np.asarray(imag)[0]

    def waveform(self, mel: np.ndarray) -> np.ndarray:
        """mel [1, T, M] → waveform float32 (iSTFT overlap-add, khớp torch.istft)."""
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
    gate_threshold: float = 0.5,
    overrides: dict[str, str] | None = None,
    peak_normalize: bool = False,
    max_chars: int = 400,
    sample_rate: int | None = None,
    provider: str = "auto",
    steady: bool = False,
    pause_comma: float = STEADY_PAUSE_COMMA,
    pause_period: float = STEADY_PAUSE_PERIOD,
    pause_default: float = STEADY_PAUSE_DEFAULT,
    target_lufs: float | None = None,
) -> tuple[int, int]:
    """text → WAV bằng ONNX Runtime + numpy (không cần PyTorch).

    `provider`: auto|cpu|cuda|coreml|tensorrt — xem `resolve_providers`.
    `steady`: chế độ "đọc đều" — nghỉ cố định theo dấu câu (`iter_synthesize_onnx`).
    `target_lufs`: chuẩn hoá độ to cả file (None/0 = tắt), chặn true-peak −1 dBTP.
    """
    from vifasttts.infer import _apply_target_lufs, write_wav

    rate = sample_rate
    pieces: list[np.ndarray] = []
    for audio, chunk_rate in iter_synthesize_onnx(
        text, acoustic_path, vocoder_path=vocoder_path, dialect=dialect, speed=speed,
        gate_threshold=gate_threshold,
        overrides=overrides, max_chars=max_chars, sample_rate=sample_rate, provider=provider,
        steady=steady, pause_comma=pause_comma, pause_period=pause_period,
        pause_default=pause_default,
    ):
        rate = chunk_rate
        pieces.append(audio)
    waveform = postprocess(np.concatenate(pieces), int(rate or 24000),
                           target_db=-3.0 if peak_normalize else None)
    waveform = _apply_target_lufs(waveform, int(rate or 24000), target_lufs)
    write_wav(output, waveform, int(rate or 24000))
    return int(waveform.size), int(rate or 24000)


def iter_synthesize_onnx(
    text: str,
    acoustic_path: str | Path,
    *,
    vocoder_path: str | Path,
    dialect: str = "north",
    speed: float = 1.0,
    gate_threshold: float = 0.5,
    overrides: dict[str, str] | None = None,
    max_chars: int = 400,
    sample_rate: int | None = None,
    provider: str = "auto",
    steady: bool = False,
    pause_comma: float = STEADY_PAUSE_COMMA,
    pause_period: float = STEADY_PAUSE_PERIOD,
    pause_default: float = STEADY_PAUSE_DEFAULT,
) -> Iterator[tuple[np.ndarray, int]]:
    """Streaming ONNX: yield (waveform numpy, sample_rate) từng đoạn + khoảng nghỉ.

    `steady=True`: cắt tại mọi dấu câu + nghỉ cố định (`pause_comma` sau phẩy;
    `pause_period` sau chấm/hỏi/than; `pause_default` cho phần còn lại).
    """
    acoustic = OnnxAcousticModel(acoustic_path, provider)
    vocoder = OnnxVocoder(vocoder_path, sample_rate=sample_rate, provider=provider)
    rate = sample_rate or vocoder.sample_rate or int(acoustic.metadata.get("sample_rate", 24000))
    options = NormalizerOptions(dialect=dialect, exact_overrides=dict(overrides or {}))  # type: ignore[arg-type]

    chunks = split_clauses(text, max_chars=max_chars) if steady else chunk_text(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("Không có âm tiết nào để tổng hợp")
    rendered_any = False
    for index, chunk in enumerate(chunks):
        features = acoustic.features_from_text(chunk, dialect=dialect, options=options)
        is_v2 = acoustic.metadata.get("arch") == "v2_slot5"
        outputs = (acoustic.run(features, speed=speed, gate_threshold=gate_threshold)
                   if is_v2 else acoustic.run(features))
        mel = np.asarray(outputs["mel"])
        if speed != 1.0 and not is_v2:
            mel = _stretch_mel(mel, speed)
        yield vocoder.waveform(mel), rate
        rendered_any = True
        if index < len(chunks) - 1:
            seconds = (steady_pause_seconds(chunk, comma=pause_comma, period=pause_period,
                                            default=pause_default) if steady else pause_seconds(chunk))
            pause = int(rate * seconds)
            if pause:
                yield np.zeros(pause, dtype=np.float32), rate
    if not rendered_any:  # pragma: no cover - chunk_text đã bảo đảm không rỗng
        raise ValueError("Không có âm tiết nào để tổng hợp")

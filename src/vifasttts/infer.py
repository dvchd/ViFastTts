"""Suy luận bằng PyTorch: text → WAV (chunk câu dài, override, hậu xử lý).

`Synthesizer` giữ model + vocoder đã nạp để dùng cho batch/server/streaming;
hàm `synthesize` chạy một lần và `iter_synthesize` cho streaming là API tiện dụng.
Luồng chi tiết: `docs/MODEL_FLOW.md` §6.
"""

from __future__ import annotations

import warnings
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from vifasttts.config import validate_config
from vifasttts.frontend.chunking import (
    STEADY_PAUSE_COMMA,
    STEADY_PAUSE_DEFAULT,
    STEADY_PAUSE_PERIOD,
    chunk_text,
    pause_seconds,
    split_clauses,
    steady_pause_seconds,
)
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


#: f0_stats mặc định của repo khi checkpoint v2 thiếu (khớp `tools/eval_acoustic.py`).
V2_DEFAULT_F0_STATS = "data/codelung/f0_stats_v2.json"


def _v2_f0_stats_fallback(config: dict) -> dict | None:
    """f0_stats cho v2 khi checkpoint thiếu: `config f0_stats_path` → mặc định repo."""
    import json

    for candidate in (config.get("f0_stats_path"), V2_DEFAULT_F0_STATS):
        if candidate and Path(candidate).exists():
            return json.loads(Path(candidate).read_text(encoding="utf-8"))
    return None


class Synthesizer:
    """Giữ model + vocoder đã nạp để tổng hợp nhiều câu (batch/server/streaming).

    Hỗ trợ cả hai `arch` (đọc từ checkpoint bằng `arch_of_checkpoint`):
    - v1 (`v1_syl`, checkpoint thiếu `arch`): `_render` giữ hành vi cũ — 2 forward
      khi speed≠1 (forward 1 lấy durations, scale rồi forward 2).
    - v2 (`v2_slot5`): `_render` chỉ **một** forward `model(features, speed=[speed])`
      (speed float32 `[1]` nằm trong graph, model tự scale d_syl + gap —
      docs/ACOUSTIC_V2.md §5; mel ra đã gồm khoảng nghỉ G trong câu).

    `iter_synthesize`/`synthesize` giữ nguyên API nên CLI `synthesize`,
    `synthesize-batch`, `serve` (đi qua `Synthesizer`) tự có v2.
    """

    def __init__(
        self,
        config: dict,
        checkpoint: str,
        vocoder_checkpoint: str | None = None,
        *,
        device: str | None = None,
    ) -> None:
        """Nạp acoustic + vocoder lên `device` (auto = cuda→mps→cpu).

        Không có trọng số vocoder → cảnh báo RuntimeWarning (âm thanh random).
        """
        import torch

        from vifasttts.data.features import canonical_mel_config, verify_mel_hash
        from vifasttts.device import resolve_device
        from vifasttts.frontend.vocab import ComponentVocabs
        from vifasttts.models.factory import (
            ARCH_V2,
            arch_of_checkpoint,
            build_model,
            load_model_state,
        )
        from vifasttts.models.istftnet import build_generator, load_generator_state

        validate_config(config, "acoustic")
        self.config = config
        self.device = torch.device(resolve_device(device))
        self.sample_rate = int(config.get("sample_rate", 24000))
        self.vocabs = ComponentVocabs.default()
        sizes = [len(self.vocabs.onset), len(self.vocabs.medial), len(self.vocabs.nucleus),
                 len(self.vocabs.coda), len(self.vocabs.tone), len(self.vocabs.boundary),
                 len(self.vocabs.punctuation)]
        state = torch.load(checkpoint, map_location="cpu")
        verify_mel_hash(state, canonical_mel_config(config), str(checkpoint))
        self.arch = arch_of_checkpoint(state if isinstance(state, dict) else {})
        self.model = build_model(config, sizes, arch=self.arch)
        load_model_state(self.model, state.get("model", state))
        if self.arch == ARCH_V2 and hasattr(self.model, "load_f0_stats"):
            stats = state.get("f0_stats") if isinstance(state, dict) else None
            if stats is None:
                stats = _v2_f0_stats_fallback(config)
            if stats is not None:
                self.model.load_f0_stats(dict(stats))
            else:
                warnings.warn(
                    "Checkpoint v2 thiếu f0_stats và không tìm thấy "
                    f"`f0_stats_path`/`{V2_DEFAULT_F0_STATS}` — dùng hằng số mặc định "
                    "của model (cao độ có thể lệch).",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.model.to(self.device)
        self.model.eval()

        mel_config = config["mel"]
        vocoder_training = config.get("vocoder_training", {})
        # Cùng quy ước với train-vocoder: generator chọn theo `vocoder_training.generator`.
        self.vocoder = build_generator(vocoder_training, mel_config)
        if vocoder_checkpoint and Path(vocoder_checkpoint).exists():
            vocoder_state = torch.load(vocoder_checkpoint, map_location="cpu")
            verify_mel_hash(vocoder_state, canonical_mel_config(config), str(vocoder_checkpoint))
            load_generator_state(self.vocoder, vocoder_checkpoint, state=vocoder_state)
        elif isinstance(state, dict) and "vocoder" in state:
            self.vocoder.load_state_dict(state["vocoder"])
        elif getattr(self.vocoder, "pretrained_loaded", False):
            pass  # zero-shot: đã nạp checkpoint pretrained trong build_generator (config)
        else:
            warnings.warn(
                "Chưa có trọng số vocoder (random weights). Huấn luyện bằng "
                "`vifasttts train-vocoder` hoặc truyền --vocoder-checkpoint.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.vocoder.to(self.device)
        self.vocoder.eval()

    def _render(self, text: str, dialect: str, speed: float, options,
                gate_threshold: float = 0.5, sharpen_alpha: float = 0.0) -> np.ndarray | None:
        """Tổng hợp một đoạn: parse → model → vocoder; None nếu đoạn không có âm tiết.

        v2 (`v2_slot5`): **một** forward `model(features, speed=[speed])` — speed
        nằm trong graph (§5); v1: giữ hành vi cũ (2 forward khi speed≠1).
        """
        import torch

        from vifasttts.models.factory import ARCH_V2

        items = parse_text(text, dialect=dialect, options=options, resolve_oov=True)
        if not items:
            return None
        features = torch.tensor(
            [[self.vocabs.encode(item) for item in items]], dtype=torch.long, device=self.device,
        )
        with torch.no_grad():
            if self.arch == ARCH_V2:
                speed_t = torch.tensor([float(speed)], dtype=torch.float32, device=self.device)
                mel = self.model(features, speed=speed_t,
                                 gate_threshold=gate_threshold)["mel"][0]
            else:
                outputs = self.model(features)
                if speed != 1.0:
                    durations = torch.round(outputs["durations"].float() / max(speed, 1e-3)).clamp_min(1).long()
                    outputs = self.model(features, durations=durations)
                mel = outputs["mel"][0]
            from vifasttts.runtime.postprocess import sharpen_mel

            waveform = self.vocoder(sharpen_mel(mel, sharpen_alpha).unsqueeze(0))[0]
        return np.asarray(waveform.cpu().numpy(), dtype=np.float32)

    def iter_synthesize(
        self,
        text: str,
        *,
        dialect: str = "north",
        speed: float = 1.0,
        gate_threshold: float = 0.5,
        sharpen_alpha: float = 0.0,
        overrides: dict[str, str] | None = None,
        max_chars: int = 400,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Yield (waveform, sample_rate) từng đoạn — gồm cả khoảng nghỉ giữa câu."""
        from vifasttts.frontend.normalizer import NormalizerOptions

        options = NormalizerOptions(dialect=dialect, exact_overrides=dict(overrides or {}))  # type: ignore[arg-type]
        chunks = chunk_text(text, max_chars=max_chars)
        if not chunks:
            raise ValueError("Không có âm tiết nào để tổng hợp")
        rendered_any = False
        for index, chunk in enumerate(chunks):
            rendered = self._render(chunk, dialect, speed, options,
                                    gate_threshold=gate_threshold, sharpen_alpha=sharpen_alpha)
            if rendered is None:
                continue
            rendered_any = True
            yield rendered, self.sample_rate
            if index < len(chunks) - 1:
                silence = int(self.sample_rate * pause_seconds(chunk))
                if silence:
                    yield np.zeros(silence, dtype=np.float32), self.sample_rate
        if not rendered_any:
            raise ValueError("Không có âm tiết nào để tổng hợp")

    def synthesize(
        self,
        text: str,
        output: str | Path,
        *,
        dialect: str = "north",
        speed: float = 1.0,
        gate_threshold: float = 0.5,
        sharpen_alpha: float = 0.0,
        overrides: dict[str, str] | None = None,
        peak_normalize: bool = False,
        max_chars: int = 400,
    ) -> tuple[int, int]:
        """Ghép các đoạn từ `iter_synthesize`, hậu xử lý và ghi WAV."""
        from vifasttts.runtime.postprocess import postprocess

        pieces = [audio for audio, _ in self.iter_synthesize(
            text, dialect=dialect, speed=speed, gate_threshold=gate_threshold,
            sharpen_alpha=sharpen_alpha, overrides=overrides, max_chars=max_chars,
        )]
        waveform = postprocess(np.concatenate(pieces), self.sample_rate,
                               target_db=-3.0 if peak_normalize else None)
        write_wav(output, waveform, self.sample_rate)
        return int(waveform.size), self.sample_rate

    def iter_synthesize_steady(
        self,
        text: str,
        *,
        dialect: str = "north",
        speed: float = 1.0,
        gate_threshold: float = 0.999,
        sharpen_alpha: float = 0.0,
        overrides: dict[str, str] | None = None,
        max_chars: int = 400,
        pause_comma: float = STEADY_PAUSE_COMMA,
        pause_period: float = STEADY_PAUSE_PERIOD,
        pause_default: float = STEADY_PAUSE_DEFAULT,
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Chế độ "đọc đều": nghỉ CỐ ĐỊNH theo dấu câu, chặn pause model học.

        Cắt tại mọi dấu câu (`split_clauses`); mỗi mảnh tổng hợp với
        `gate_threshold` cao (mặc định 0,999 — gần như chặn hết pause nội câu),
        rồi chèn khoảng nghỉ: sau phẩy `pause_comma`, sau chấm/hỏi/than
        `pause_period`, mảnh không dấu `pause_default`.
        """
        from vifasttts.frontend.normalizer import NormalizerOptions

        options = NormalizerOptions(dialect=dialect, exact_overrides=dict(overrides or {}))  # type: ignore[arg-type]
        chunks = split_clauses(text, max_chars=max_chars)
        if not chunks:
            raise ValueError("Không có âm tiết nào để tổng hợp")
        rendered_any = False
        for index, chunk in enumerate(chunks):
            rendered = self._render(chunk, dialect, speed, options,
                                    gate_threshold=gate_threshold, sharpen_alpha=sharpen_alpha)
            if rendered is None:
                continue
            rendered_any = True
            yield rendered, self.sample_rate
            if index < len(chunks) - 1:
                seconds = steady_pause_seconds(chunk, comma=pause_comma,
                                               period=pause_period, default=pause_default)
                silence = int(self.sample_rate * seconds)
                if silence:
                    yield np.zeros(silence, dtype=np.float32), self.sample_rate
        if not rendered_any:
            raise ValueError("Không có âm tiết nào để tổng hợp")

    def synthesize_steady(
        self,
        text: str,
        output: str | Path,
        *,
        dialect: str = "north",
        speed: float = 1.0,
        gate_threshold: float = 0.999,
        sharpen_alpha: float = 0.0,
        overrides: dict[str, str] | None = None,
        peak_normalize: bool = False,
        max_chars: int = 400,
        pause_comma: float = STEADY_PAUSE_COMMA,
        pause_period: float = STEADY_PAUSE_PERIOD,
        pause_default: float = STEADY_PAUSE_DEFAULT,
    ) -> tuple[int, int]:
        """Ghép các mảnh "đọc đều" (pause cố định theo dấu câu), ghi WAV."""
        from vifasttts.runtime.postprocess import postprocess

        pieces = [audio for audio, _ in self.iter_synthesize_steady(
            text, dialect=dialect, speed=speed, gate_threshold=gate_threshold,
            sharpen_alpha=sharpen_alpha, overrides=overrides, max_chars=max_chars,
            pause_comma=pause_comma, pause_period=pause_period, pause_default=pause_default,
        )]
        waveform = postprocess(np.concatenate(pieces), self.sample_rate,
                               target_db=-3.0 if peak_normalize else None)
        write_wav(output, waveform, self.sample_rate)
        return int(waveform.size), self.sample_rate


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
    device: str | None = None,
) -> tuple[int, int]:
    """text -> WAV (nạp model một lần). Trả về (số mẫu, sample_rate)."""
    engine = Synthesizer(config, checkpoint, vocoder_checkpoint, device=device)
    return engine.synthesize(
        text, output, dialect=dialect, speed=speed, overrides=overrides,
        peak_normalize=peak_normalize, max_chars=max_chars,
    )


def iter_synthesize(
    config: dict,
    checkpoint: str,
    text: str,
    *,
    dialect: str = "north",
    speed: float = 1.0,
    vocoder_checkpoint: str | None = None,
    overrides: dict[str, str] | None = None,
    max_chars: int = 400,
    device: str | None = None,
) -> Iterator[tuple[np.ndarray, int]]:
    """Streaming: yield từng đoạn waveform khi tổng hợp (xem `Synthesizer`)."""
    engine = Synthesizer(config, checkpoint, vocoder_checkpoint, device=device)
    return engine.iter_synthesize(
        text, dialect=dialect, speed=speed, overrides=overrides, max_chars=max_chars,
    )


def synthesize_batch(
    config: dict,
    checkpoint: str,
    rows: list[tuple[str, str]],
    output_dir: str | Path,
    *,
    vocoder_checkpoint: str | None = None,
    dialect: str = "north",
    speed: float = 1.0,
    overrides: dict[str, str] | None = None,
    peak_normalize: bool = False,
    max_chars: int = 400,
    device: str | None = None,
) -> list[Path]:
    """Tổng hợp danh sách (id, text) → `<output_dir>/<id>.wav` (nạp model một lần)."""
    engine = Synthesizer(config, checkpoint, vocoder_checkpoint, device=device)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for sample_id, text in rows:
        path = directory / f"{sample_id}.wav"
        engine.synthesize(
            text, path, dialect=dialect, speed=speed, overrides=overrides,
            peak_normalize=peak_normalize, max_chars=max_chars,
        )
        paths.append(path)
    return paths


def load_overrides(path: str | Path) -> dict[str, str]:
    """Đọc file JSON {"mặt chữ": "cách đọc"} cho normalizer."""
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in payload.items()):
        raise ValueError("File overrides phải là JSON object chuỗi→chuỗi")
    return dict(payload)

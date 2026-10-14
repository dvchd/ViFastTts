"""Xuất ONNX: acoustic (7 ID → mel), vocoder (mel → phổ), kèm metadata + quantization.

Ghi chú kỹ thuật quan trọng (vì sao có vài lựa chọn lạ):
- `dynamo=False` vì dùng exporter TorchScript ổn định, không cần onnxscript.
- `exp-1` thay `expm1` trong model để export được (ONNX không có op tương ứng).
- Metadata (`sample_rate`, `n_fft`, `hop_length`...) để runtime không cần config.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from vifasttts.config import validate_config

PRECISIONS = ("fp32", "fp16", "int8")


def _export_torch(model, dummy, output: str, input_names: list[str], output_names: list[str],
                  dynamic_axes: dict, opset_version: int) -> None:
    """Exporter TorchScript (ổn định, không cần onnxscript); tự dò tham số `dynamo`."""
    import torch

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        torch.onnx.export(
            model, (dummy,), output,
            input_names=input_names, output_names=output_names,
            opset_version=opset_version, dynamic_axes=dynamic_axes, dynamo=False,
        )
    else:  # torch cũ không có tham số dynamo
        torch.onnx.export(
            model, (dummy,), output,
            input_names=input_names, output_names=output_names,
            opset_version=opset_version, dynamic_axes=dynamic_axes,
        )


def _write_metadata(path: str, props: dict[str, object]) -> None:
    """Ghi custom metadata (sample_rate, n_fft...) để runtime đọc lại."""
    try:
        import onnx
    except ImportError:  # pragma: no cover - thiếu onnx thì runtime cần tham số tường minh
        return
    model = onnx.load(str(path))
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key = str(key)
        entry.value = str(value)
    onnx.save(model, str(path))


def _select_outputs_module(model, names: tuple[str, ...]):
    """Module con chọn đúng bộ output theo hợp đồng ONNX (loại `mel_pre` và key phụ khỏi graph).

    Bắt buộc khi `use_postnet=True`: dict output của model có `mel_pre` → exporter TorchScript
    gán tên theo vị trí làm **lệch tên 1 slot** (`tone_logits`←mel_pre, `lengths`←tone_logits,
    `lengths` thật bị đặt tên tự động) — đã đo thực tế trên torch 2.14.
    Hàm factory (không class module-level) để giữ `import torch` lazily như phần còn lại.
    """
    import torch

    class _Wrapper(torch.nn.Module):
        """Bọc model (submodule — không dùng closure kẻo tham số bị bake thành constant)."""

        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.names = tuple(names)

        def forward(self, features):  # type: ignore[no-untyped-def]
            outputs = self.model(features)
            return tuple(outputs[name] for name in self.names)

    return _Wrapper().eval()


def _export_fp32(config: dict, checkpoint: str, output: str, opset_version: int) -> None:
    """Xuất acoustic ONNX fp32, kèm metadata; danh sách output đổi theo `use_phonation`."""
    import torch

    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.models.vifasttts import ViFastTtsModel

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda), len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    model = ViFastTtsModel(config, sizes)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval()
    dummy = torch.zeros((1, 8, 7), dtype=torch.long)
    output_names = [
        "mel", "log_duration", "durations", "f0",
        "vuv_logits", "energy", "tone_logits", "lengths",
    ]
    if getattr(model, "use_phonation", False):
        # Khớp thứ tự dict output: chèn sau "energy", trước "tone_logits".
        output_names.insert(6, "phonation")
    wrapper = _select_outputs_module(model, tuple(output_names))
    _export_torch(
        wrapper, dummy, output,
        input_names=["features"], output_names=output_names,
        dynamic_axes={"features": {0: "batch", 1: "syllables"}},
        opset_version=opset_version,
    )
    _write_metadata(output, {
        "vifasttts.model": "acoustic",
        "sample_rate": int(config.get("sample_rate", 24000)),
        "n_mels": int(config["mel"]["n_mels"]),
        "use_phonation": bool(getattr(model, "use_phonation", False)),
    })


def _export_v2_fp32(config: dict, checkpoint: str, output: str, opset_version: int) -> None:
    """Xuất acoustic v2 (`arch="v2_slot5"`) fp32: input `features` + `speed`, kèm metadata `arch`.

    Đường v1 (`_export_fp32`) giữ nguyên tuyệt đối — không đổi tên/thứ tự output hay metadata v1.
    """
    import torch

    from vifasttts.frontend.vocab import ComponentVocabs
    from vifasttts.models.factory import ARCH_V2, build_model, load_model_state

    vocabs = ComponentVocabs.default()
    sizes = [len(vocabs.onset), len(vocabs.medial), len(vocabs.nucleus), len(vocabs.coda), len(vocabs.tone), len(vocabs.boundary), len(vocabs.punctuation)]
    model = build_model(config, sizes, arch=ARCH_V2)
    ckpt = torch.load(checkpoint, map_location="cpu")
    load_model_state(model, ckpt.get("model", ckpt))
    f0_stats = ckpt.get("f0_stats")
    if isinstance(f0_stats, dict):
        model.load_f0_stats(f0_stats)
    model.eval()

    output_names = [
        "mel", "log_duration", "durations", "f0",
        "vuv_logits", "energy", "tone_logits", "lengths",
        "slot_durations", "pause_gate",
    ]
    if getattr(model, "use_phonation", False):
        # Khớp thứ tự dict output v1: chèn sau "energy", trước "tone_logits".
        output_names.insert(6, "phonation")

    class _V2ExportWrapper(torch.nn.Module):
        """Bọc v2 để export đúng thứ tự output (model là submodule — closure bake tham số thành constant)."""

        def __init__(self, module) -> None:
            """Nhận model v2 đã nạp trọng số + eval."""
            super().__init__()
            self.model = module

        def forward(self, features: torch.Tensor, speed: torch.Tensor,
                    gate_threshold: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """features + speed + gate_threshold (float32 [1]) → tuple output theo `output_names`."""
            outputs = self.model(features, speed=speed, gate_threshold=gate_threshold)
            return tuple(outputs[name] for name in output_names)

    wrapper = _V2ExportWrapper(model).eval()
    dummy_features = torch.zeros((1, 8, 7), dtype=torch.long)
    dummy_speed = torch.tensor([1.0], dtype=torch.float32)
    dummy_gate = torch.tensor([0.5], dtype=torch.float32)  # V11.A7: ngưỡng gate = input CUỐI
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():  # bắt buộc: v2 slice/index Parameter trong forward (baked thành constant)
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            torch.onnx.export(
                wrapper, (dummy_features, dummy_speed, dummy_gate), output,
                input_names=["features", "speed", "gate_threshold"], output_names=output_names,
                dynamic_axes={"features": {0: "batch", 1: "syllables"}},
                opset_version=opset_version, dynamo=False,
            )
        else:  # torch cũ không có tham số dynamo
            torch.onnx.export(
                wrapper, (dummy_features, dummy_speed, dummy_gate), output,
                input_names=["features", "speed", "gate_threshold"], output_names=output_names,
                dynamic_axes={"features": {0: "batch", 1: "syllables"}},
                opset_version=opset_version,
            )
    props: dict[str, object] = {
        "vifasttts.model": "acoustic",
        "sample_rate": int(config.get("sample_rate", 24000)),
        "n_mels": int(config["mel"]["n_mels"]),
        "use_phonation": bool(getattr(model, "use_phonation", False)),
        "arch": ARCH_V2,
    }
    for key in ("seg_method", "seg_method_version", "seg_weights_hash", "f0_stats_hash"):
        if key in ckpt:
            props[key] = ckpt[key]
    if isinstance(f0_stats, dict):
        for key in ("f0_log_mean", "f0_log_std"):
            if key in f0_stats:
                props[key] = float(f0_stats[key])
    for key in ("f0_log_mean", "f0_log_std"):  # checkpoint cũng có thể ghi trực tiếp ở top-level
        if key in ckpt:
            props[key] = float(ckpt[key])
    _write_metadata(output, props)


def _build_vocoder(config: dict, checkpoint: str):
    """Dựng generator theo `vocoder_training` (lite/vocos/vocos_pretrained) và nạp trọng số.

    Nhận checkpoint `train-vocoder` (khóa `vocoder`) hoặc file HF thô `pytorch_model.bin`.
    """
    from vifasttts.models.istftnet import build_generator, load_generator_state

    mel_config = config["mel"]
    vt = config.get("vocoder_training", {})
    vocoder = build_generator(vt, mel_config)
    load_generator_state(vocoder, checkpoint)
    vocoder.eval()
    return vocoder, vt


def convert_fp16(source: str | Path, target: str | Path) -> None:
    """Chuyển ONNX fp32 sang fp16 (giữ kiểu input/output)."""
    try:
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError("Cần `onnx` và `onnxruntime` (có transformers.float16) để xuất fp16") from exc
    model = onnx.load(str(source))
    model16 = convert_float_to_float16(model, keep_io_types=True)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model16, str(target))


def quantize_int8(source: str | Path, target: str | Path) -> None:
    """Quantize động weight INT8 (onnxruntime)."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError("Cần `onnxruntime` để quantize INT8") from exc
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(source), str(target), weight_type=QuantType.QUInt8)


def export_onnx(
    config: dict,
    checkpoint: str,
    output: str,
    *,
    opset_version: int = 18,
    precision: str = "fp32",
) -> None:
    """Xuất ONNX acoustic với tuỳ chọn fp32 (mặc định), fp16 hoặc int8.

    Bước chuyển đổi ghi ra file đích từ file trung gian, nên `output` luôn
    là bản hoàn chỉnh; file trung gian bị xoá sau khi thành công.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision không hợp lệ: {precision!r} (dùng {', '.join(PRECISIONS)})")
    validate_config(config, "acoustic")
    import torch

    from vifasttts.models.factory import ARCH_V2, arch_of_checkpoint

    ckpt_peek = torch.load(checkpoint, map_location="cpu")
    if arch_of_checkpoint(ckpt_peek if isinstance(ckpt_peek, dict) else {}) == ARCH_V2:
        if precision != "fp32":
            raise NotImplementedError(
                f"precision={precision!r} cho arch v2_slot5 chưa hỗ trợ (mới chỉ có fp32)"
            )
        _export_v2_fp32(config, checkpoint, output, opset_version)
        return
    target = Path(output)
    if precision == "fp32":
        _export_fp32(config, checkpoint, output, opset_version)
        return

    intermediate = target.with_name(target.stem + ".fp32.onnx")
    _export_fp32(config, checkpoint, str(intermediate), opset_version)
    try:
        if precision == "fp16":
            convert_fp16(intermediate, target)
        else:
            quantize_int8(intermediate, target)
    finally:
        intermediate.unlink(missing_ok=True)


def export_vocoder_onnx(config: dict, checkpoint: str, output: str, *, opset_version: int = 18,
                        precision: str = "fp32") -> None:
    """Xuất vocoder ONNX: mel [B,T,n_mels] → spec_real/spec_imag [B,bins,T*ratio].

    Kèm metadata (sample_rate, n_fft, hop_length, mel_hop_length, upsample_ratio) để
    `vifasttts synthesize-onnx` ghép waveform bằng iSTFT numpy. `hop_length` ghi vào
    metadata là hop iSTFT **của graph** (hybrid = hop_mel/ratio, Vocos = hop_mel).

    `precision="fp16"`: giữ nguyên phép kẹp `exp(mag) ≤ 1e2` trong graph (bắt buộc để
    không tràn số fp16 — Vocos gốc cũng kẹp); nên đối chiếu fp16 với fp32 trước khi dùng
    (xem `tests/test_vocos_generator.py::test_vocoder_fp16_matches_fp32`).
    """
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"precision không hợp lệ cho vocoder: {precision!r} (dùng fp32|fp16)")
    validate_config(config, "vocoder")
    vocoder, vt = _build_vocoder(config, checkpoint)

    target = Path(output)
    if precision == "fp16":
        intermediate = target.with_name(target.stem + ".fp32.onnx")
        _export_vocoder_fp32(config, vt, vocoder, str(intermediate), opset_version)
        try:
            convert_fp16(intermediate, target)
        finally:
            intermediate.unlink(missing_ok=True)
        return
    _export_vocoder_fp32(config, vt, vocoder, output, opset_version)


def _export_vocoder_fp32(config: dict, vt: dict, vocoder, output: str, opset_version: int) -> None:
    """Xuất graph vocoder fp32 (dùng chung cho fp32/fp16) + ghi metadata."""
    import torch

    class _VocoderSpectrogramWrapper(torch.nn.Module):
        """mel -> (spec_real, spec_imag); iSTFT ghép ở runtime bằng numpy."""

        def __init__(self, module) -> None:
            """Nhận generator để gọi `spectrogram_parts` (không tạo tensor complex)."""
            super().__init__()
            self.vocoder = module

        def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """mel [B, T, M] → (real, imag) [B, bins, T*ratio] cho runtime iSTFT."""
            real, imag = self.vocoder.spectrogram_parts(mel)
            return real, imag

    module = _VocoderSpectrogramWrapper(vocoder)
    module.eval()
    dummy = torch.zeros((1, 8, int(config["mel"]["n_mels"])))
    _export_torch(
        module, dummy, output,
        input_names=["mel"], output_names=["spec_real", "spec_imag"],
        dynamic_axes={
            "mel": {0: "batch", 1: "frames"},
            "spec_real": {0: "batch", 2: "spec_frames"},
            "spec_imag": {0: "batch", 2: "spec_frames"},
        },
        opset_version=opset_version,
    )
    _write_metadata(output, {
        "vifasttts.model": "vocoder",
        "sample_rate": int(config.get("sample_rate", 24000)),
        "n_fft": int(config["mel"]["n_fft"]),
        # Hop iSTFT THẬT của graph: hybrid = hop_mel // upsample_ratio; Vocos = hop_mel.
        "hop_length": int(vocoder.hop_length),
        "mel_hop_length": int(config["mel"]["hop_length"]),
        "n_mels": int(config["mel"]["n_mels"]),
        "upsample_ratio": int(getattr(vocoder, "upsample_ratio", 1)),
        "generator": str(vt.get("generator", "lite")),
    })

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


def _export_fp32(config: dict, checkpoint: str, output: str, opset_version: int) -> None:
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
    _export_torch(
        model, dummy, output,
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


def _build_vocoder(config: dict, checkpoint: str):
    import torch

    from vifasttts.models.istftnet import ISTFTNetLite

    mel_config = config["mel"]
    vt = config.get("vocoder_training", {})
    vocoder = ISTFTNetLite(
        n_mels=int(mel_config["n_mels"]),
        hidden=int(vt.get("hidden", 256)),
        n_fft=int(mel_config["n_fft"]),
        hop_length=int(mel_config["hop_length"]),
        upsample_ratio=int(vt.get("upsample_ratio", 4)),
    )
    state = torch.load(checkpoint, map_location="cpu")
    vocoder.load_state_dict(state.get("vocoder", state))
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


def export_vocoder_onnx(config: dict, checkpoint: str, output: str, *, opset_version: int = 18) -> None:
    """Xuất vocoder ONNX: mel [B,T,n_mels] → spec_real/spec_imag [B,bins,T*ratio].

    Kèm metadata (sample_rate, n_fft, hop_length, upsample_ratio) để
    `vifasttts synthesize-onnx` ghép waveform bằng iSTFT numpy.
    """
    import torch

    from vifasttts.models.istftnet import ISTFTNetLite

    validate_config(config, "vocoder")
    vocoder, vt = _build_vocoder(config, checkpoint)

    class _VocoderSpectrogramWrapper(torch.nn.Module):
        """mel -> (spec_real, spec_imag); iSTFT ghép ở runtime bằng numpy."""

        def __init__(self, module: ISTFTNetLite) -> None:
            super().__init__()
            self.vocoder = module

        def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        "hop_length": int(config["mel"]["hop_length"]),
        "n_mels": int(config["mel"]["n_mels"]),
        "upsample_ratio": int(vt.get("upsample_ratio", 4)),
    })

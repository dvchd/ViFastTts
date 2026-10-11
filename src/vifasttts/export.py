from __future__ import annotations

import inspect
from pathlib import Path

PRECISIONS = ("fp32", "fp16", "int8")


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
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    input_names = ["features"]
    output_names = [
        "mel", "log_duration", "durations", "f0",
        "vuv_logits", "energy", "tone_logits", "lengths",
    ]
    dynamic_axes = {"features": {0: "batch", 1: "syllables"}}
    # Exporter TorchScript ổn định và không cần onnxscript; chỉ truyền tham số
    # `dynamo` khi torch hỗ trợ để tương thích torch >= 2.3.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        torch.onnx.export(
            model, (dummy,), output,
            input_names=input_names, output_names=output_names,
            opset_version=opset_version, dynamic_axes=dynamic_axes, dynamo=False,
        )
    else:
        torch.onnx.export(
            model, (dummy,), output,
            input_names=input_names, output_names=output_names,
            opset_version=opset_version, dynamic_axes=dynamic_axes,
        )


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
    """Xuất ONNX với tuỳ chọn fp32 (mặc định), fp16 hoặc int8.

    Bước chuyển đổi ghi ra file đích từ file trung gian, nên `output` luôn
    là bản hoàn chỉnh; file trung gian bị xoá sau khi thành công.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"precision không hợp lệ: {precision!r} (dùng {', '.join(PRECISIONS)})")
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

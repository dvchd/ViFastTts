from __future__ import annotations

import inspect
from pathlib import Path


def export_onnx(config: dict, checkpoint: str, output: str, *, opset_version: int = 18) -> None:
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

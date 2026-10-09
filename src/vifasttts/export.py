from __future__ import annotations

from pathlib import Path


def export_onnx(config: dict, checkpoint: str, output: str) -> None:
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
    torch.onnx.export(model, (dummy,), output, input_names=["features"], output_names=["outputs"], opset_version=18, dynamic_axes={"features": {0: "batch", 1: "syllables"}})

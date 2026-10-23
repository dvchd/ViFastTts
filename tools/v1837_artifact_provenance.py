"""v18.37 — kiểm provenance artifact bằng tensor, không dựa vào tên file/tài liệu.

Đọc initializer ONNX và so **từng tensor** với các nguồn khả dĩ:
- acoustic: P2 EMA (`_ckpt/step_02000.pt`), P2 live (`ckpt_2000.pt["model"]`),
  G1 live/EMA (`runs/v1830_g1/ckpt_2000.pt`);
- vocoder: Vocos pretrained (`artifacts/pretrained/.../pytorch_model.bin`) và
  E4-C (`runs/vocoder_vocos_stftc/last.pt["vocoder"]`).

Ghi `artifacts/bench/v1837_provenance.json`. Không sửa/không ghi đè artifact.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from onnx import numpy_helper


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def onnx_initializers(path: str | Path) -> dict[str, np.ndarray]:
    import onnx

    model = onnx.load(str(path))
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def compare(init: dict[str, np.ndarray], state: dict[str, torch.Tensor], *,
            strip: str = "", add_prefix: str = "") -> dict:
    """So các tensor trùng tên; trả exact/total/maxdiff + worst key."""
    exact = total = 0
    missing: list[str] = []
    max_diff = 0.0
    worst = None
    for name, array in init.items():
        key = name[len(strip):] if strip and name.startswith(strip) else name
        key = add_prefix + key
        tensor = state.get(key)
        if tensor is None:
            missing.append(key)
            continue
        value = tensor.detach().cpu().numpy()
        if value.shape != array.shape:
            missing.append(key)
            continue
        total += 1
        if np.array_equal(value, array):
            exact += 1
        diff = float(np.max(np.abs(value.astype(np.float64) - array.astype(np.float64))))
        if diff > max_diff:
            max_diff = diff
            worst = key
    return {
        "exact": exact, "overlap": total, "missing": len(missing),
        "max_abs_diff": max_diff, "worst_key": worst,
        "missing_examples": missing[:5],
    }


def load_state(path: str | Path, key: str | None = None) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if key is not None:
        payload = payload[key]
    return payload


def pretrained_vocos_state(path: str | Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    state = {}
    for key, value in payload.items():
        if key.startswith("feature_extractor."):
            continue
        if key == "head.istft.window":
            key = "head.window"
        state[key] = value
    return state


def main() -> None:
    import yaml

    from vifasttts.data.features import canonical_mel_config, mel_config_hash
    from vifasttts.runtime.onnx_engine import read_metadata

    cfg = yaml.safe_load(Path("configs/small_codelung_v32.yaml").read_text(encoding="utf-8"))
    mel_hash = mel_config_hash(canonical_mel_config(cfg))
    p2_path = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
    p2_live_path = "runs/adv_p2_20k/ckpt_2000.pt"
    g1_path = "runs/v1830_g1/ckpt_2000.pt"
    voc_pretrained = "artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin"
    voc_e4c = "runs/vocoder_vocos_stftc/last.pt"

    p2 = load_state(p2_path, "model")
    p2_live = load_state(p2_live_path, "model")
    g1 = load_state(g1_path)
    acoustic_sources = {
        "p2_ema": p2,
        "p2_live": p2_live,
        "g1_live": g1["model"],
        "g1_ema": g1["ema"],
    }
    acoustic_onnx = "artifacts/onnx/v32_adv_p2_2k.onnx"
    acoustic_init = onnx_initializers(acoustic_onnx)
    acoustic_compare = {
        name: compare(acoustic_init, state, strip="model.")
        for name, state in acoustic_sources.items()
    }
    acoustic_best = max(acoustic_compare, key=lambda key: (
        acoustic_compare[key]["exact"], -acoustic_compare[key]["max_abs_diff"]))

    voc_init = onnx_initializers("artifacts/onnx/v32_adv_p2_2k_vocoder.onnx")
    voc_sources = {
        "vocos_pretrained": pretrained_vocos_state(voc_pretrained),
        "e4c_stftc": load_state(voc_e4c, "vocoder"),
    }
    voc_compare = {
        name: compare(voc_init, state, strip="vocoder.", add_prefix="")
        for name, state in voc_sources.items()
    }
    voc_best = max(voc_compare, key=lambda key: (
        voc_compare[key]["exact"], -voc_compare[key]["max_abs_diff"]))

    result = {
        "production": {
            "label": "P2 EMA — không đổi trong v18.37",
            "acoustic_config": "configs/small_codelung_v32.yaml",
            "acoustic_checkpoint": p2_path,
            "acoustic_sha256": sha256(p2_path),
            "acoustic_weights": "ema (model của _ckpt == ema của ckpt_2000)",
            "acoustic_source": "runs/adv_p2_20k/ckpt_2000.pt:ema",
            "mel_hash_checkpoint": torch.load(p2_path, map_location="cpu", weights_only=False).get("mel_hash"),
            "mel_hash_current": mel_hash,
            "mel_hash_match": torch.load(p2_path, map_location="cpu", weights_only=False).get("mel_hash") == mel_hash,
            "vocoder": voc_pretrained,
            "vocoder_sha256": sha256(voc_pretrained),
            "vocoder_source": "charactr/vocos-mel-24khz (pretrained zero-shot)",
            "vocoder_sha256_config": "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a",
        },
        "onnx": {
            "acoustic": {
                "path": acoustic_onnx,
                "sha256": sha256(acoustic_onnx),
                "metadata": read_metadata(acoustic_onnx),
                "tensor_match": acoustic_compare,
                "best_source": acoustic_best,
                "status": ("khớp P2 EMA" if acoustic_best == "p2_ema" else "KHÔNG khớp P2 EMA"),
            },
            "vocoder": {
                "path": "artifacts/onnx/v32_adv_p2_2k_vocoder.onnx",
                "sha256": sha256("artifacts/onnx/v32_adv_p2_2k_vocoder.onnx"),
                "metadata": read_metadata("artifacts/onnx/v32_adv_p2_2k_vocoder.onnx"),
                "tensor_match": voc_compare,
                "best_source": voc_best,
                "status": ("khớp Vocos pretrained" if voc_best == "vocos_pretrained"
                           else "MISMATCH: khớp E4-C, không phải pretrained"),
            },
        },
        "new_exports": {},
    }
    new_dir = Path("artifacts/onnx/v1837")
    new_specs = {
        "acoustic": (new_dir / "p2ema_acoustic.onnx", p2, "model."),
        "vocoder": (new_dir / "vocos_pretrained_vocoder.onnx",
                    pretrained_vocos_state(voc_pretrained), "vocoder."),
    }
    for name, (path, state, strip) in new_specs.items():
        if path.exists():
            init = onnx_initializers(path)
            result["new_exports"][name] = {
                "path": str(path),
                "sha256": sha256(path),
                "metadata": read_metadata(path),
                "tensor_match": compare(init, state, strip=strip),
            }

    out = Path("artifacts/bench/v1837_provenance.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

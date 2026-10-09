from __future__ import annotations

import json
import random
from pathlib import Path
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_reference(config: dict, train_manifest: str, val_manifest: str, output: str) -> None:
    """Reference training entrypoint.

    Feature extraction and alignment targets are dataset-specific. This function
    validates manifests and creates a reproducible run directory. Use the
    documented stages to add cached mel, F0, energy and duration tensors before
    starting GPU training.
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(int(config.get("seed", 42)))
    for path in [train_manifest, val_manifest]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    (out / "resolved_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "NEXT_STEPS.txt").write_text(
        "Manifest đã được xác minh. Hãy chạy pipeline feature extraction để tạo mel, F0, energy và duration target trước khi bật full optimizer loop.\n",
        encoding="utf-8",
    )
    print(f"Đã khởi tạo run tại {out}")

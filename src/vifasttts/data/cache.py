"""Cache feature/duration theo mẫu — tăng tốc training quy mô lớn.

Mel/energy/f0/vuv là tất định theo (audio, config), nên lần trước tính xong
được lưu `.pt` khoá theo hash(audio path + size + mtime + mel config). Lần
sau đọc thẳng, bỏ qua decode/STFT. Xem docs/DATA_PIPELINE.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch

from vifasttts.data.features import extract_features


def feature_cache_key(row: dict[str, Any], config: dict[str, Any]) -> str:
    """Hash ổn định cho (audio thật, cấu hình trích feature)."""
    audio = Path(str(row.get("audio_path", "")))
    try:
        stat = audio.stat()
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        stamp = "missing"
    mel = config.get("mel", {})
    payload = json.dumps(
        {
            "audio": str(audio),
            "stamp": stamp,
            "sample_rate": int(config.get("sample_rate", 24000)),
            "mel": {str(key): mel[key] for key in sorted(mel)},
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class FeatureCache:
    """Thư mục cache feature dạng `<root>/<sha1>.pt`."""

    def __init__(self, root: str | Path, config: dict[str, Any]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config

    def path_for(self, row: dict[str, Any]) -> Path:
        return self.root / f"{feature_cache_key(row, self.config)}.pt"

    def get(self, row: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        path = self.path_for(row)
        if not path.exists():
            return None
        return cast(dict[str, torch.Tensor], torch.load(path, map_location="cpu", weights_only=False))

    def put(self, row: dict[str, Any], features: dict[str, torch.Tensor]) -> None:
        torch.save(features, self.path_for(row))


def precompute_features(
    manifest: str | Path,
    config: dict[str, Any],
    cache_dir: str | Path,
    *,
    limit: int | None = None,
) -> int:
    """Tính trước feature cho toàn bộ manifest; trả về số mẫu đã xử lý."""
    from vifasttts.data.dataset import ManifestDataset

    rows = ManifestDataset(manifest).rows
    if limit is not None:
        rows = rows[:limit]
    cache = FeatureCache(cache_dir, config)
    done = 0
    for row in rows:
        if cache.get(row) is None:
            cache.put(row, extract_features(row["audio_path"], config))
        done += 1
    return done

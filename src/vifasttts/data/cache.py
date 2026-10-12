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
            "phonation": config.get("phonation", {}),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class FeatureCache:
    """Thư mục cache feature dạng `<root>/<sha1>.pt`."""

    def __init__(self, root: str | Path, config: dict[str, Any]) -> None:
        """Mở/tạo thư mục cache; `config` dùng để tính khoá (mel, sample_rate, phonation)."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config

    def path_for(self, row: dict[str, Any]) -> Path:
        """Đường dẫn file `.pt` cho một mẫu (khoá theo audio thật + config)."""
        return self.root / f"{feature_cache_key(row, self.config)}.pt"

    def get(self, row: dict[str, Any]) -> dict[str, torch.Tensor] | None:
        """Đọc feature đã cache (None nếu chưa có — caller tự trích)."""
        path = self.path_for(row)
        if not path.exists():
            return None
        return cast(dict[str, torch.Tensor], torch.load(path, map_location="cpu", weights_only=False))

    def put(self, row: dict[str, Any], features: dict[str, torch.Tensor]) -> None:
        """Lưu feature của một mẫu vào cache."""
        torch.save(features, self.path_for(row))


def _extract_one(payload: tuple[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Worker (module-level) cho ProcessPoolExecutor: trích feature một file."""
    audio_path, config = payload
    return extract_features(audio_path, config)


def precompute_features(
    manifest: str | Path,
    config: dict[str, Any],
    cache_dir: str | Path,
    *,
    limit: int | None = None,
    workers: int = 0,
) -> int:
    """Tính trước feature cho toàn bộ manifest; trả về số mẫu đã xử lý.

    `workers > 1` dùng ProcessPoolExecutor (mỗi tiến trình một file) — nhanh hơn
    nhiều cho dataset hàng chục giờ; mặc định 0 = tuần tự (an toàn cho test).
    """
    from vifasttts.data.dataset import ManifestDataset

    rows = ManifestDataset(manifest).rows
    if limit is not None:
        rows = rows[:limit]
    cache = FeatureCache(cache_dir, config)
    # Chỉ kiểm tra sự tồn tại (path), KHÔNG load file — tránh đọc lại hàng GB cache.
    pending = [row for row in rows if not cache.path_for(row).exists()]
    done = len(rows) - len(pending)
    if workers and workers > 1 and len(pending) > 1:
        from concurrent.futures import ProcessPoolExecutor

        payloads = [(str(row["audio_path"]), config) for row in pending]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row, features in zip(pending, pool.map(_extract_one, payloads), strict=True):
                cache.put(row, features)
                done += 1
    else:
        for row in pending:
            cache.put(row, extract_features(row["audio_path"], config))
            done += 1
    return done

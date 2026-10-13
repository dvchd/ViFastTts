"""Factory model acoustic: chọn `arch` + warm-start tường minh v1 → v2 (docs/ACOUSTIC_V2.md §10–§11).

- `arch` nằm trong **checkpoint** (`"v1_syl"` mặc định khi thiếu; `"v2_slot5"` cho v2);
  factory đọc và dựng đúng lớp — **không** load chéo bằng `strict=False` ngầm.
- `warm_start_v2_from_v1`: chỉ copy **subset key tường minh** (§11: embedding, encoder, decoder,
  mel, postnet, tone_classifier, duration); module mới của v2 giữ khởi tạo riêng.
"""

from __future__ import annotations

from torch import nn

from vifasttts.models.v2_slot5 import ViFastTtsModelV2
from vifasttts.models.vifasttts import ViFastTtsModel

ARCH_V1 = "v1_syl"
ARCH_V2 = "v2_slot5"
ARCHES = (ARCH_V1, ARCH_V2)

# Key được warm-start từ B0/v1 sang v2 (§11) — khớp theo tiền tố.
WARM_START_PREFIXES = ("embedding.", "encoder.", "decoder.", "mel.", "postnet.",
                       "tone_classifier.", "duration.")


def arch_of_checkpoint(checkpoint: dict) -> str:
    """`arch` trong checkpoint; thiếu → `v1_syl` (tương thích ngược toàn bộ checkpoint cũ)."""
    return str(checkpoint.get("arch") or ARCH_V1)


def build_model(config: dict, vocab_sizes: list[int], arch: str | None = None) -> nn.Module:
    """Dựng model theo `arch` (ưu tiên tham số, rồi `config.model.arch`, mặc định `v1_syl`)."""
    chosen = str(arch or config.get("model", {}).get("arch") or ARCH_V1)
    if chosen == ARCH_V1:
        return ViFastTtsModel(config, vocab_sizes)
    if chosen == ARCH_V2:
        return ViFastTtsModelV2(config, vocab_sizes)
    raise ValueError(f"arch không hợp lệ: {chosen!r} (dùng {', '.join(ARCHES)})")


def warm_start_v2_from_v1(model_v2: nn.Module, checkpoint_v1: dict) -> dict[str, list[str]]:
    """Copy trọng số v1 → v2 theo danh sách tường minh; trả `{"loaded", "skipped"}`.

    Raise `ValueError` khi shape lệch (bắt lỗi cấu hình sớm thay vì mismatch im lặng).
    """
    source = checkpoint_v1.get("model", checkpoint_v1)
    target = model_v2.state_dict()
    subset: dict = {}
    loaded: list[str] = []
    skipped: list[str] = []
    for key, value in source.items():
        if not key.startswith(WARM_START_PREFIXES):
            continue
        if key not in target:
            skipped.append(key)
            continue
        if tuple(target[key].shape) != tuple(value.shape):
            raise ValueError(f"shape lệch cho {key}: checkpoint {tuple(value.shape)} "
                             f"vs model {tuple(target[key].shape)}")
        subset[key] = value
        loaded.append(key)
    model_v2.load_state_dict(subset, strict=False)
    return {"loaded": sorted(loaded), "skipped": sorted(skipped)}

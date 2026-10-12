"""Parse lại text sau khi mở rộng overrides — KHÔNG cần recut audio.

Chỉ đổi text (metadata.csv + manifest.jsonl) và chia lại split; audio/cache giữ nguyên
(recut B3 đã xong). Xem `docs/REVISION_PLAN_V2.md` rev.5 §9.3.

Chạy:
    python tools/reparse_overrides.py --dataset data/codelung \
        --overrides configs/overrides_tech_vi.json --val-videos 4 --seed 13
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from vifasttts.data.manifest import build_manifest, coverage_report
from vifasttts.data.split import write_split
from vifasttts.data.youtube import _prepare_dataset_text
from vifasttts.frontend.dialect import Dialect
from vifasttts.frontend.normalizer import NormalizerOptions


def load_overrides(path: str | Path) -> dict[str, str]:
    """Đọc JSON 'mặt chữ → cách đọc'."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        raise ValueError("Overrides phải là JSON dạng {mặt chữ: cách đọc}")
    return dict(payload)


def reparse_dataset(dataset: str | Path, overrides: dict[str, str], *, dialect: str = "north",
                    respell_codes: bool = True, val_videos: int | None = 4,
                    seed: int = 13) -> dict[str, object]:
    """Áp overrides lên metadata.csv → manifest + split; trả thống kê trước/sau."""
    root = Path(dataset)
    metadata = root / "metadata.csv"
    if not metadata.exists():
        raise FileNotFoundError(f"Thiếu {metadata} (cần cho parse lại)")
    options = NormalizerOptions(
        dialect=Dialect(dialect),
        exact_overrides=dict(overrides),
        english_overrides={key.lower(): value for key, value in overrides.items()},
    )

    before = {"fallback": 0, "syllables": 0}
    manifest_path = root / "manifest.jsonl"
    if manifest_path.exists():
        report = coverage_report(manifest_path)
        before = {"fallback": report["fallback"], "syllables": report["syllables"]}

    # Ghi bản gốc để đối chiếu (chỉ lần đầu).
    backup = root / "metadata.pre_reparse.csv"
    if not backup.exists():
        shutil.copy2(metadata, backup)

    lines = []
    changed = 0
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample_id, text = line.split("|", 1)
        new_text = _prepare_dataset_text(text, options, respell_codes)
        if new_text != text:
            changed += 1
        lines.append(f"{sample_id}|{new_text}")
    metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")

    build_manifest(str(metadata), str(root / "audio"), str(manifest_path))
    stats = write_split(str(manifest_path), str(root), val_videos=val_videos, val_ratio=0.05, seed=seed)
    after = coverage_report(manifest_path)
    return {
        "changed_texts": changed,
        "fallback_before": before["fallback"],
        "fallback_after": after["fallback"],
        "syllables_before": before["syllables"],
        "syllables_after": after["syllables"],
        "train": stats["train"],
        "val": stats["val"],
        "val_videos": stats["val_videos"],
    }


def main() -> None:
    """Đọc tham số, parse lại, in thống kê trước/sau."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Thư mục dataset (có metadata.csv + audio/)")
    parser.add_argument("--overrides", required=True, help="File JSON overrides mới")
    parser.add_argument("--dialect", default="north", help="north | central | south (khớp lúc recut)")
    parser.add_argument("--no-respell-codes", action="store_true", help="Tắt respell mã chữ-số")
    parser.add_argument("--val-videos", type=int, default=4, help="Số video val (khớp split gốc)")
    parser.add_argument("--seed", type=int, default=13, help="Seed chia split (khớp split gốc)")
    args = parser.parse_args()

    stats = reparse_dataset(
        args.dataset, load_overrides(args.overrides),
        dialect=args.dialect, respell_codes=not args.no_respell_codes,
        val_videos=args.val_videos, seed=args.seed,
    )
    print(f"Đổi {stats['changed_texts']} dòng text · fallback {stats['fallback_before']} → "
          f"{stats['fallback_after']} ({stats['syllables_before']} → {stats['syllables_after']} âm tiết)")
    print(f"Train {stats['train']} · Val {stats['val']} · video val: {', '.join(stats['val_videos'])}")


if __name__ == "__main__":
    main()

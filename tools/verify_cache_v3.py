"""Kiểm tra cache feature v3 (có phonation) sau khi build xong.

Mỗi row manifest được khoá bằng `FeatureCache(cache_dir, config).path_for(row)`;
tool đếm file thiếu, kiểm tra cấu trúc từng entry (đủ kênh, cùng độ dài,
không NaN/Inf, miền giá trị jitter/glottalization) và tuỳ chọn trích lại
một số file bằng `extract_features` để so với cache.

Chạy:
    .venv/bin/python tools/verify_cache_v3.py \
        --manifest data/codelung/train_clean_v2_dur_seg.jsonl \
        --manifest data/codelung/val_clean_v2_dur_seg.jsonl \
        --config configs/small_codelung_v2.yaml --cache-dir cache/codelung_v2 \
        --report artifacts/cache_v3_verify.json --sample 0 --recompute 20 --workers 8

Exit 0 khi mọi file đều có + không lỗi cấu trúc + recompute không mismatch,
ngược lại exit 1 (in rõ lý do). Không ghi đè cache, chỉ đọc.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import torch

from vifasttts.data.cache import FeatureCache

REQUIRED: tuple[str, ...] = ("mel", "energy", "f0", "vuv", "hnr", "glottalization", "jitter")


def verify_entry(
    features: dict[str, Any],
    required: tuple[str, ...] = REQUIRED,
    *,
    tolerance: float = 1e-4,
) -> list[str]:
    """Kiểm tra thuần một entry cache; trả danh sách lỗi (rỗng = đạt)."""
    errors: list[str] = []
    for key in required:
        if key not in features:
            errors.append(f"thieu_khoa:{key}")
    if "mel" not in features or not torch.is_tensor(features["mel"]):
        return errors  # không có mốc độ dài để kiểm tiếp
    frames = int(features["mel"].size(0))
    for key, values in features.items():
        if not torch.is_tensor(values):
            errors.append(f"sai_kieu:{key}")
            continue
        if values.dim() >= 1 and int(values.size(0)) != frames:
            errors.append(f"lech_do_dai:{key}:{int(values.size(0))}!={frames}")
        flat = values.reshape(-1) if values.numel() else values.reshape(0)
        if flat.numel() and bool(torch.isnan(flat).any()):
            errors.append(f"nan:{key}")
        if flat.numel() and bool(torch.isinf(flat).any()):
            errors.append(f"inf:{key}")
    jitter = features.get("jitter")
    if torch.is_tensor(jitter) and jitter.numel() and bool((jitter < 0).any()):
        errors.append("mien_gia_tri:jitter<0")
    glot = features.get("glottalization")
    if (
        torch.is_tensor(glot)
        and glot.numel()
        and bool(((glot < -tolerance) | (glot > 1 + tolerance)).any())
    ):
        errors.append("mien_gia_tri:glottalization-ngoai-[0,1]")
    return errors


def _pick_rows(rows: list[dict[str, Any]], sample: int, seed: int) -> list[dict[str, Any]]:
    """Lấy toàn bộ row, hoặc mọi row thuộc N file audio phân biệt (seed cố định)."""
    if sample <= 0:
        return rows
    by_audio: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_audio.setdefault(str(row.get("audio_path", "")), []).append(row)
    audios = sorted(by_audio)
    picked = set(random.Random(seed).sample(audios, min(sample, len(audios))))
    return [row for row in rows if str(row.get("audio_path", "")) in picked]


def verify_manifest(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    cache_dir: str | Path,
    sample: int,
    seed: int,
) -> dict[str, Any]:
    """Kiểm tra cấu trúc cache cho các row của một manifest (không recompute)."""
    checked = _pick_rows(rows, sample, seed)
    cache = FeatureCache(cache_dir, config)
    missing_examples: list[str] = []
    issue_counts: Counter[str] = Counter()
    issue_examples: list[str] = []
    missing = 0
    hits = 0
    for row in checked:
        path = cache.path_for(row)
        label = str(row.get("audio_path", "?"))
        if not path.exists():
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(label)
            continue
        hits += 1
        features = torch.load(path, map_location="cpu", weights_only=False)
        for err in verify_entry(features):
            issue_counts[err] += 1
            if len(issue_examples) < 5:
                issue_examples.append(f"{label} :: {err}")
    files = len({str(r.get("audio_path", "")) for r in checked})
    return {
        "rows": len(checked),
        "files": files,
        "hits": hits,
        "missing": missing,
        "missing_examples": missing_examples,
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_examples": issue_examples,
        "recomputed": 0,
        "recompute_mismatch": 0,
    }


def _recompute_one(payload: tuple[str, dict[str, Any], str]) -> tuple[str, list[str]]:
    """Worker trích lại một file audio rồi so với cache (chạy trong tiến trình con)."""
    audio_path, config, cache_path = payload
    from vifasttts.data.features import extract_features

    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    fresh = extract_features(audio_path, config)
    bad: list[str] = []
    for key, old in cached.items():
        new = fresh.get(key)
        if new is None:
            bad.append(f"{key}:thieu_trong_trich_lai")
        elif not torch.is_tensor(new) or tuple(new.shape) != tuple(old.shape):
            bad.append(f"{key}:lech_shape")
        elif not bool(torch.allclose(new.float(), old.float(), atol=1e-5, rtol=1e-4)):
            bad.append(f"{key}:lech_gia_tri")
    return audio_path, bad


def _recompute(
    checked_audios: list[str],
    config: dict[str, Any],
    cache: FeatureCache,
    count: int,
    workers: int,
) -> tuple[int, int]:
    """Trích lại tối đa `count` file phân biệt; trả (số đã trích, số file mismatch)."""
    candidates = [a for a in sorted(set(checked_audios)) if Path(a).exists()]
    picked = random.Random(42).sample(candidates, min(count, len(candidates)))
    payloads = [(a, config, str(cache.path_for({"audio_path": a}))) for a in picked]
    # Bỏ file chưa có cache (đã đếm ở missing) khỏi recompute.
    payloads = [p for p in payloads if Path(p[2]).exists()]
    mismatch = 0
    if payloads:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            for audio_path, bad in pool.map(_recompute_one, payloads):
                if bad:
                    mismatch += 1
                    print(f"  recompute MISMATCH {audio_path}: {bad}")
    return len(payloads), mismatch


def _load_rows(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", action="append", required=True, help="JSONL manifest (lặp nhiều lần)")
    ap.add_argument("--config", required=True, help="YAML config dùng để tính khoá cache")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--report", required=True, help="đường dẫn JSON ghi báo cáo")
    ap.add_argument("--sample", type=int, default=0, help="0 = tất cả row; >0 = N file audio/manifest")
    ap.add_argument("--recompute", type=int, default=0, help="0 = không; >0 = trích lại K file")
    ap.add_argument("--workers", type=int, default=8, help="số tiến trình cho recompute")
    args = ap.parse_args()

    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cache = FeatureCache(args.cache_dir, config)
    manifests: dict[str, Any] = {}
    for manifest in args.manifest:
        rows = _load_rows(manifest)
        stem = Path(manifest).stem
        result = verify_manifest(rows, config, args.cache_dir, args.sample, seed=42)
        if args.recompute > 0:
            checked = _pick_rows(rows, args.sample, seed=42)
            n, bad = _recompute([str(r.get("audio_path", "")) for r in checked], config, cache,
                                args.recompute, args.workers)
            result["recomputed"] = n
            result["recompute_mismatch"] = bad
        manifests[stem] = result
        print(f"{stem}: rows={result['rows']} files={result['files']} hits={result['hits']} "
              f"missing={result['missing']} loi={sum(result['issue_counts'].values())} "
              f"recomputed={result['recomputed']} mismatch={result['recompute_mismatch']}")

    all_present = all(m["missing"] == 0 for m in manifests.values())
    all_consistent = all(
        not m["issue_counts"] and m["recompute_mismatch"] == 0 for m in manifests.values()
    )
    report = {
        "version": "cache-v3-verify",
        "config": str(args.config),
        "cache_dir": str(args.cache_dir),
        "manifests": manifests,
        "summary": {"all_present": all_present, "all_consistent": all_consistent},
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"bao_cao: {out} all_present={all_present} all_consistent={all_consistent}")
    if not all_present:
        print("LY_DO_EXIT_1: còn file cache thiếu (xem missing_examples trong report).")
    if not all_consistent:
        print("LY_DO_EXIT_1: còn lỗi cấu trúc hoặc recompute mismatch (xem issue_examples).")
    return 0 if (all_present and all_consistent) else 1


if __name__ == "__main__":
    raise SystemExit(main())

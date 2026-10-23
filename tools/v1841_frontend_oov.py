"""v18.41 — trace frontend OOV: PyTorch production vs ONNX trước/sau sửa cờ resolve.

Bộ cố định (chọn trước): custom reading, English fallback, tên/chữ hoa OOV,
token không phân giải, số + dấu câu, câu Việt thường, dialect Nam, input rỗng.

Ghi normalized text, token/status, feature IDs, khác biệt cũ/mới, thông báo lỗi.
Với input hợp lệ: smoke ONNX end-to-end (finite, độ dài > 0, không crash).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("VIFASTT_INTRA_THREADS", "2")

from vifasttts.frontend.custom_reading import apply_custom_reading
from vifasttts.frontend.normalizer import normalize_text
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.vocab import ComponentVocabs
from vifasttts.runtime.onnx_engine import (
    OnnxAcousticModel,
    OnnxVocoder,
    synthesize_onnx,
)

ACOUSTIC = "artifacts/onnx/v1837/p2ema_acoustic.onnx"
VOCODER = "artifacts/onnx/v1837/vocos_pretrained_vocoder.onnx"
OUT = Path("artifacts/bench/v1841_frontend_oov")

CASES = [
    {"kind": "custom_reading", "text": "VBA.", "dialect": "north"},
    {"kind": "english_fallback", "text": "OpenAI.", "dialect": "north"},
    {"kind": "oov_name_short", "text": "John.", "dialect": "north"},
    {"kind": "oov_upper", "text": "XYZ.", "dialect": "north"},
    {"kind": "unresolved", "text": "xqzzt.", "dialect": "north"},
    {"kind": "number_punctuation", "text": "Năm 2026, giá 15,5%.", "dialect": "north"},
    {"kind": "control_vi", "text": "Xin chào các bạn.", "dialect": "north"},
    {"kind": "dialect_south", "text": "1500.", "dialect": "south"},
    {"kind": "empty", "text": "...", "dialect": "north"},
]


def encode(items) -> np.ndarray:
    vocabs = ComponentVocabs.default()
    return np.array([[vocabs.encode(item) for item in items]], dtype=np.int64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = object.__new__(OnnxAcousticModel)  # frontend thuần, không cần session
    OnnxAcousticModel(ACOUSTIC, provider="cpu")
    OnnxVocoder(VOCODER, provider="cpu")  # mở session sớm để lỗi cấu hình lộ ngay
    rows = []
    for index, case in enumerate(CASES):
        text, dialect = case["text"], case["dialect"]
        normalized = normalize_text(apply_custom_reading(text), dialect)
        true_items = parse_text(text, dialect=dialect, resolve_oov=True)
        old_items = parse_text(text, dialect=dialect, resolve_oov=False)
        true_ids = encode(true_items)
        old_ids = encode(old_items)
        new_ids = None
        error = None
        smoke = None
        if case["kind"] != "empty":
            new_ids = engine.features_from_text(text, dialect=dialect)  # default phải resolve
            path = OUT / f"{index:02d}.wav"
            try:
                samples, rate = synthesize_onnx(
                    text, ACOUSTIC, path, vocoder_path=VOCODER, provider="cpu",
                    dialect=dialect, target_lufs=None, peak_normalize=False,
                    sample_rate=24000, steady=False,
                )
                wave, _ = sf.read(path, dtype="float32")
                smoke = {
                    "samples": int(samples), "sample_rate": int(rate),
                    "finite": bool(np.isfinite(np.asarray(wave)).all()),
                    "waveform_len": int(np.asarray(wave).size),
                    "ok": bool(samples > 0 and np.isfinite(np.asarray(wave)).all()),
                }
            except (ImportError, RuntimeError, ValueError, OSError) as exc:  # pragma: no cover - lỗi runtime
                error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                engine.features_from_text(text, dialect=dialect, resolve_oov=True)
                error = "KHÔNG có lỗi (bất ngờ)"
            except ValueError as exc:
                error = str(exc)
        rows.append({
            "kind": case["kind"], "text": text, "dialect": dialect,
            "normalized": normalized,
            "pytorch": {
                "items": len(true_items),
                "raw": [item.raw for item in true_items],
                "normalized": [item.normalized for item in true_items],
                "status": [item.status.value for item in true_items],
                "punctuation": [item.punctuation for item in true_items],
                "ids": true_ids.tolist(),
            },
            "onnx_old_resolve_false": {
                "items": len(old_items),
                "status": [item.status.value for item in old_items],
                "ids": old_ids.tolist(),
            },
            "onnx_new_resolve_true": {
                "items": len(true_items), "ids": new_ids.tolist() if new_ids is not None else [],
            },
            "diff_old_vs_pytorch": {
                "same_shape": old_ids.shape == true_ids.shape,
                "equal": bool(np.array_equal(old_ids, true_ids)),
            },
            "diff_new_vs_pytorch": {
                "same_shape": new_ids is not None and new_ids.shape == true_ids.shape,
                "equal": bool(new_ids is not None and np.array_equal(new_ids, true_ids)),
            },
            "error": error, "smoke": smoke,
        })

    valid = [row for row in rows if row["kind"] != "empty"]
    summary = {
        "cases": len(rows),
        "valid_cases": len(valid),
        "old_equal": sum(1 for row in valid if row["diff_old_vs_pytorch"]["equal"]),
        "new_equal": sum(1 for row in valid if row["diff_new_vs_pytorch"]["equal"]),
        "smoke_ok": sum(1 for row in rows if row["smoke"] and row["smoke"]["ok"]),
        "smoke_total": sum(1 for row in rows if row["smoke"] is not None),
        "error_cases": [row["kind"] for row in rows if row["error"]],
        "old_mismatch_cases": [row["kind"] for row in valid
                               if not row["diff_old_vs_pytorch"]["equal"]],
        "new_mismatch_cases": [row["kind"] for row in valid
                               if not row["diff_new_vs_pytorch"]["equal"]],
    }
    result = {
        "source": "PyTorch parse_text(resolve_oov=True) vs ONNX features_from_text",
        "onnx_acoustic": ACOUSTIC, "onnx_vocoder": VOCODER,
        "threads": os.environ.get("VIFASTT_INTRA_THREADS"),
        "cases": rows, "summary": summary,
    }
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

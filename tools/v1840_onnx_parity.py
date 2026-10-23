"""v18.40 — parity artifact ONNX v1837: P2 EMA + Vocos pretrained.

Ba tầng:
A. acoustic: cùng features/speed/gate, so duration/slot/mel/f0/... PyTorch–ONNX.
B. vocoder: cùng mel, so spectrum trước iSTFT + waveform sau iSTFT.
C. end-to-end chunk mode: cùng frontend/chunking/pause/fade, target_lufs=None.

Tiêu chí được khai báo ở `CRITERIA` **trước** khi chạy. Không dùng BAK/UTMOS,
không benchmark tốc độ/RAM, không đổi production.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("VIFASTT_INTRA_THREADS", "2")

from vifasttts.data.split import video_id_of_row
from vifasttts.eval_text import build_eval_text
from vifasttts.frontend.chunking import chunk_text
from vifasttts.frontend.pipeline import parse_text
from vifasttts.infer import Synthesizer
from vifasttts.runtime.onnx_engine import (
    OnnxAcousticModel,
    OnnxVocoder,
    read_metadata,
    synthesize_onnx,
)

CFG32 = "configs/small_codelung_v32.yaml"
P2 = "artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt"
ACOUSTIC = "artifacts/onnx/v1837/p2ema_acoustic.onnx"
VOCODER = "artifacts/onnx/v1837/vocos_pretrained_vocoder.onnx"
OLD_ACOUSTIC = "artifacts/onnx/v32_adv_p2_2k.onnx"
OLD_VOCODER = "artifacts/onnx/v32_adv_p2_2k_vocoder.onnx"
OUT = Path("artifacts/bench/v1840_onnx_parity")

# Khai báo TRƯỚC khi chạy. Đơn vị: mel z-score / log-F0 / waveform [-1,1].
CRITERIA = {
    "acoustic": {
        "discrete_exact": True,
        "mel_max_abs": 2e-3,
        "mel_mean_abs": 2e-4,
        "continuous_max_abs": 2e-3,
    },
    "vocoder": {
        "length_equal": True,
        "spectrum_max_abs": 2e-2,
        "spectrum_mean_abs": 2e-3,
        "waveform_max_abs": 5e-4,
        "waveform_rmse": 5e-5,
    },
    "end_to_end": {
        "length_equal": True,
        "max_abs": 2e-3,
        "rmse": 2e-4,
        "rms_ratio_min": 0.99,
        "rms_ratio_max": 1.01,
        "clipping_extra": 0,
    },
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_stats(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return {"shape_left": list(left.shape), "shape_right": list(right.shape),
                "shape_equal": False, "finite": bool(np.isfinite(left).all() and np.isfinite(right).all())}
    diff = left - right
    return {
        "shape_left": list(left.shape), "shape_right": list(right.shape), "shape_equal": True,
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0,
        "finite": bool(np.isfinite(left).all() and np.isfinite(right).all()),
    }


def exact(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left)
    right = np.asarray(right)
    return {"shape_equal": list(left.shape) == list(right.shape),
            "equal": bool(np.array_equal(left, right)),
            "max_abs": float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
            if left.shape == right.shape and left.size else None}


def select_inputs() -> list[dict]:
    """10 row manifest (4 short/3 mid/3 long) + 1 âm tiết + 1 custom reading."""
    rows = {json.loads(line)["id"]: json.loads(line) for line in
            Path("data/codelung/v32/val_clean_v2_dur_seg_slot4.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()}
    pause = json.loads(Path("artifacts/bench/v1837_pause/selection.json")
                       .read_text(encoding="utf-8"))
    picked = ([item for item in pause if item["bucket"] == "short"] +
              [item for item in pause if item["bucket"] == "mid"][:3] +
              [item for item in pause if item["bucket"] == "long"][:3])
    if len(picked) != 10:
        raise RuntimeError(f"cần 10 row manifest, nhận {len(picked)}")
    items = []
    for entry in picked:
        row = rows[entry["id"]]
        items.append({
            "kind": "manifest", "id": entry["id"], "video": video_id_of_row(row),
            "bucket": entry["bucket"], "text": build_eval_text(row),
        })
    items.append({"kind": "one_syllable", "id": "synthetic_one", "video": "-",
                  "bucket": "one", "text": "Chào."})
    items.append({"kind": "custom_reading", "id": "synthetic_custom", "video": "-",
                  "bucket": "custom", "text": "VBA."})
    return items


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cfg = yaml.safe_load(Path(CFG32).read_text(encoding="utf-8"))
    engine = Synthesizer(cfg, P2, device="cpu")
    engine.model.eval()
    engine.vocoder.eval()
    acoustic = OnnxAcousticModel(ACOUSTIC, provider="cpu")
    vocoder = OnnxVocoder(VOCODER, provider="cpu")
    items = select_inputs()
    (OUT / "inputs.json").write_text(json.dumps(items, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    provenance = json.loads(Path("artifacts/bench/v1837_provenance.json")
                            .read_text(encoding="utf-8"))
    acoustic_meta = read_metadata(ACOUSTIC)
    vocoder_meta = read_metadata(VOCODER)
    provenance_checks = {
        "acoustic_sha256": sha256(ACOUSTIC),
        "vocoder_sha256": sha256(VOCODER),
        "acoustic_sha_matches_v1837": sha256(ACOUSTIC)
        == provenance["new_exports"]["acoustic"]["sha256"],
        "vocoder_sha_matches_v1837": sha256(VOCODER)
        == provenance["new_exports"]["vocoder"]["sha256"],
        "acoustic_mel_hash": acoustic_meta.get("mel_hash"),
        "vocoder_mel_hash": vocoder_meta.get("mel_hash"),
        "mel_hash_expected": "e065776b45b74b874c483dfdc68c1f9c5e45918b",
        "acoustic_checkpoint_sha256": acoustic_meta.get("checkpoint_sha256"),
        "vocoder_checkpoint_sha256": vocoder_meta.get("checkpoint_sha256"),
        "vocoder_pretrained_sha256": vocoder_meta.get("pretrained_sha256"),
        "export_version": acoustic_meta.get("export_version"),
        "acoustic_status": provenance["new_exports"]["acoustic"]["tensor_match"],
        "vocoder_status": provenance["new_exports"]["vocoder"]["tensor_match"],
    }

    acoustic_results = []
    vocoder_results = []
    e2e_results = []
    determinism = {"pytorch": [], "onnx": []}
    for index, item in enumerate(items):
        text = item["text"]
        parsed = parse_text(text, resolve_oov=True)
        features = np.array([[engine.vocabs.encode(parsed_item) for parsed_item in parsed]],
                            dtype=np.int64)
        item["frontend"] = {
            "n_syllables": len(parsed),
            "ids": features.tolist(),
            "sha256": hashlib.sha256(features.tobytes()).hexdigest(),
        }
        # A. acoustic, 3 cấu hình speed/gate
        for speed, gate in ((1.0, 0.5), (0.75, 1.01), (1.25, 0.5)):
            with torch.no_grad():
                ref = engine.model(torch.from_numpy(features),
                                   speed=torch.tensor([speed], dtype=torch.float32),
                                   gate_threshold=gate)
            got = acoustic.run(features, speed=speed, gate_threshold=gate)
            names = sorted(set(ref) & set(got))
            per_name = {}
            for name in names:
                left = ref[name].detach().cpu().numpy()
                right = got[name]
                if name in ("durations", "slot_durations", "lengths") or np.issubdtype(left.dtype, np.integer):
                    per_name[name] = exact(left, right)
                else:
                    per_name[name] = array_stats(left, right)
            mel = per_name.get("mel", {})
            ok = (
                per_name.get("durations", {}).get("equal", False)
                and per_name.get("slot_durations", {}).get("equal", False)
                and per_name.get("lengths", {}).get("equal", False)
                and mel.get("shape_equal", False)
                and mel.get("finite", False)
                and mel.get("max_abs", 1e9) <= CRITERIA["acoustic"]["mel_max_abs"]
                and mel.get("mean_abs", 1e9) <= CRITERIA["acoustic"]["mel_mean_abs"]
                and all(value.get("finite", False) and value.get("max_abs", 0.0)
                        <= CRITERIA["acoustic"]["continuous_max_abs"]
                        for key, value in per_name.items()
                        if key not in ("durations", "slot_durations", "lengths", "mel"))
            )
            acoustic_results.append({
                "id": item["id"], "kind": item["kind"], "speed": speed, "gate": gate,
                "outputs": per_name, "pass": bool(ok),
            })
        # B. vocoder dùng mel PyTorch speed=1/gate=0.5
        with torch.no_grad():
            ref_acoustic = engine.model(torch.from_numpy(features),
                                        speed=torch.tensor([1.0], dtype=torch.float32),
                                        gate_threshold=0.5)
            mel_t = ref_acoustic["mel"].detach().float()
            ref_real, ref_imag = engine.vocoder.spectrogram_parts(mel_t)
            ref_wave = engine.vocoder(mel_t)[0].detach().cpu().numpy()
        got_real, got_imag = vocoder.spectrogram(mel_t.cpu().numpy())
        got_wave = vocoder.waveform(mel_t.cpu().numpy())
        ref_wave = np.asarray(ref_wave).reshape(-1)
        got_wave = np.asarray(got_wave).reshape(-1)
        spectrum_real = array_stats(ref_real[0].cpu().numpy(), got_real)
        spectrum_imag = array_stats(ref_imag[0].cpu().numpy(), got_imag)
        wave = array_stats(ref_wave, got_wave)
        wave_ok = (
            wave.get("shape_equal", False)
            and wave.get("max_abs", 1e9) <= CRITERIA["vocoder"]["waveform_max_abs"]
            and wave.get("rmse", 1e9) <= CRITERIA["vocoder"]["waveform_rmse"]
            and spectrum_real.get("max_abs", 1e9) <= CRITERIA["vocoder"]["spectrum_max_abs"]
            and spectrum_imag.get("max_abs", 1e9) <= CRITERIA["vocoder"]["spectrum_max_abs"]
            and spectrum_real.get("mean_abs", 1e9) <= CRITERIA["vocoder"]["spectrum_mean_abs"]
            and spectrum_imag.get("mean_abs", 1e9) <= CRITERIA["vocoder"]["spectrum_mean_abs"]
        )
        vocoder_results.append({
            "id": item["id"], "kind": item["kind"],
            "spectrum_real": spectrum_real, "spectrum_imag": spectrum_imag,
            "waveform": wave,
            "ref_rms": round(float(np.sqrt(np.mean(ref_wave ** 2))), 6),
            "got_rms": round(float(np.sqrt(np.mean(got_wave ** 2))), 6),
            "ref_peak": round(float(np.max(np.abs(ref_wave))), 6),
            "got_peak": round(float(np.max(np.abs(got_wave))), 6),
            "clipping_ref": int(np.sum(np.abs(ref_wave) >= 0.999)),
            "clipping_got": int(np.sum(np.abs(got_wave) >= 0.999)),
            "pass": bool(wave_ok),
        })
        # C. end-to-end chunk mode, cùng frontend/chunking/pause/fade, không LUFS
        pt_path = OUT / f"{index:02d}_pytorch.wav"
        onnx_path = OUT / f"{index:02d}_onnx.wav"
        engine.synthesize(text, pt_path, gate_threshold=0.5, sharpen_alpha=0.0,
                          target_lufs=None, peak_normalize=False, max_chars=400)
        synthesize_onnx(text, ACOUSTIC, onnx_path, vocoder_path=VOCODER, provider="cpu",
                        gate_threshold=0.5, target_lufs=None, peak_normalize=False,
                        max_chars=400, steady=False, sample_rate=24000)
        pt_wave, pt_sr = sf.read(pt_path, dtype="float32")
        onnx_wave, onnx_sr = sf.read(onnx_path, dtype="float32")
        pt_wave = np.asarray(pt_wave).reshape(-1)
        onnx_wave = np.asarray(onnx_wave).reshape(-1)
        e2e = array_stats(pt_wave, onnx_wave)
        rms_ratio = (float(np.sqrt(np.mean(onnx_wave ** 2)))
                     / max(float(np.sqrt(np.mean(pt_wave ** 2))), 1e-12))
        clipping_extra = int(np.sum(np.abs(onnx_wave) >= 0.999)) - int(np.sum(np.abs(pt_wave) >= 0.999))
        e2e_ok = (
            e2e.get("shape_equal", False)
            and e2e.get("max_abs", 1e9) <= CRITERIA["end_to_end"]["max_abs"]
            and e2e.get("rmse", 1e9) <= CRITERIA["end_to_end"]["rmse"]
            and CRITERIA["end_to_end"]["rms_ratio_min"] <= rms_ratio
            <= CRITERIA["end_to_end"]["rms_ratio_max"]
            and clipping_extra <= CRITERIA["end_to_end"]["clipping_extra"]
        )
        e2e_results.append({
            "id": item["id"], "kind": item["kind"], "sample_rate_pt": pt_sr,
            "sample_rate_onnx": onnx_sr, "waveform": e2e,
            "rms_ratio": round(rms_ratio, 6), "clipping_extra": clipping_extra,
            "pass": bool(e2e_ok),
        })
        # determinism: chạy lại cùng input trên cùng runtime
        pt2 = OUT / f"{index:02d}_pytorch_2.wav"
        onnx2 = OUT / f"{index:02d}_onnx_2.wav"
        engine.synthesize(text, pt2, gate_threshold=0.5, sharpen_alpha=0.0,
                          target_lufs=None, peak_normalize=False, max_chars=400)
        synthesize_onnx(text, ACOUSTIC, onnx2, vocoder_path=VOCODER, provider="cpu",
                        gate_threshold=0.5, target_lufs=None, peak_normalize=False,
                        max_chars=400, steady=False, sample_rate=24000)
        pt2_wave, _ = sf.read(pt2, dtype="float32")
        onnx2_wave, _ = sf.read(onnx2, dtype="float32")
        determinism["pytorch"].append(bool(np.array_equal(pt_wave, np.asarray(pt2_wave).reshape(-1))))
        determinism["onnx"].append(bool(np.array_equal(onnx_wave, np.asarray(onnx2_wave).reshape(-1))))

    # frontend comparison (chunk mode dùng cùng chunk_text/parse_text)
    frontend = []
    for item in items:
        chunks = chunk_text(item["text"], max_chars=400)
        for chunk in chunks:
            parsed = parse_text(chunk, resolve_oov=True)
            left = np.array([[engine.vocabs.encode(x) for x in parsed]], dtype=np.int64)
            right = acoustic.features_from_text(chunk)
            frontend.append({"id": item["id"], "chunk": chunk,
                             "equal": bool(np.array_equal(left, right)),
                             "shape_left": list(left.shape), "shape_right": list(right.shape)})

    result = {
        "criteria": CRITERIA,
        "runtime": {
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "VIFASTT_INTRA_THREADS": os.environ.get("VIFASTT_INTRA_THREADS"),
            "onnx_provider": "CPUExecutionProvider",
            "pyTorch_acoustic": P2, "pyTorch_acoustic_sha256": sha256(P2),
            "pyTorch_vocoder": cfg["vocoder_training"]["pretrained_path"],
            "pyTorch_vocoder_sha256": sha256(cfg["vocoder_training"]["pretrained_path"]),
            "onnx_acoustic": ACOUSTIC, "onnx_vocoder": VOCODER,
        },
        "metadata": {"acoustic": acoustic_meta, "vocoder": vocoder_meta},
        "provenance": provenance_checks,
        "inputs": items,
        "frontend": frontend,
        "acoustic": acoustic_results,
        "vocoder": vocoder_results,
        "end_to_end": e2e_results,
        "determinism": {
            "pytorch_all_equal": all(determinism["pytorch"]),
            "onnx_all_equal": all(determinism["onnx"]),
            "pytorch": determinism["pytorch"], "onnx": determinism["onnx"],
        },
    }
    result["summary"] = {
        "acoustic_pass": sum(1 for row in acoustic_results if row["pass"]),
        "acoustic_total": len(acoustic_results),
        "vocoder_pass": sum(1 for row in vocoder_results if row["pass"]),
        "vocoder_total": len(vocoder_results),
        "end_to_end_pass": sum(1 for row in e2e_results if row["pass"]),
        "end_to_end_total": len(e2e_results),
        "frontend_equal": sum(1 for row in frontend if row["equal"]),
        "frontend_total": len(frontend),
        "provenance_ok": all([
            provenance_checks["acoustic_sha_matches_v1837"],
            provenance_checks["vocoder_sha_matches_v1837"],
            provenance_checks["acoustic_mel_hash"] == provenance_checks["mel_hash_expected"],
            provenance_checks["vocoder_mel_hash"] == provenance_checks["mel_hash_expected"],
        ]),
        "deterministic": result["determinism"]["pytorch_all_equal"]
        and result["determinism"]["onnx_all_equal"],
    }
    result["ready_to_switch_onnx"] = bool(
        result["summary"]["acoustic_pass"] == result["summary"]["acoustic_total"]
        and result["summary"]["vocoder_pass"] == result["summary"]["vocoder_total"]
        and result["summary"]["end_to_end_pass"] == result["summary"]["end_to_end_total"]
        and result["summary"]["frontend_equal"] == result["summary"]["frontend_total"]
        and result["summary"]["provenance_ok"]
        and result["summary"]["deterministic"]
    )
    result["switch_commands"] = {
        "onnx": f"vifasttts synthesize-onnx --acoustic {ACOUSTIC} --vocoder {VOCODER} --text '<text>' --output out.wav",
        "rollback_pytorch": f"vifasttts synthesize --config {CFG32} --checkpoint {P2} --text '<text>' --output out.wav",
        "rollback_onnx_old": f"vifasttts synthesize-onnx --acoustic {OLD_ACOUSTIC} --vocoder {OLD_VOCODER} --text '<text>' --output out.wav",
        "note": "chưa chuyển production; old ONNX giữ rollback lịch sử (vocoder E4-C)",
    }
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(json.dumps({"summary": result["summary"],
                      "ready_to_switch_onnx": result["ready_to_switch_onnx"],
                      "determinism": result["determinism"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

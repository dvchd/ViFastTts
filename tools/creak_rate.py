"""tools/creak_rate.py — tỉ lệ block creak (CreaPy) theo nhóm thanh trên wav đã sinh.

Dùng cho cổng rè (docs/ACOUSTIC_V2.md §38–§45): so bản tổng hợp với audio THẬT —
ngã/nặng phải creak nhiều hơn ngang/huyền, và bản synth **không** được vượt xa thật.

Tham số theo V11S2.A4 (đã kiểm chứng trên 25 clip thật: 1,73×): `min creak 60 ms ·
ngưỡng xác suất 0,8 · STE phân vị 30 mỗi clip`, `gender_model=male` (F0 trung vị 122 Hz).
CreaPy là GPL-3.0, **không** là dependency của repo: cài ngoài (`~/creapy-venv`,
nguồn `~/creapy-src`) rồi trỏ `--creapy-src` (mặc định `~/creapy-src`).

Chạy:
    .venv/bin/python tools/creak_rate.py --clips artifacts/creak25/clips25.json \
        --synth-dir artifacts/creak25/copy_synth --out artifacts/creak25/copy_synth.json

Kết quả JSON: `{creapy: {…tham số}, groups: {ngã+nặng: {n, mean, median}, …}, ratio,
n_clips, clips_loi}` — `ratio` = (ngã+nặng)/(ngang+huyền), mốc THẬT = 1,73.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HOP_S = 256 / 24000  # hop 256 @ 24 kHz = độ dài frame manifest (giây)
GROUP = {"NGA": "ngã+nặng", "NANG": "ngã+nặng", "NGANG": "ngang+huyền", "HUYEN": "ngang+huyền"}


def _load_creapy(source: str):
    """Nạp CreaPy từ thư mục nguồn (ngoài repo); thiếu ⇒ lỗi rõ ràng, không crash lạ."""
    if source and source not in sys.path:
        sys.path.insert(0, source)
    try:
        import creapy  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - phụ thuộc máy
        raise SystemExit(
            f"Thiếu CreaPy (GPL-3.0, không nằm trong repo): clone về rồi trỏ --creapy-src.\n"
            f"  git clone https://github.com/SPSC-TUGraz/creapy {source}\n"
            f"({exc})"
        ) from exc
    return creapy


def measure(clips_path: str | Path, synth_dir: str | Path, *, creapy_src: str = "~/creapy-src",
            gender: str = "male", threshold: float = 0.8, ste_percentile: float = 30.0,
            real: bool = False, windows_path: str | Path | None = None) -> dict:
    """Tính tỉ lệ block creak theo nhóm thanh trên `synth_dir/<parent>.wav`.

    `real=True` (mốc đối chứng): đọc chính `row["audio_path"]` cắt theo `frame_range`
    thay vì wav tổng hợp — cùng một thước, cùng cách quy cửa sổ âm tiết.

    `windows_path`: JSON `{parent: [{tone, start, end}, …]}` (giây) — **bắt buộc cho
    FREE-RUN**: model tự đặt thời lượng nên cửa sổ phải lấy từ `slot_durations` của model
    (`tools/free_run_clips.py --windows-out`), không dùng `row["durations"]` của manifest.
    """
    import numpy as np
    import soundfile as sf

    creapy = _load_creapy(str(Path(creapy_src).expanduser()))
    creapy.set_config(gender_model=gender, creak_threshold=threshold, verbose=False)

    clips = json.loads(Path(clips_path).read_text(encoding="utf-8"))
    windows = json.loads(Path(windows_path).read_text(encoding="utf-8")) if windows_path else {}
    store: dict[str, list[float]] = {}
    failed: list[str] = []
    for entry in clips:
        wav = Path(synth_dir) / f"{entry['parent']}.wav"
        if real:
            row = entry["row"]
            audio, sample_rate = sf.read(row["audio_path"], dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            start, end = row["frame_range"]
            y = audio[start * 256:end * 256]
            sr = sample_rate
        else:
            if not wav.exists():
                failed.append(entry["parent"])
                continue
            y, sr = sf.read(str(wav), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
        # STE phân vị theo TỪNG clip (A4): ngưỡng năng lượng thích ứng độ to từng bản.
        frame_len, hop = int(0.04 * sr), int(0.01 * sr)
        count = 1 + max(0, (len(y) - frame_len) // hop)
        idx = np.arange(frame_len)[None, :] + hop * np.arange(count)[:, None]
        ste = (y[idx] ** 2).mean(axis=1)
        creapy.set_config(ste_threshold=float(np.percentile(ste, ste_percentile)))
        try:
            _x, y_pred, _sr = creapy.process_file(str(wav) if not real else entry["row"]["audio_path"],
                                                  gender_model=gender)
        except Exception:  # noqa: BLE001 - clip lỗi: bỏ và ghi lại, không làm sập cả lượt đo
            failed.append(entry["parent"])
            continue
        row = entry["row"]
        if windows_path is not None:
            syllables = [(w["start"], w["end"], w["tone"]) for w in windows[entry["parent"]]]
        else:
            tones = [s["abstract"]["tone"] for s in row["syllables"]]
            durs, gaps = row["durations"], row["gaps"]
            cursor = row.get("lead_frames", 0) * HOP_S
            syllables = []
            for index, (duration, tone) in enumerate(zip(durs, tones)):
                syllables.append((cursor, cursor + duration * HOP_S, tone))
                cursor = cursor + duration * HOP_S + ((gaps[index] * HOP_S) if index < len(gaps) else 0.0)
        for start, end, tone in syllables:
            lo, hi = int(start / 0.01), max(int(end / 0.01), int(start / 0.01) + 1)
            block = y_pred[lo:min(hi, len(y_pred))]
            group = GROUP.get(tone)
            if group and len(block) >= 2:
                store.setdefault(group, []).append(float(np.mean(block > 0.5)))

    groups = {name: {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values))}
              for name, values in sorted(store.items())}
    ratio = None
    if "ngã+nặng" in groups and "ngang+huyền" in groups:
        ratio = groups["ngã+nặng"]["mean"] / max(groups["ngang+huyền"]["mean"], 1e-9)
    return {
        "clips": str(clips_path),
        "synth_dir": str(synth_dir),
        "real": bool(real),
        "windows": str(windows_path) if windows_path else None,
        "creapy": {"gender": gender, "threshold": threshold, "ste_percentile": ste_percentile,
                   "min_creak_ms": 60},
        "n_clips": len(clips),
        "clips_loi": failed,
        "groups": groups,
        "ratio": ratio,
    }


def main() -> None:
    """CLI: đọc clips + synth dir, ghi JSON + in bảng gọn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, help="JSON [{parent, row}, …] (25 clip ngã/nặng)")
    parser.add_argument("--synth-dir", required=True, help="Thư mục wav đã sinh, tên <parent>.wav")
    parser.add_argument("--out", required=True, help="Tệp JSON kết quả")
    parser.add_argument("--creapy-src", default="~/creapy-src")
    parser.add_argument("--gender", default="male")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--ste-percentile", type=float, default=30.0)
    parser.add_argument("--real", action="store_true",
                        help="Mốc đối chứng: đọc audio gốc theo frame_range thay vì wav tổng hợp")
    parser.add_argument("--windows", default=None,
                        help="JSON cửa sổ âm tiết theo model (BẮT BUỘC cho free-run)")
    args = parser.parse_args()
    result = measure(args.clips, args.synth_dir, creapy_src=args.creapy_src, gender=args.gender,
                     threshold=args.threshold, ste_percentile=args.ste_percentile, real=args.real,
                     windows_path=args.windows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"clip dùng {result['n_clips'] - len(result['clips_loi'])}/{result['n_clips']}"
          f"{' (lỗi: ' + ', '.join(result['clips_loi'][:3]) + '…)' if result['clips_loi'] else ''}")
    for name, values in result["groups"].items():
        print(f"  {name:<12} n={values['n']:>5}  mean={values['mean']:.4f}  median={values['median']:.4f}")
    if result["ratio"] is not None:
        print(f"  tỉ lệ ngã+nặng / ngang+huyền = {result['ratio']:.3f} (thật 1,73)")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()

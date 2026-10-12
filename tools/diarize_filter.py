"""Lọc âm thanh không phải giọng chủ kênh trong dataset YouTube một-giọng.

Dùng **NVIDIA Nemotron 3 Diarization** (transformers, CPU) theo quy trình 2 lượt đã kiểm chứng:

  Lượt A (anchor toàn cục): ghép ~6s anchor từ 1–2 clip sạch bạn chỉ định (--anchor-video/--anchor-clip)
      → kết quả sơ bộ để chọn anchor riêng cho từng video.
  Lượt B (anchor theo video): mỗi video dùng anchor lấy từ chính video đó (cùng mic/phòng)
      → kết quả cuối + danh sách clip cần loại.

Vì sao phải 2 lượt: anchor từ video khác điều kiện thu âm có thể làm **giọng chủ kênh bị gán nhầm
sang speaker khác**. Đo thực tế trên kênh tham chiếu: lượt A báo nhầm 134/149 clip (giữ lại đúng);
lượt B chỉ còn 48 clip bị loại, khớp kiểm chứng thủ công.

Nguyên lý: speakers của model xếp theo thứ tự xuất hiện trong audio; vì anchor (chủ kênh) đứng đầu
mỗi clip nên **spk0 = chủ kênh**, mọi `spk>0` trong vùng clip là âm thanh khác (TTS/AI/demo/khách).

Yêu cầu: `pip install -e ".[diar]"` (transformers>=5.19, librosa, torch).

Chạy (ví dụ kênh tham chiếu):
    python tools/diarize_filter.py \
        --manifest data/codelung/manifest.jsonl \
        --anchor-video EM_ObHWvd94 \
        --exclude-video KIrbA-wEURg,LY8_RaLDJT4,W1m6birJBj0,_U4H5FSIb-c,nEPr-jIokLw,pLXBrJ-M-Xo \
        --out-dir data/codelung/review/diar

Kết quả trong --out-dir:
    pass_a.jsonl        kết quả lượt A (để chọn anchor theo video)
    clip_other.jsonl    kết quả cuối: duration, anchor_dom/cov, owner_s, other_s, unlabeled_frac, segments
    drop_other_ge_{t}.txt   danh sách clip có other_s >= ngưỡng (mặc định 0,5s)
    summary.json        thống kê theo video + danh sách clip "anchor không đạt" (cov<0,95 hoặc dom≠0)
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "nvidia/Nemotron-3-Diarization"


# ---------------------------------------------------------------- tiện ích
def load_wav16(path: str | Path) -> Any:
    """Đọc WAV mono PCM16 và resample về 16 kHz (low-pass nhẹ + nội suy tuyến tính)."""
    import numpy as np

    with wave.open(str(path), "rb") as fh:
        if fh.getsampwidth() != 2:
            raise ValueError(f"Chỉ hỗ trợ WAV PCM16: {path}")
        x = np.frombuffer(fh.readframes(fh.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if fh.getnchannels() > 1:
            x = x.reshape(-1, fh.getnchannels()).mean(1)
        rate = fh.getframerate()
    if rate != 16000:
        x = np.convolve(x, np.ones(5, np.float32) / 5, mode="same")
        n16 = round(len(x) * 16000 / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n16, dtype=np.float64), np.arange(len(x)), x).astype(np.float32)
    return x


def audio_path_of(row: dict[str, Any], root: Path) -> Path:
    """Đường dẫn audio của một dòng manifest.

    Ưu tiên: đường dẫn tuyệt đối → đường dẫn tương đối so với CWD (như manifest của repo này)
    → cuối cùng mới ghép với `--audio-root`.
    """
    path = Path(str(row["audio_path"]))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


class _Engine:
    """Bọc processor + model + anchor; tách riêng để test/tái dùng."""

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(model_id).eval()

    def diarize(self, audio16: Any, anchor_seconds: float) -> dict[str, Any]:
        """Chạy model trên [anchor + clip]; trả owner/other/unlabeled/anchor-quality/segments."""
        import numpy as np

        torch = self.torch
        inputs = self.processor(audio16, sampling_rate=16000).to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        probs = logits[0].sigmoid().numpy()
        segs = self.processor.extract_speaker_dict(logits, inputs.attention_mask)[0]

        n_a = min(int(anchor_seconds * 100), probs.shape[0])
        ap = probs[:n_a]
        dom = int(ap.sum(0).argmax())
        cov = float((ap.argmax(1) == dom).mean())

        clip16 = audio16[n_a * 160:]
        n_frames = min(len(clip16) // 160, probs.shape[0] - n_a)
        frames = clip16[: n_frames * 160].reshape(n_frames, 160)
        rms_db = 20 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(1) + 1e-12) + 1e-12)
        speech = rms_db >= (np.percentile(rms_db, 90) - 20.0) if n_frames else np.zeros(0, bool)
        pmax = probs[n_a: n_a + n_frames].max(1) if n_frames else np.zeros(0)
        unl = float((pmax[speech] < 0.3).mean()) if speech.any() else 0.0

        owner = other = 0.0
        out_segs = []
        total_s = len(audio16) / 16000
        for s in segs:
            st, en, sp = float(s["Start"]), float(s["End"]), int(s["Speaker"])
            if en <= anchor_seconds + 0.05:
                continue
            st = max(st, anchor_seconds) - anchor_seconds
            en = min(en, total_s) - anchor_seconds
            if en - st <= 0:
                continue
            out_segs.append([round(st, 2), round(en, 2), sp])
            if sp == 0:
                owner += en - st
            else:
                other += en - st
        return {"anchor_dom": dom, "anchor_cov": round(cov, 4),
                "owner_s": round(owner, 2), "other_s": round(other, 2),
                "unlabeled_frac": round(unl, 4), "segments": out_segs}


# ---------------------------------------------------------------- các lượt
def build_anchor_clips(rows: list[dict[str, Any]], anchor_video: str | None,
                       anchor_clips: list[str], durations: dict[str, float], n: int = 2) -> list[str]:
    """Chọn clip làm anchor từ --anchor-clip hoặc video chỉ định (chọn độ dài gần 12s)."""
    if anchor_clips:
        return anchor_clips[:n]
    if not anchor_video:
        raise SystemExit("Cần --anchor-video hoặc --anchor-clip")
    pool = [r for r in rows if r["id"].startswith(anchor_video + "_")]
    pool.sort(key=lambda r: abs(durations.get(r["id"], 12.0) - 12.0))
    if not pool:
        raise SystemExit(f"Không tìm thấy clip nào của video anchor {anchor_video!r}")
    return [r["id"] for r in pool[:n]]


def load_durations(manifest: Path, clips_path: str | None) -> dict[str, float]:
    """Đọc thời lượng clip từ clips.jsonl (mặc định cạnh manifest) để chọn anchor."""
    path = Path(clips_path) if clips_path else manifest.parent / "clips.jsonl"
    durations: dict[str, float] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                durations[str(record.get("id"))] = float(record.get("duration", 12.0))
    return durations


def make_anchor(engine_ids: list[str], rows_by_id: dict[str, dict[str, Any]], root: Path,
                seconds: float) -> Any:
    """Ghép 3s đầu của từng anchor clip thành một anchor ~6s."""
    import numpy as np

    parts = [load_wav16(audio_path_of(rows_by_id[cid], root))[: int(seconds * 16000)] for cid in engine_ids]
    return np.concatenate(parts)


def run_pass(rows: list[dict[str, Any]], root: Path, engine: _Engine, anchors: dict[str, Any] | None,
             default_anchor: Any, out_path: Path, label: str) -> dict[str, dict[str, Any]]:
    """Chạy diarization cho danh sách rows; resume theo out_path; trả {id: kết quả}."""
    done: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if "error" not in record:
                    done[record["id"]] = record
    todo = [r for r in rows if r["id"] not in done]
    print(f"[{label}] {len(rows)} clip · đã có {len(done)} · còn {len(todo)}", flush=True)
    started = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as fh:
        for index, row in enumerate(todo, 1):
            video = row["id"].rsplit("_", 1)[0]
            anchor = (anchors or {}).get(video, default_anchor)
            try:
                clip = load_wav16(audio_path_of(row, root))
                anchor_seconds = len(anchor) / 16000
                import numpy as np

                result = engine.diarize(np.concatenate([anchor, clip]), anchor_seconds)
                record = {"id": row["id"], "video": video, "duration": round(len(clip) / 16000, 2), **result}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[row["id"]] = record
            except Exception as exc:  # noqa: BLE001 - tiếp tục với clip khác, ghi lại lỗi
                fh.write(json.dumps({"id": row["id"], "error": str(exc)}, ensure_ascii=False) + "\n")
            if index % 500 == 0:
                fh.flush()
                rate = index / max(1e-9, time.perf_counter() - started)
                print(f"  {index}/{len(todo)} · {rate:.1f} clip/s · còn ~{(len(todo)-index)/rate/60:.1f} phút", flush=True)
    return done


def choose_video_anchors(pass_a: dict[str, dict[str, Any]], n: int = 2) -> dict[str, list[str]]:
    """Từ kết quả lượt A chọn 2 clip sạch nhất mỗi video (giọng chủ kênh, không lẫn tiếng khác)."""
    pool: dict[str, list[str]] = defaultdict(list)
    for cid, record in pass_a.items():
        clean = (record.get("other_s", 1.0) < 0.05 and record.get("owner_s", 0.0) > 2.0
                 and record.get("unlabeled_frac", 1.0) < 0.3)
        if clean:
            pool[record["video"]].append(cid)
    chosen = {}
    for video, ids in pool.items():
        ids.sort(key=lambda cid: abs(pass_a[cid]["duration"] - 12))
        chosen[video] = ids[:n]
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="Manifest JSONL của dataset")
    parser.add_argument("--audio-root", default=None, help="Gốc đường dẫn audio tương đối (mặc định: thư mục manifest)")
    parser.add_argument("--clips", default=None, help="clips.jsonl để chọn anchor theo thời lượng (mặc định cạnh manifest)")
    parser.add_argument("--anchor-video", default=None, help="Video sạch để lấy anchor lượt A")
    parser.add_argument("--anchor-clip", action="append", default=[], help="Clip id làm anchor (lặp lại được)")
    parser.add_argument("--exclude-video", default="", help="Danh sách video loại hẳn (phẩy)")
    parser.add_argument("--out-dir", required=True, help="Thư mục kết quả")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng other_s để loại clip (giây)")
    parser.add_argument("--anchor-seconds", type=float, default=3.0, help="Giây lấy từ mỗi anchor clip")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số clip (chạy thử)")
    parser.add_argument("--skip-pass-a", action="store_true", help="Bỏ lượt A (đã có pass_a.jsonl)")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    root = Path(args.audio_root) if args.audio_root else manifest.parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded = {v for v in args.exclude_video.split(",") if v}
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r["id"].rsplit("_", 1)[0] not in excluded]
    if args.limit:
        rows = rows[: args.limit]
    rows_by_id = {r["id"]: r for r in rows}
    print(f"{len(rows)} clip · loại {len(excluded)} video (nếu có) · root {root}", flush=True)

    engine = _Engine(args.model)
    durations = load_durations(manifest, args.clips)
    anchor_ids = build_anchor_clips(rows, args.anchor_video, args.anchor_clip, durations)
    default_anchor = make_anchor(anchor_ids, rows_by_id, root, args.anchor_seconds)
    print(f"anchor toàn cục từ {anchor_ids} ({len(default_anchor)/16000:.1f}s)", flush=True)

    pass_a_path = out_dir / "pass_a.jsonl"
    if not args.skip_pass_a:
        pass_a = run_pass(rows, root, engine, None, default_anchor, pass_a_path, "A/global")
    else:
        pass_a = {json.loads(l)["id"]: json.loads(l) for l in pass_a_path.read_text(encoding="utf-8").splitlines() if l.strip() and '"error"' not in l}

    video_anchor_ids = choose_video_anchors(pass_a)
    missing = sorted({r["id"].rsplit("_", 1)[0] for r in rows} - set(video_anchor_ids))
    if missing:
        print(f"CẢNH BÁO: {len(missing)} video không có clip sạch cho anchor riêng → dùng anchor toàn cục: {missing}", flush=True)
    anchors = {v: make_anchor(ids, rows_by_id, root, args.anchor_seconds) for v, ids in video_anchor_ids.items()}

    final = run_pass(rows, root, engine, anchors, default_anchor, out_dir / "clip_other.jsonl", "B/per-video")

    drop = sorted(cid for cid, r in final.items()
                  if r.get("other_s", 0.0) >= args.threshold)
    (out_dir / f"drop_other_ge_{str(args.threshold).replace('.', 'p')}.txt").write_text(
        "\n".join(drop) + ("\n" if drop else ""), encoding="utf-8")

    total_h = sum(r["duration"] for r in final.values()) / 3600
    other_h = sum(r.get("other_s", 0.0) for r in final.values()) / 3600
    by_video: dict[str, dict[str, float]] = {}
    for r in final.values():
        entry = by_video.setdefault(r["video"], {"clips": 0, "hours": 0.0, "other_s": 0.0, "drop": 0})
        entry["clips"] += 1
        entry["hours"] += r["duration"] / 3600
        entry["other_s"] += r.get("other_s", 0.0)
        entry["drop"] += int(r.get("other_s", 0.0) >= args.threshold)
    anchor_fail = [cid for cid, r in final.items() if r.get("anchor_dom", 0) != 0 or r.get("anchor_cov", 1) < 0.95]
    summary = {
        "clips": len(final), "hours": round(total_h, 3), "other_hours": round(other_h, 4),
        "other_pct": round(other_h / max(total_h, 1e-9) * 100, 3),
        "threshold_s": args.threshold, "drop_clips": len(drop),
        "anchor_fail_ids": anchor_fail, "video_anchors": video_anchor_ids,
        "by_video": {v: {k: round(x, 4) for k, x in e.items()} for v, e in sorted(by_video.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"XONG: {len(final)} clip · tiếng khác {other_h:.3f}h ({summary['other_pct']}%) · "
          f"loại {len(drop)} clip · anchor không đạt: {len(anchor_fail)}")
    print(f"Kết quả: {out_dir}/clip_other.jsonl · {out_dir}/summary.json")


if __name__ == "__main__":
    main()

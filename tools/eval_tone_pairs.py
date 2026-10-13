"""tools/eval_tone_pairs.py — chấm thanh điệu trên minimal pair tổng hợp (T3.3).

Mục đích (§12 docs/ACOUSTIC_V2.md): đặt mỗi âm tiết vào **câu khung cố định**,
tổng hợp bằng model v2, rồi chấm **âm tiết đích sau align** — báo độ chính xác
thanh theo nhóm + trang nghe A/B mù.

Cách làm (ghi rõ để tái lập):

1. Bộ minimal pair: mỗi base × 6 thanh (NGANG/SAC/HUYEN/HOI/NGA/NANG). Dấu đặt
   bằng `frontend/orthography.py::compose_canonical` qua `apply_tone_to_base`
   (parse base lấy onset/medial/nucleus/coda rồi dựng lại chính tả canonical) —
   không tự đặt dấu thủ công.
2. Câu khung: `"Tôi nói {syl} nhé"` → âm tiết đích ở giữa (index 2/4).
3. Tổng hợp: `vifasttts.infer.Synthesizer` (public `iter_synthesize`) cho WAV;
   frame của âm tiết đích lấy từ **một forward thẳng** `model(features,
   speed=[speed])` trên cùng `engine.model` (tất định → durations khớp WAV):
   `slot_durations` layout `[lead, (O,M,N,C,G)×N]` → cửa sổ đích = 4 slot nội
   dung (bỏ G), nhân `hop_length` ra mẫu audio (clamp theo WAV thật).
4. F0 thật: `data/features.py::estimate_f0_pyworld` (harvest+stonemask) trên mẫu
   của âm tiết đích; chỉ giữ frame `vuv > 0,5` (F4).
5. Đặc trưng + phân loại ("chấm theo thống kê thật"): z-score log-F0 bằng
   `f0_log_mean/std` → `dct4/√n` = `s_k` (4 hệ số, khớp §6: `s_k = c_k/√n`,
   `s0 == mean`) → z-score bằng `shape_mean/std` → chọn thanh có `tone_prior_z`
   gần nhất (Euclid; hoà phá theo thứ tự NGANG, HUYEN, SAC, HOI, NGA, NANG —
   `argmin` lấy đầu). `tone_prior_z` tính đúng công thức
   `model.load_f0_stats` (prior mean trừ `shape_mean` chia `shape_std`).
6. Báo cáo JSON (`--out`): accuracy tổng + per-thanh + ma trận nhầm 6×6 +
   danh sách ca sai + `chance = 1/6`; in bảng ngắn ra stdout.
7. Trang A/B mù (`--html` + `--answer-key`): audio nhúng base64, thứ tự xáo
   trộn (seed cố định), **không ghi tên thanh** trên trang; key JSON giữ thứ tự
   → nhãn đúng. File local, **không upload lên GitHub**.

Lưu ý: F0 đo từ audio tổng hợp nên checkpoint v2 cần kèm trọng số vocoder đã
huấn luyện (không thì WAV ngẫu nhiên, điểm vô nghĩa). Mặc định chạy CPU để
không tranh MPS với tiến trình train.

Ví dụ:
  python tools/eval_tone_pairs.py --config configs/small_codelung_v2.yaml \\
      --checkpoint runs/acoustic_v2_slot5b/best.pt \\
      --out artifacts/eval_tone_pairs/slot5b.json \\
      --html artifacts/eval_tone_pairs/slot5b_ab.html \\
      --answer-key artifacts/eval_tone_pairs/slot5b_ab_key.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vifasttts.frontend.orthography import compose_canonical
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.syllable import safe_parse_syllable
from vifasttts.types import Tone

#: Thứ tự chuẩn (cột/hàng ma trận nhầm + thứ tự hoà phá khi hoà khoảng cách).
TONES = ["NGANG", "HUYEN", "SAC", "HOI", "NGA", "NANG"]
CHANCE = 1.0 / 6.0
DEFAULT_BASES = ["ma", "xa", "la", "ban", "men", "đi"]
DEFAULT_TEMPLATE = "Tôi nói {syl} nhé"
V2_DEFAULT_F0_STATS = "data/codelung/f0_stats_v2.json"


def apply_tone_to_base(base: str, tone: str | Tone) -> str:
    """Gắn thanh vào base bằng chính tả canonical (dùng `compose_canonical` có sẵn)."""
    item = safe_parse_syllable(base)
    if item is None:
        raise ValueError(f"Base không parse được thành âm tiết Việt: {base!r}")
    tone_enum = tone if isinstance(tone, Tone) else Tone(str(tone).upper())
    orth = item.orthography
    return compose_canonical(orth.onset, orth.medial, orth.nucleus, orth.coda, tone_enum)


def build_pairs(bases: list[str], tones: list[str]) -> list[dict]:
    """Mỗi base × mỗi thanh → (base, tone, mặt chữ có dấu)."""
    pairs = []
    for base in bases:
        for tone in tones:
            pairs.append({"base": base, "tone": tone, "syl": apply_tone_to_base(base, tone)})
    return pairs


def frame_sentence(syl: str, template: str = DEFAULT_TEMPLATE) -> str:
    """Đặt âm tiết đích vào câu khung cố định."""
    return template.format(syl=syl)


def parse_target_index(text: str, target_syl: str):
    """Parse câu khung → (items, index âm tiết đích so khớp `normalized`)."""
    items = parse_text(text)
    reference = safe_parse_syllable(target_syl)
    key = reference.normalized if reference is not None else target_syl.lower()
    for index, item in enumerate(items):
        if item.normalized == key:
            return items, index
    raise ValueError(f"Không tìm thấy âm tiết đích {target_syl!r} trong {text!r}")


def dct4_numpy(values: np.ndarray, keep: int = 4) -> np.ndarray:
    """DCT-II trực giao khớp `data/features.py::dct4_frames` (bản numpy, float64)."""
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        return np.zeros(0)
    k = min(int(keep), n)
    index = np.arange(k)[:, None]
    frames = np.arange(n)[None, :]
    basis = np.cos(np.pi * index * (frames + 0.5) / n)
    basis[0] /= np.sqrt(2.0)
    return basis @ values * np.sqrt(2.0 / n)


def contour_to_sk_z(f0_hz: np.ndarray, *, f0_log_mean: float, f0_log_std: float,
                    shape_mean: np.ndarray, shape_std: np.ndarray) -> np.ndarray:
    """F0 (Hz, chỉ frame hữu thanh) → `s_k` z-score (4 hệ số; thiếu thì pad 0)."""
    f0 = np.maximum(np.asarray(f0_hz, dtype=np.float64), 1e-6)
    z_scored = (np.log(f0) - float(f0_log_mean)) / max(float(f0_log_std), 1e-6)
    coeffs = dct4_numpy(z_scored) / np.sqrt(max(z_scored.size, 1))
    shaped = np.zeros(4)
    shaped[:coeffs.size] = coeffs[:4]
    mean = np.asarray(shape_mean, dtype=np.float64)
    std = np.maximum(np.asarray(shape_std, dtype=np.float64), 1e-6)
    return (shaped - mean) / std


def tone_prior_z_from_stats(stats: dict) -> np.ndarray:
    """`shape_tone_prior` → `[6,4]` z-score, đúng công thức `model.load_f0_stats`."""
    mean = np.asarray(stats["shape_mean"], dtype=np.float64)
    std = np.maximum(np.asarray(stats["shape_std"], dtype=np.float64), 1e-6)
    table = np.zeros((6, 4))
    prior = stats.get("shape_tone_prior") or {}
    for index, tone in enumerate(TONES):
        entry = prior.get(tone) or {}
        values = entry.get("mean") or []
        if len(values) == 4:
            table[index] = (np.asarray(values, dtype=np.float64) - mean) / std
    return table


def classify_shape(s_k_z: np.ndarray, tone_prior_z: np.ndarray) -> str:
    """Chọn thanh có prior gần nhất (Euclid); hoà phá theo thứ tự `TONES`."""
    observed = np.asarray(s_k_z, dtype=np.float64).reshape(4)
    priors = np.asarray(tone_prior_z, dtype=np.float64).reshape(6, 4)
    distances = ((priors - observed) ** 2).sum(axis=1)
    return TONES[int(np.argmin(distances))]


def target_frame_window(slot_durations: list[int], target_index: int,
                        which: str = "nucleus") -> tuple[int, int]:
    """`(start_frame, span_frames)` của cửa sổ đo trong âm tiết đích.

    `which="nucleus"` (mặc định): chỉ slot N — khớp định nghĩa `tone_prior_z`/`shape_mean`
    (R5b/T3.2 tính trên frame **nucleus**, mask `vuv`); `"syllable"`: cả 4 slot O/M/N/C (chẩn đoán).
    """
    lead = int(slot_durations[0])
    base = 1 + 5 * int(target_index)
    if which == "syllable":
        return lead + sum(int(v) for v in slot_durations[1:base]), \
            sum(int(v) for v in slot_durations[base:base + 4])
    start = lead + sum(int(v) for v in slot_durations[1:base]) + sum(
        int(v) for v in slot_durations[base:base + 2])  # + O + M
    return start, int(slot_durations[base + 2])  # span = N


def _robust_f0(segment, sample_rate: int, hop: int):
    """F0 "chắc tay" cho audio tổng hợp: harvest → **dio** → autocorr, chọn bộ nhiều frame
    hữu thanh nhất (đo thật: ảnh Vocos làm harvest rụng hữu thanh ở nhiều cửa sổ đích)."""
    import numpy as np
    import pyworld
    import torch

    from vifasttts.data.features import estimate_f0, estimate_f0_pyworld

    candidates = []
    f0, vuv = estimate_f0_pyworld(segment, sample_rate, hop, f_min=70, f_max=400, interpolate=False)
    candidates.append((f0, vuv))
    samples = np.ascontiguousarray(segment.numpy().astype(np.float64))
    frame_period = 1000.0 * hop / sample_rate
    raw, dio_times = pyworld.dio(samples, sample_rate, f0_floor=70, f0_ceil=400,
                                 frame_period=frame_period)
    raw = pyworld.stonemask(samples, raw, dio_times, sample_rate)
    target = 1 + int(samples.shape[0]) // hop
    if raw.shape[0] != target and raw.shape[0] > 0:
        raw = np.interp(np.linspace(0.0, raw.shape[0] - 1.0, target), np.arange(raw.shape[0]), raw)
    f0_dio = torch.from_numpy(np.ascontiguousarray(raw)).float()
    candidates.append((f0_dio, (f0_dio > 0).float()))
    f0_auto, _ = estimate_f0(segment, sample_rate, hop, f_min=70, f_max=400)
    candidates.append((f0_auto, (f0_auto > 0).float()))
    return max(candidates, key=lambda pair: int((pair[1] > 0.5).sum()))


def wav_bytes_mono(wav: np.ndarray, sample_rate: int) -> bytes:
    """Waveform float → WAV PCM 16-bit mono (để nhúng base64 vào trang nghe)."""
    pcm = (np.clip(np.asarray(wav, dtype=np.float64), -1.0, 1.0) * 32767.0).round().astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    return buffer.getvalue()


def build_ab_page(entries: list[dict], seed: int) -> tuple[str, dict]:
    """Trang A/B mù: xáo trộn seed cố định, chỉ ghi base (không dấu thanh) + audio.

    `entries`: mỗi phần tử có `uid`, `base`, `tone`, `wav_b64`. Trả về
    (html, answer_key); key giữ `uid` hiển thị → nhãn đúng.
    """
    order = list(entries)
    random.Random(int(seed)).shuffle(order)
    parts = ["<!DOCTYPE html><html lang='vi'><head><meta charset='utf-8'>",
             "<title>Nghe A/B mù — minimal pair thanh điệu (local, không upload)</title>",
             "</head><body><h1>Nghe và đoán thanh (6 thanh: ngang/sắc/huyền/hỏi/ngã/nặng)</h1>",
             ("<p>Thứ tự đã xáo trộn. Nghe mỗi mẫu, ghi lại base + thanh đoán, rồi đối chiếu "
              "với file answer-key.</p><ol>")]
    key_items = []
    for position, entry in enumerate(order, start=1):
        label = f"Mẫu {position:02d} (base: {entry['base']})"
        parts.append(f"<li><p>{label}</p>"
                     f"<audio controls src='data:audio/wav;base64,{entry['wav_b64']}'></audio></li>")
        key_items.append({"display": position, "uid": entry["uid"],
                          "base": entry["base"], "tone": entry["tone"]})
    parts.append("</ol></body></html>")
    key = {"seed": int(seed), "items": key_items,
           "note": "File local — không upload lên GitHub."}
    return "".join(parts), key


def resolve_f0_stats(checkpoint: str, config: dict, cli_value: str | None = None) -> dict:
    """Ưu tiên: `--f0-stats` > `state['f0_stats']` > config → mặc định repo."""
    import torch

    if cli_value:
        return json.loads(Path(cli_value).read_text(encoding="utf-8"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and state.get("f0_stats"):
        return dict(state["f0_stats"])
    for candidate in (config.get("f0_stats_path"), V2_DEFAULT_F0_STATS):
        if candidate and Path(candidate).exists():
            return json.loads(Path(candidate).read_text(encoding="utf-8"))
    raise SystemExit("v2 cần f0_stats: truyền --f0-stats hoặc để f0_stats trong checkpoint")


def score_pair(engine, pair: dict, stats: dict, priors: np.ndarray, *,
               template: str, dialect: str, speed: float, hop: int,
               window: str = "nucleus", source: str = "audio") -> dict:
    """Một pair: align âm tiết đích rồi phân loại thanh.

    `source="audio"`: tổng hợp qua vocoder, trích F0 từ audio (đo "nghe được").
    `source="model"`: lấy thẳng contour `f0` của model trong free mode (đo "ý định",
    bỏ qua vocoder) — dùng chẩn đoán tách lỗi model vs lỗi đường audio.
    """
    import torch

    from vifasttts.frontend.normalizer import NormalizerOptions

    text = frame_sentence(pair["syl"], template)
    options = NormalizerOptions(dialect=dialect)  # type: ignore[arg-type]
    wav = None
    if source == "audio":
        pieces = [audio for audio, _ in engine.iter_synthesize(
            text, dialect=dialect, speed=speed, overrides=None)]
        wav = np.concatenate(pieces).astype(np.float32)
    sample_rate = int(engine.sample_rate)

    items = parse_text(text, dialect=dialect, options=options)
    _, target_index = parse_target_index(text, pair["syl"])
    features = torch.tensor([[engine.vocabs.encode(item) for item in items]],
                            dtype=torch.long, device=engine.device)
    speed_t = torch.tensor([float(speed)], dtype=torch.float32, device=engine.device)
    with torch.no_grad():
        outputs = engine.model(features, speed=speed_t)
    slots = [int(v) for v in outputs["slot_durations"][0].cpu().tolist()]
    start_frame, span_frames = target_frame_window(slots, target_index, window)
    if source == "model":
        raw = outputs["f0"][0][start_frame:start_frame + span_frames].cpu().numpy()
        if raw.size < 2:
            return {"base": pair["base"], "tone_true": pair["tone"], "text": text,
                    "tone_pred": None, "correct": False, "note": "cửa sổ đích rỗng"}
        # model xuất **log-F0 tự nhiên**; `contour_to_sk_z` nhận Hz (tự log) → exp trước.
        contour = np.exp(raw)
        s_k_z = contour_to_sk_z(contour, f0_log_mean=float(stats["f0_log_mean"]),
                                f0_log_std=float(stats["f0_log_std"]),
                                shape_mean=np.asarray(stats["shape_mean"], dtype=np.float64),
                                shape_std=np.asarray(stats["shape_std"], dtype=np.float64))
        predicted = classify_shape(s_k_z, priors)
        return {"base": pair["base"], "tone_true": pair["tone"], "text": text,
                "n_voiced": int(span_frames), "s_k_z": [round(float(v), 4) for v in s_k_z],
                "tone_pred": predicted, "correct": bool(predicted == pair["tone"])}

    # Frame mel → mẫu audio (khớp quy ước `T = 1 + len(wav)//hop`), clamp WAV thật.
    start_sample = max(int(start_frame * hop), 0)
    end_sample = min(int((start_frame + span_frames) * hop), wav.shape[0])
    result: dict = {"base": pair["base"], "tone_true": pair["tone"], "text": text,
                    "target_index": int(target_index), "start_frame": start_frame,
                    "span_frames": span_frames, "n_syllables": len(items)}
    if end_sample <= start_sample:
        result.update({"tone_pred": None, "correct": False, "note": "cửa sổ đích rỗng"})
        return result
    segment = torch.from_numpy(wav[start_sample:end_sample])
    f0, vuv = _robust_f0(segment, sample_rate, hop)
    voiced = (vuv > 0.5).numpy()
    result["n_voiced"] = int(voiced.sum())
    if int(voiced.sum()) < 2:
        result.update({"tone_pred": None, "correct": False, "note": "không đủ frame hữu thanh"})
        return result
    s_k_z = contour_to_sk_z(f0.numpy()[voiced],
                            f0_log_mean=float(stats["f0_log_mean"]),
                            f0_log_std=float(stats["f0_log_std"]),
                            shape_mean=np.asarray(stats["shape_mean"], dtype=np.float64),
                            shape_std=np.asarray(stats["shape_std"], dtype=np.float64))
    predicted = classify_shape(s_k_z, priors)
    result.update({"s_k_z": [round(float(v), 4) for v in s_k_z],
                   "tone_pred": predicted, "correct": bool(predicted == pair["tone"])})
    return result


def summarize(results: list[dict], *, checkpoint: str, template: str,
              bases: list[str], tones: list[str]) -> dict:
    """Gộp accuracy tổng + per-thanh + ma trận nhầm 6×6 + ca sai."""
    per_tone = {tone: {"n": 0, "correct": 0} for tone in TONES}
    confusion = {true: {pred: 0 for pred in TONES} for true in TONES}
    errors, unvoiced = [], 0
    for entry in results:
        true = entry["tone_true"]
        per_tone[true]["n"] += 1
        if entry.get("tone_pred") in TONES:
            confusion[true][entry["tone_pred"]] += 1
        else:
            unvoiced += 1
        if entry.get("correct"):
            per_tone[true]["correct"] += 1
        else:
            errors.append({k: entry.get(k) for k in
                           ("base", "tone_true", "tone_pred", "s_k_z", "n_voiced", "text")})
    for counts in per_tone.values():
        counts["accuracy"] = round(counts["correct"] / counts["n"], 4) if counts["n"] else None
    total = len(results)
    hits = sum(1 for entry in results if entry.get("correct"))
    return {"checkpoint": checkpoint, "template": template, "bases": bases, "tones": tones,
            "total": total, "correct": hits,
            "accuracy": round(hits / total, 4) if total else None, "chance": round(CHANCE, 4),
            "per_tone": per_tone, "confusion": confusion, "unvoiced": unvoiced,
            "errors": errors, "results": results}


def print_table(summary: dict) -> None:
    """In bảng ngắn: accuracy tổng + per-thanh."""
    print(f"Minimal pairs: {summary['correct']}/{summary['total']} đúng "
          f"(acc {summary['accuracy']:.3f}, chance {summary['chance']:.3f}, "
          f"unvoiced {summary['unvoiced']})")
    for tone in TONES:
        counts = summary["per_tone"][tone]
        acc = f"{counts['accuracy']:.3f}" if counts["accuracy"] is not None else "n/a"
        print(f"  {tone:6s} {counts['correct']:3d}/{counts['n']:3d} = {acc}")
    print("  Ma trận nhầm (hàng = thật, cột = đoán):")
    print("         " + " ".join(f"{t[:4]:>4s}" for t in TONES))
    for true in TONES:
        row = summary["confusion"][true]
        print(f"  {true:6s} " + " ".join(f"{row[pred]:>4d}" for pred in TONES))


def main() -> None:
    """Đọc tham số, chấm toàn bộ minimal pair, ghi JSON (+ HTML mù tuỳ chọn)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bases", nargs="+", default=DEFAULT_BASES)
    parser.add_argument("--tones", nargs="+", default=TONES)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--dialect", default="north")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--f0-stats", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--source", choices=["audio", "model"], default="audio",
                        help="Chấm thanh từ audio tái tạo (mặc định) hay từ f0 của model (chẩn đoán)")
    parser.add_argument("--window", choices=["nucleus", "syllable"], default="nucleus",
                        help="Cửa sổ đo F0: nucleus (khớp prior T3.2, mặc định) | syllable (chẩn đoán)")
    parser.add_argument("--html", default=None)
    parser.add_argument("--answer-key", default=None)
    args = parser.parse_args()

    import yaml

    from vifasttts.infer import Synthesizer

    tones = [str(t).upper() for t in args.tones]
    for tone in tones:
        if tone not in TONES:
            raise SystemExit(f"Thanh không hợp lệ: {tone!r} (chọn trong {TONES})")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hop = int(config["mel"]["hop_length"])
    engine = Synthesizer(config, args.checkpoint, device=args.device)
    if engine.arch != "v2_slot5":
        raise SystemExit(f"Tool này cần checkpoint v2 (arch=v2_slot5), gặp {engine.arch}")
    stats = resolve_f0_stats(args.checkpoint, config, args.f0_stats)
    priors = tone_prior_z_from_stats(stats)

    pairs = build_pairs(list(args.bases), tones)
    results = []
    for pair in pairs:
        entry = score_pair(engine, pair, stats, priors, template=args.template,
                           dialect=args.dialect, speed=args.speed, hop=hop,
                           window=args.window, source=args.source)
        results.append(entry)
        mark = "✓" if entry.get("correct") else "✗"
        print(f"  [{mark}] {pair['base']:5s} {pair['tone']:6s} → {entry.get('tone_pred')} "
              f"({entry.get('n_voiced', 0)} frame hữu thanh)")

    summary = summarize(results, checkpoint=args.checkpoint, template=args.template,
                        bases=list(args.bases), tones=tones)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nĐã ghi {out}")
    print_table(summary)

    if args.html or args.answer_key:
        # Tổng hợp lại WAV cho trang nghe (dùng cùng engine; local, không upload).
        entries = []
        for pair in pairs:
            text = frame_sentence(pair["syl"], args.template)
            wav = np.concatenate([audio for audio, _ in engine.iter_synthesize(
                text, dialect=args.dialect, speed=args.speed, overrides=None)])
            entries.append({"uid": f"{pair['base']}-{pair['tone']}", "base": pair["base"],
                            "tone": pair["tone"],
                            "wav_b64": base64.b64encode(
                                wav_bytes_mono(wav, engine.sample_rate)).decode("ascii")})
        html, key = build_ab_page(entries, args.seed)
        if args.html:
            html_path = Path(args.html)
            html_path.parent.mkdir(parents=True, exist_ok=True)
            html_path.write_text(html, encoding="utf-8")
            print(f"Đã ghi trang nghe mù {html_path} (local — không upload)")
        if args.answer_key:
            key_path = Path(args.answer_key)
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key_path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"Đã ghi đáp án {key_path} (giữ riêng, không công khai cùng trang nghe)")


if __name__ == "__main__":
    main()

"""Tải và làm sạch dataset từ kênh YouTube của chính bạn (demo + chạy thật).

Luồng: liệt kê video → tải phụ đề json3 (timing chính xác) → tải audio bestaudio
`-x` về WAV 24 kHz mono (qua ffmpeg) → gộp cue thành câu → cắt clip có fade →
lọc im lặng/trùng/fallback cao → ghi `metadata.csv`, `manifest.jsonl`, SRT sạch,
`stats.json`, `dataset_card.md` và `LICENSE-DATASET.txt`.

Công cụ ngoài: `yt-dlp` (bắt buộc) và `ffmpeg` (khuyến nghị, có thể lấy tự động
từ `imageio-ffmpeg`); đặt biến môi trường `VIFASTTTS_YTDLP` / `VIFASTTTS_FFMPEG`
nếu muốn chỉ định đường dẫn riêng. Toàn bộ chạy qua subprocess, không nhúng code
của bên thứ ba vào package.

Dữ liệu do bạn sở hữu: mặc định ghi giấy phép mở `CC BY 4.0` vào dataset card;
đổi bằng `--license`. Xem `docs/YOUTUBE_DATASET.md`.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import wave
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vifasttts import __version__
from vifasttts.data.manifest import build_manifest, coverage_report, validate_manifest
from vifasttts.data.subtitles import Cue, format_srt, load_subtitles, merge_cues
from vifasttts.frontend.dialect import Dialect
from vifasttts.frontend.normalizer import NormalizerOptions, normalize_text
from vifasttts.frontend.oov import LETTER_READINGS, resolve_oov_token, spell_token
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.syllable import safe_parse_syllable
from vifasttts.runtime.postprocess import fade
from vifasttts.types import ParseStatus

PAD_START = 0.06  # giây cắt trước câu để không mất phụ âm đầu
PAD_END = 0.12    # giây cắt sau câu để không mất âm cuối
LogFn = Callable[[str], None]

_LICENSE_LINKS = {
    "cc by 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc by-sa 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc by-nd 4.0": "https://creativecommons.org/licenses/by-nd/4.0/",
    "cc by-nc 4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc0 1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}


@dataclass
class FetchOptions:
    """Tham số cho một lần chạy `fetch_dataset` (CLI truyền vào từ `youtube-fetch`)."""

    channel: str
    out: Path
    limit: int = 0
    demo: bool = False
    langs: tuple[str, ...] = ("vi", "vi-orig")
    dialect: str = "north"
    min_duration: float = 1.0
    max_duration: float = 20.0
    max_gap: float = 0.35
    max_chars: int = 220
    min_video_duration: float = 0.0
    max_video_duration: float = 0.0
    title_filter: str | None = None
    keep_wav: bool = True
    dedupe: bool = True
    min_rms_db: float = -45.0
    max_fallback_ratio: float = 0.34
    min_caption_density: float = 2.0  # ký tự/giây tối thiểu (video nhiều nhạc → bỏ)
    respell_codes: bool = True  # tách chữ-số dính ("v2") và đánh vần mã ngắn ("MP" → "em pê")
    overrides: dict[str, str] = field(default_factory=dict)  # "mặt chữ" → cách đọc
    overrides_source: str | None = None  # đường dẫn file overrides (ghi vào dataset card)
    sample_rate: int = 24000
    license: str = "CC BY 4.0"
    sleep: float = 1.0
    retry_delay: float = 6.0  # giây × lần thử (nội bộ; test đặt 0 để chạy nhanh)
    demo_candidates: int = 5  # demo thử lần lượt bao nhiêu video ngắn nhất
    recut: bool = False  # xoá kết quả cũ rồi cắt lại từ raw/subs cache (không tải)
    dry_run: bool = False
    refresh: bool = False
    ytdlp: list[str] | None = None
    ffmpeg: str | None = None


def find_ytdlp() -> list[str]:
    """Tìm lệnh yt-dlp: env `VIFASTTTS_YTDLP` → PATH → `python -m yt_dlp`."""
    override = os.environ.get("VIFASTTTS_YTDLP")
    if override:
        return shlex.split(override)
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]
    probe = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--version"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", "yt_dlp"]
    raise RuntimeError(
        "Không tìm thấy yt-dlp. Cài bằng: pip install 'vifasttts[youtube]' "
        "hoặc đặt VIFASTTTS_YTDLP=/đường/dẫn/yt-dlp"
    )


def find_ffmpeg() -> str | None:
    """Tìm ffmpeg: env `VIFASTTTS_FFMPEG` → PATH → binary kèm của imageio-ffmpeg."""
    override = os.environ.get("VIFASTTTS_FFMPEG")
    if override:
        return override
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001 - thiếu imageio-ffmpeg không phải lỗi cứng
        return None


def _run(command: list[str], *, log: LogFn, retries: int = 3, retry_delay: float = 6.0) -> str:
    """Chạy lệnh ngoài, trả stdout; lỗi tạm thời (429/5xx/timeout) thì thử lại.

    YouTube hay trả `HTTP Error 429` khi tải phụ đề liên tục; backoff 6s, 12s
    thường đủ để vượt qua mà không cần đổi IP. `retry_delay=0` khi test.
    """
    transient = ("HTTP Error 429", "HTTP Error 5", "timed out", "Temporary failure",
                 "Read timed out", "Connection reset")
    for attempt in range(1, retries + 1):
        log(f"$ {' '.join(shlex.quote(part) for part in command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return result.stdout
        tail = "\n".join(result.stderr.strip().splitlines()[-10:])
        if attempt < retries and any(marker in result.stderr for marker in transient):
            delay = retry_delay * attempt
            log(f"    lỗi tạm thời, thử lại sau {delay:.0f}s ({attempt}/{retries})…")
            if delay > 0:
                time.sleep(delay)
            continue
        raise RuntimeError(f"Lệnh thất bại (mã {result.returncode}): {tail}")
    raise AssertionError("không thể tới đây")  # pragma: no cover


def list_videos(options: FetchOptions, ytdlp: list[str], *, log: LogFn = print) -> list[dict[str, Any]]:
    """Liệt kê video của kênh/playlist, có cache `.videos.json` trong thư mục ra.

    Trả danh sách `{"id", "title", "duration", "url"}` đã lọc theo độ dài video
    và `title_filter`; `demo=True` chọn một video ngắn (60–900 giây).
    """
    cache_path = options.out / ".videos.json"
    entries: list[dict[str, Any]] | None = None
    if cache_path.exists() and not options.refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("channel") == options.channel:
            entries = cached.get("entries") or []

    if entries is None:
        command = [*ytdlp, "--no-warnings", "--flat-playlist", "-J"]
        if options.limit > 0 and not options.demo and not options.title_filter:
            command += ["--playlist-end", str(options.limit)]
        command.append(options.channel)
        data = json.loads(_run(command, log=log, retry_delay=options.retry_delay))
        raw_entries = data.get("entries")
        entries = list(raw_entries) if raw_entries else [data]
        options.out.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"channel": options.channel, "entries": entries}, ensure_ascii=False),
            encoding="utf-8",
        )

    picked: list[dict[str, Any]] = []
    for entry in entries:
        video_id = entry.get("id")
        if not video_id:
            continue
        duration = float(entry.get("duration") or 0.0)
        if duration and options.min_video_duration and duration < options.min_video_duration:
            continue
        if duration and options.max_video_duration and duration > options.max_video_duration:
            continue
        title = str(entry.get("title") or "")
        if options.title_filter and not re.search(options.title_filter, title, re.IGNORECASE):
            continue
        picked.append({
            "id": str(video_id),
            "title": title,
            "duration": duration,
            "url": str(entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"),
        })

    if options.demo:
        # Demo: thử lần lượt các video ngắn nhất cho tới khi có ít nhất 1 clip.
        picked = sorted(picked, key=lambda video: video["duration"] or 1e18)
        picked = picked[: max(1, options.demo_candidates)]
    elif options.limit > 0:
        picked = picked[: options.limit]
    return picked


def _subtitle_file(subs_dir: Path, video_id: str, langs: tuple[str, ...]) -> Path | None:
    """Chọn file phụ đề ưu tiên theo thứ tự ngôn ngữ trong `langs`."""
    for lang in langs:
        candidate = subs_dir / f"{video_id}.{lang}.json3"
        if candidate.exists():
            return candidate
    fallback = sorted(subs_dir.glob(f"{video_id}.*.json3"))
    return fallback[0] if fallback else None


def download_subtitles(
    video: dict[str, Any], options: FetchOptions, ytdlp: list[str], *, log: LogFn = print,
) -> list[Cue] | None:
    """Tải phụ đề json3 của một video và trả danh sách Cue (None nếu không có)."""
    subs_dir = options.out / "subs"
    subs_dir.mkdir(parents=True, exist_ok=True)
    if _subtitle_file(subs_dir, video["id"], options.langs) is None:
        command = [
            *ytdlp, "--no-warnings", "--skip-download", "--write-subs", "--write-auto-subs",
            "--sub-langs", ",".join(options.langs), "--sub-format", "json3",
            "--retries", "3", "--extractor-retries", "3",
            "--sleep-requests", str(options.sleep),
            "-o", str(subs_dir / "%(id)s"), video["url"],
        ]
        _run(command, log=log, retry_delay=options.retry_delay)
    path = _subtitle_file(subs_dir, video["id"], options.langs)
    if path is None:
        return None
    return load_subtitles(path)


def download_audio(
    video: dict[str, Any], options: FetchOptions, ytdlp: list[str],
    ffmpeg: str | None, *, log: LogFn = print,
) -> Path:
    """Tải audio gốc của video thành WAV mono `sample_rate` (PCM 16-bit)."""
    raw_dir = options.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{video['id']}.wav"
    if target.exists():
        return target
    if ffmpeg is None:
        raise RuntimeError(
            "Không tìm thấy ffmpeg. Cài bằng: pip install 'vifasttts[youtube]' "
            "(imageio-ffmpeg) hoặc brew install ffmpeg"
        )
    command = [
        *ytdlp, "--no-warnings", "-f", "bestaudio", "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ac 1 -ar {options.sample_rate} -c:a pcm_s16le",
        "--retries", "3", "--extractor-retries", "3", "--fragment-retries", "3",
        "--sleep-requests", str(options.sleep), "--sleep-interval", str(options.sleep),
        "--ffmpeg-location", ffmpeg,
        "-o", str(raw_dir / "%(id)s.%(ext)s"), video["url"],
    ]
    _run(command, log=log, retry_delay=options.retry_delay)
    if not target.exists():
        candidates = sorted(raw_dir.glob(f"{video['id']}.*.wav"))
        if not candidates:
            raise RuntimeError(f"yt-dlp không tạo WAV cho {video['id']}")
        target = candidates[0]
    return target


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Đọc WAV PCM (ưu tiên 16-bit mono) thành mảng float32 trong [-1, 1]."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"Chỉ hỗ trợ WAV PCM 16-bit, nhận {width * 8}-bit: {path}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def write_wav_mono(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Ghi mảng float32 [-1, 1] thành WAV PCM 16-bit mono."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _normalized_key(text: str) -> str:
    """Khoá chống trùng: chữ thường, bỏ dấu câu, gộp khoảng trắng."""
    lowered = re.sub(r"[^\wÀ-ỹĐđ ]", " ", text.lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _fallback_items(text: str, options: NormalizerOptions) -> tuple[float, list[str]]:
    """Tỉ lệ fallback + danh sách token lạ (OOV) của một câu đã chuẩn hoá."""
    items = parse_text(text, options=options)
    if not items:
        return 1.0, []
    fallback = [item.raw for item in items if item.status is ParseStatus.FALLBACK_REQUIRED]
    return len(fallback) / len(items), fallback


_GLUE_RE = re.compile(r"(?<=[A-Za-zÀ-ỹĐđ])(?=\d)|(?<=\d)(?=[A-Za-zÀ-ỹĐđ])")
_LETTERS_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+")
_VERSION_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")  # 1.13.1 / 24.19.0 (số phiên bản)


def _respell_codes(text: str, options: NormalizerOptions) -> str:
    """Làm sạch mã chữ-số cho dataset: "Chương2" → "Chương hai", "MP3" → "em pê ba".

    Chỉ dùng khi làm dataset (frontend giữ nguyên hành vi thận trọng với mã phiên
    bản). Token ≥3 chữ cái viết thường vẫn giữ nguyên để không đánh vần nhầm các
    từ vay/viết sai chính tả.
    """
    text = _GLUE_RE.sub(" ", text)
    # Số phiên bản nhiều tầng (1.13.1) đọc thành dãy số; số thập phân thật đã được
    # normalizer xử lý ("3.14" → "ba phẩy một bốn") trước khi tới đây.
    text = _VERSION_DOT_RE.sub(" ", text)
    # Dấu chấm dính số ở rìa ("0.1.0." hoặc ".5") cũng tách để đọc được nốt.
    text = re.sub(r"(?<=\d)\.(?=\D|$)", " ", text)
    text = re.sub(r"(?<!\w)\.(?=\d)", " ", text)
    text = normalize_text(text, options=options)

    def _spell(match: re.Match[str]) -> str:
        token = match.group(0)
        if safe_parse_syllable(token) is not None:
            return token
        reading = resolve_oov_token(token)
        if reading is not None and all(safe_parse_syllable(word) for word in reading.split()):
            return reading
        spellable = all(char in LETTER_READINGS for char in token.lower())
        if spellable and len(token) <= 4 and (token.isupper() or len(token) <= 2):
            spelled = spell_token(token)
            if spelled:
                return spelled
        return token

    return _LETTERS_RE.sub(_spell, text)


def _prepare_dataset_text(raw: str, options: NormalizerOptions, respell_codes: bool) -> str:
    """Chuẩn hoá text phụ đề thành câu đọc được cho dataset (số, OOV, mã chữ-số)."""
    text = normalize_text(raw, options=options)
    if respell_codes:
        text = _respell_codes(text, options)
    return re.sub(r"\s+", " ", text).strip()


def cut_clips(
    video: dict[str, Any],
    cues: list[Cue],
    samples: np.ndarray,
    sample_rate: int,
    options: FetchOptions,
    normalizer_options: NormalizerOptions,
    seen: set[str],
    stats: dict[str, int],
    oov_counter: Counter[str],
    *,
    log: LogFn = print,
) -> list[dict[str, Any]]:
    """Cắt các cue đã gộp thành clip WAV sạch, trả về bản ghi cho `clips.jsonl`."""
    merged = merge_cues(
        cues,
        max_gap=options.max_gap,
        max_duration=options.max_duration,
        max_chars=options.max_chars,
    )
    audio_dir = options.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    total_seconds = samples.size / sample_rate
    records: list[dict[str, Any]] = []
    for index, cue in enumerate(merged, 1):
        start = max(0.0, cue.start - PAD_START)
        end = min(total_seconds, cue.end + PAD_END)
        if end - start < options.min_duration:
            stats["skipped_short"] += 1
            continue
        segment = samples[round(start * sample_rate):round(end * sample_rate)]
        if segment.size == 0:
            stats["skipped_short"] += 1
            continue
        rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(np.square(segment)))) + 1e-12)
        if rms_db < options.min_rms_db:
            stats["skipped_silence"] += 1
            continue
        text = _prepare_dataset_text(cue.text, normalizer_options, options.respell_codes)
        if not text:
            stats["skipped_short"] += 1
            continue
        key = _normalized_key(text)
        if options.dedupe and key in seen:
            stats["skipped_duplicate"] += 1
            continue
        ratio, oov_tokens = _fallback_items(text, normalizer_options)
        if ratio > options.max_fallback_ratio:
            stats["skipped_fallback"] += 1
            continue
        for token in oov_tokens:
            oov_counter[token] += 1
        clip_id = f"{video['id']}_{index:04d}"
        clip_path = audio_dir / f"{clip_id}.wav"
        write_wav_mono(clip_path, fade(segment.astype(np.float32), sample_rate), sample_rate)
        seen.add(key)
        stats["clips"] += 1
        records.append({
            "id": clip_id,
            "video": video["id"],
            "video_title": video["title"],
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "rms_db": round(rms_db, 2),
            "fallback_ratio": round(ratio, 4),
            "text": text,
            "audio_path": str(clip_path),
        })
    log(f"    {len(merged)} câu → {len(records)} clip hợp lệ")
    return records


def write_video_srt(video: dict[str, Any], cues: list[Cue], options: FetchOptions) -> Path:
    """Ghi lại SRT đã làm sạch + gộp câu cho một video (mở cho mọi người dùng)."""
    merged = merge_cues(
        cues,
        max_gap=options.max_gap,
        max_duration=options.max_duration,
        max_chars=options.max_chars,
    )
    path = options.out / "subs" / f"{video['id']}.srt"
    path.write_text(format_srt(merged), encoding="utf-8")
    return path


def write_metadata(options: FetchOptions, records: list[dict[str, Any]]) -> Path:
    """Ghi `metadata.csv` (`id|text`) — đúng định dạng `build-manifest` cần."""
    path = options.out / "metadata.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        for record in records:
            writer.writerow([record["id"], record["text"]])
    return path


def _command_hint(options: FetchOptions) -> str:
    """Câu lệnh gợi ý để tái tạo dataset (ghi vào dataset card)."""
    parts = [f'vifasttts youtube-fetch "{options.channel}"', f"--out {options.out}"]
    if options.demo:
        parts.append("--demo")
    if options.limit:
        parts.append(f"--limit {options.limit}")
    if options.overrides_source:
        parts.append(f"--overrides {options.overrides_source}")
    parts.append(f'--license "{options.license}"')
    return " ".join(parts)


def write_dataset_card(
    options: FetchOptions, videos: list[dict[str, Any]], records: list[dict[str, Any]],
    stats: dict[str, Any],
) -> Path:
    """Ghi `dataset_card.md` + `LICENSE-DATASET.txt` mô tả nguồn, giấy phép, cách tái tạo."""
    total_seconds = sum(float(r["duration"]) for r in records)
    hours = total_seconds / 3600.0
    license_link = _LICENSE_LINKS.get(options.license.strip().lower(), "")
    license_line = f"{options.license} — {license_link}" if license_link else options.license
    coverage = stats.get("coverage", {})
    oov_lines: list[str] = []
    if stats.get("oov_top"):
        preview = ", ".join(f"`{token}`×{count}" for token, count in stats["oov_top"][:7])
        oov_lines = [f"- Từ lạ (OOV) nhiều nhất: {preview} — điền vào overrides rồi chạy lại", ""]
    lines = [
        f"# Dataset: {options.channel}",
        "",
        f"- **Nguồn:** kênh YouTube của chủ dự án — {options.channel}",
        f"- **Giấy phép dữ liệu:** {license_line}",
        f"- **Sinh lúc:** {stats.get('generated_at', '')} bằng ViFastTts {__version__}",
        f"- **Tái tạo:** `{_command_hint(options)}`",
        "",
        "## Thống kê",
        "",
        f"- Video đã xử lý: **{stats.get('videos_processed', 0)}** / {stats.get('videos_listed', 0)} liệt kê",
        (
            f"- Clip: **{stats.get('clips', 0)}** · tổng **{hours:.2f} giờ** audio · TB "
            f"{total_seconds / max(1, len(records)):.2f} giây/clip"
        ),
        f"- Âm tiết: {coverage.get('syllables', 0)} · fallback: {coverage.get('fallback', 0)}",
        (
            f"- Video không có phụ đề: {stats.get('videos_no_subs', 0)} · phụ đề quá thưa: "
            f"{stats.get('videos_sparse_subs', 0)} · lỗi: {stats.get('videos_error', 0)}"
        ),
        (
            f"- Đã bỏ: {stats.get('skipped_short', 0)} ngắn · {stats.get('skipped_silence', 0)} im lặng · "
            f"{stats.get('skipped_duplicate', 0)} trùng · {stats.get('skipped_fallback', 0)} fallback cao"
        ),
        *oov_lines,
        "",
        "## Video nguồn",
        "",
        *[f"- `{video['id']}` — {video['title']} ({video['duration']:.0f}s)" for video in videos],
        "",
        "## Cấu trúc",
        "",
        "```text",
        f"{options.out}/",
        "  audio/<video_id>_<NNNN>.wav   # clip 24 kHz mono PCM16, có fade đầu/cuối",
        "  subs/<video_id>.srt           # phụ đề sạch, đã gộp câu (mở cho mọi người dùng)",
        "  raw/<video_id>.wav            # audio gốc nếu --keep-wav (để căn chỉnh lại)",
        "  metadata.csv                  # id|text",
        "  manifest.jsonl                # text + âm tiết đã parse (đưa thẳng vào train)",
        "  stats.json                    # số liệu chi tiết",
        "  .done.json                    # video đã xử lý (chạy lại sẽ bỏ qua)",
        "```",
        "",
        "## Lưu ý chất lượng",
        "",
        "- Thời lượng clip là **cấp câu** theo phụ đề YouTube; muốn căn cấp âm tiết hãy chạy",
        "  forced alignment (MFA) — xem `docs/TRAINING.md` và `docs/DATA_PIPELINE.md`.",
        "- Phụ đề tự động có thể sai chính tả/nhận dạng; các clip fallback cao đã bị lọc bỏ.",
        "- Audio đã chuẩn hoá 24 kHz mono PCM16; text đã chuẩn hoá số/ngày/giờ theo vùng miền.",
        f"- Danh sách video nguồn nằm trong `.videos.json` ({len(videos)} mục).",
        "",
    ]
    card_path = options.out / "dataset_card.md"
    card_path.write_text("\n".join(lines), encoding="utf-8")

    license_path = options.out / "LICENSE-DATASET.txt"
    license_path.write_text(
        "\n".join([
            f"Giấy phép dữ liệu: {options.license}",
            f"Toàn văn: {license_link}" if license_link else "",
            "",
            (
                "Dữ liệu audio + phụ đề trong thư mục này do chủ kênh YouTube công bố mở; "
                "ai dùng cũng được, chỉ cần tuân theo giấy phép ở trên."
            ),
            f"Nguồn: {options.channel}",
            "",
        ]),
        encoding="utf-8",
    )
    return card_path


def _write_stats(options: FetchOptions, stats: dict[str, Any]) -> Path:
    """Ghi `stats.json` (kèm coverage) để theo dõi chất lượng giữa các lần chạy."""
    path = options.out / "stats.json"
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _clear_derived(options: FetchOptions, *, log: LogFn) -> None:
    """Xoá kết quả đã sinh (audio/clip/manifest/card) để cắt lại từ cache raw/ + subs/."""
    for name in ("audio", "clips.jsonl", ".done.json", "manifest.jsonl", "metadata.csv",
                 "stats.json", "oov_report.json", "dataset_card.md", "LICENSE-DATASET.txt"):
        path = options.out / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for srt in (options.out / "subs").glob("*.srt"):
        srt.unlink()
    log("--recut: đã xoá kết quả cũ; cắt lại từ raw/ + subs/ (không tải lại)")


def fetch_dataset(options: FetchOptions, *, log: LogFn = print) -> dict[str, Any]:
    """Chạy toàn bộ luồng lấy dữ liệu YouTube → dataset sạch, trả dict thống kê.

    Có thể chạy lại an toàn (idempotent): video đã xử lý sẽ được bỏ qua, danh
    sách video lấy từ cache `.videos.json`; clip trùng chữ bị lọc (tắt bằng
    `dedupe=False`).
    """
    options.out.mkdir(parents=True, exist_ok=True)
    ytdlp = options.ytdlp if options.ytdlp is not None else find_ytdlp()
    ffmpeg = options.ffmpeg if options.ffmpeg is not None else find_ffmpeg()
    # Override "mặt chữ → cách đọc": khớp chính xác và không phân biệt hoa thường.
    normalizer_options = NormalizerOptions(
        dialect=Dialect(options.dialect),
        exact_overrides=dict(options.overrides),
        english_overrides={key.lower(): value for key, value in options.overrides.items()},
    )
    oov_counter: Counter[str] = Counter()
    if options.recut and not options.dry_run:
        _clear_derived(options, log=log)
    log(f"yt-dlp: {' '.join(ytdlp)}")
    log(f"ffmpeg: {ffmpeg or 'KHÔNG CÓ (cài imageio-ffmpeg hoặc brew install ffmpeg)'}")

    videos = list_videos(options, ytdlp, log=log)
    candidates = sorted(videos, key=lambda v: v["duration"] or 1e18)[: max(1, options.demo_candidates)] \
        if options.demo else videos
    stats: dict[str, Any] = {
        "channel": options.channel,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "license": options.license,
        "videos_listed": len(videos),
        "videos_processed": 0,
        "videos_skipped_done": 0,
        "videos_no_subs": 0,
        "videos_sparse_subs": 0,
        "videos_error": 0,
        "clips": 0,
        "skipped_short": 0,
        "skipped_silence": 0,
        "skipped_duplicate": 0,
        "skipped_fallback": 0,
    }
    log(f"Danh sách: {len(videos)} video")
    for video in candidates:
        log(f"  - {video['id']} ({video['duration']:.0f}s) {video['title'][:70]}")
    if options.dry_run:
        log("--dry-run: chỉ liệt kê, không tải.")
        _write_stats(options, stats)
        return stats

    clips_path = options.out / "clips.jsonl"
    done_path = options.out / ".done.json"
    records: list[dict[str, Any]] = []
    if clips_path.exists():
        records = [
            json.loads(line) for line in clips_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    done_ids: set[str] = set()
    if done_path.exists():
        done_ids = set(json.loads(done_path.read_text(encoding="utf-8")).get("videos", []))
    seen = {_normalized_key(str(record.get("text", ""))) for record in records}
    stats["clips"] = len(records)  # gồm clip có sẵn từ lần chạy trước (resume)

    for order, video in enumerate(candidates, 1):
        if video["id"] in done_ids:
            stats["videos_skipped_done"] += 1
            log(f"[{order}/{len(candidates)}] {video['id']}: đã có, bỏ qua")
            continue
        log(f"[{order}/{len(candidates)}] {video['id']}: {video['title'][:70]}")
        try:
            cues = download_subtitles(video, options, ytdlp, log=log)
            if not cues:
                stats["videos_no_subs"] += 1
                log("    không có phụ đề (cần phụ đề để cắt đúng câu) → bỏ qua")
                continue
            duration = float(video["duration"] or (cues[-1].end if cues else 1.0))
            density = sum(len(cue.text) for cue in cues) / max(1.0, duration)
            if density < options.min_caption_density:
                stats["videos_sparse_subs"] += 1
                log(f"    phụ đề quá thưa ({density:.1f} ký tự/s — video nhiều nhạc/ít lời) → bỏ qua")
                continue
            audio_path = download_audio(video, options, ytdlp, ffmpeg, log=log)
            samples, sample_rate = read_wav_mono(audio_path)
            log(f"    audio {samples.size / sample_rate:.0f}s @ {sample_rate} Hz")
            write_video_srt(video, cues, options)
            # Chạy lại video này thì thay bản ghi cũ (idempotent).
            records = [record for record in records if record.get("video") != video["id"]]
            video_records = cut_clips(
                video, cues, samples, sample_rate, options, normalizer_options,
                seen, stats, oov_counter, log=log,
            )
            records.extend(video_records)
            stats["videos_processed"] += 1
            done_ids.add(video["id"])
            done_path.write_text(
                json.dumps({"videos": sorted(done_ids)}, ensure_ascii=False), encoding="utf-8",
            )
            if not options.keep_wav:
                audio_path.unlink(missing_ok=True)
            if options.demo and video_records:
                log("    demo: đã có clip hợp lệ → dừng (dùng lệnh không --demo để chạy cả kênh)")
                break
        except KeyboardInterrupt:
            raise
        except Exception as error:  # noqa: BLE001 - một video lỗi không làm chết cả kênh
            stats["videos_error"] += 1
            log(f"    LỖI: {error}")

    metadata_path = write_metadata(options, records)
    clips_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    oov_top = oov_counter.most_common(30)
    stats["oov_top"] = oov_top
    oov_path = options.out / "oov_report.json"
    oov_path.write_text(json.dumps({
        "note": "Điền cách đọc cho các từ này vào overrides JSON rồi chạy lại với --overrides.",
        "tokens": [{"token": token, "count": count} for token, count in oov_top],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = options.out / "manifest.jsonl"
    build_manifest(metadata_path, options.out / "audio", manifest_path)
    validation = validate_manifest(manifest_path, options.out / "audio")
    coverage = coverage_report(manifest_path)
    stats["coverage"] = coverage
    stats["manifest_issues"] = validation["issues"]
    write_dataset_card(options, candidates, records, stats)
    _write_stats(options, stats)

    total_seconds = sum(float(record["duration"]) for record in records)
    average = total_seconds / max(1, len(records))
    log("")
    log("=== Tổng kết ===")
    log(f"Video: {stats['videos_processed']} xử lý · {stats['videos_skipped_done']} đã có · "
        f"{stats['videos_no_subs']} không phụ đề · {stats['videos_sparse_subs']} phụ đề thưa · "
        f"{stats['videos_error']} lỗi")
    log(f"Clip: {stats['clips']} · {total_seconds / 3600.0:.2f} giờ · TB {average:.2f}s")
    log(f"Bỏ: {stats['skipped_short']} ngắn · {stats['skipped_silence']} im lặng · "
        f"{stats['skipped_duplicate']} trùng · {stats['skipped_fallback']} fallback cao")
    log(f"Manifest: {manifest_path} ({validation['total']} dòng, "
        f"{len(validation['issues'])} vấn đề)")
    log(f"Coverage: {coverage['syllables']} âm tiết, {coverage['fallback']} fallback")
    if oov_top:
        preview = ", ".join(f"{token}×{count}" for token, count in oov_top[:10])
        log(f"Từ lạ (OOV) nhiều nhất: {preview}")
        log(f"  → điền cách đọc vào overrides rồi chạy lại với --overrides (xem {oov_path.name})")
    log(f"Dataset card + license: {options.out / 'dataset_card.md'}")
    return stats

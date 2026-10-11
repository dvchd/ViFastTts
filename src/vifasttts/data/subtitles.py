"""Đọc phụ đề (json3/vtt/srt), làm sạch và gộp thành câu cho dataset YouTube.

Thuần stdlib, không cần mạng/torch nên test được offline. Điểm quan trọng:
phụ đề tự động của YouTube là caption "cuộn" — mỗi event chứa lại phần chữ đã
hiện trước đó rồi thêm chữ mới. `normalize_cues` tự cắt phần lặp đó để mỗi câu
chỉ còn một bản ghi với thời gian đúng; `merge_cues` gộp các câu gần nhau thành
một clip dài `min_duration..max_duration` giây.
"""

from __future__ import annotations

import html
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

_TS_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[,.](\d{1,3}))?")
_TAG_RE = re.compile(r"<[^>]+>")
_VTT_KARAOKE_RE = re.compile(r"\{\\[^}]*\}")
_MARKER_RE = re.compile(r"[\[(]([^\])]*)[\])]")
_WORD_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]")
_SPACE_RE = re.compile(r"[\s\u200b]+")
_SPEAKER_RE = re.compile(r"^\s*(>>+|-{1,2}\s*|•\s*)")
_SENTENCE_END = (".", "!", "?", "…", ":", ";")

# Nhãn "tiếng động" phổ biến của phụ đề YouTube — bỏ nguyên nhóm thay vì đọc lên.
_NOISE_WORDS = {
    "music", "âm nhạc", "am nhac", "nhạc", "nhac", "beat", "instrumental",
    "laughter", "laughs", "tiếng cười", "tieng cuoi", "cười", "cuoi",
    "applause", "vỗ tay", "vo tay", "clapping",
    "inaudible", "không nghe rõ", "khong nghe ro", "silence", "im lặng", "im lang",
    "noise", "tiếng ồn", "tieng on", "sigh", "thở dài", "breathes", "music playing",
}


@dataclass(frozen=True)
class Cue:
    """Một đoạn thoại đã sạch: khoảng thời gian [start, end] (giây) + chữ."""

    start: float
    end: float
    text: str


def parse_timestamp(value: str) -> float:
    """Đổi mốc thời gian SRT/VTT ("HH:MM:SS,mmm" hoặc "MM:SS.mmm") thành giây.

    Phần đuôi như `align:start position:0%` của VTT bị bỏ qua; chuỗi không có
    mốc thời gian trả về 0.0 để caller tự xử lý.
    """
    match = _TS_RE.match(value.strip())
    if match is None:
        return 0.0
    hours = float(match.group(1) or 0)
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    millis = float((match.group(4) or "0").ljust(3, "0"))
    return hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0


def clean_caption(text: str) -> str:
    """Bỏ tag/thời gian nội tuyến và nhãn tiếng động, chỉ giữ phần thoại.

    Giữ nguyên chữ trong ngoặc nếu đó là nội dung thật (ví dụ "(phần 1)");
    bỏ nhóm ngoặc thuần ký hiệu hoặc chứa từ chỉ tiếng động ([Music], (nhạc)...).
    """
    text = _TAG_RE.sub(" ", text)          # <c>, <00:00:01.234>, <i>...
    text = _VTT_KARAOKE_RE.sub(" ", text)  # {\an8} của caption karaoke
    text = html.unescape(text)             # &amp;, &nbsp;...
    text = text.replace("♪", " ").replace("\u266a", " ")

    def _drop_marker(match: re.Match[str]) -> str:
        inner = match.group(1).strip().lower()
        if not inner or not _WORD_RE.search(inner):
            return " "
        if any(word in inner for word in _NOISE_WORDS):
            return " "
        return match.group(0)

    text = _MARKER_RE.sub(_drop_marker, text)
    lines = [_SPEAKER_RE.sub("", line) for line in text.splitlines()]
    text = _SPACE_RE.sub(" ", " ".join(lines)).strip()
    if not _WORD_RE.search(text):
        return ""
    return text


def _blocks(text: str) -> list[list[str]]:
    """Tách văn bản phụ đề thành các block (mỗi block là list dòng không rỗng)."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = re.split(r"\n\s*\n", normalized.strip())
    return [[line for line in block.splitlines() if line.strip()] for block in raw_blocks]


def parse_srt(text: str) -> list[Cue]:
    """Đọc SRT thành danh sách Cue; bỏ block thiếu mốc thời gian hoặc rỗng chữ."""
    cues: list[Cue] = []
    for lines in _blocks(text):
        if "-->" not in lines[0]:
            if len(lines) < 2 or "-->" not in lines[1]:
                continue
            lines = lines[1:]
        start_raw, end_raw = (part.strip() for part in lines[0].split("-->")[:2])
        body = clean_caption(" ".join(lines[1:]))
        if not body:
            continue
        start, end = parse_timestamp(start_raw), parse_timestamp(end_raw)
        cues.append(Cue(start, max(end, start), body))
    return normalize_cues(cues)


def parse_vtt(text: str) -> list[Cue]:
    """Đọc WebVTT (kể cả auto-caption cuộn của YouTube) thành danh sách Cue."""
    cues: list[Cue] = []
    for lines in _blocks(text):
        head = lines[0].strip().upper()
        if head.startswith(("WEBVTT", "NOTE", "STYLE", "REGION", "KIND", "LANGUAGE")):
            continue
        if "-->" not in lines[0]:
            if len(lines) < 2 or "-->" not in lines[1]:
                continue
            lines = lines[1:]
        start_raw, end_raw = (part.strip() for part in lines[0].split("-->")[:2])
        body = clean_caption(" ".join(lines[1:]))
        if not body:
            continue
        start, end = parse_timestamp(start_raw), parse_timestamp(end_raw)
        cues.append(Cue(start, max(end, start), body))
    return normalize_cues(cues)


def parse_json3(text: str) -> list[Cue]:
    """Đọc phụ đề json3 của YouTube (định dạng chuẩn để lấy timing chính xác).

    Mỗi event có `tStartMs`, `dDurationMs` và các `segs`. Với caption tự động,
    event sau chứa lại text của event trước; `normalize_cues` sẽ cắt phần lặp.
    Bỏ event rỗng/ký tự xuống dòng và event chỉ chứa nhãn tiếng động.
    """
    data = json.loads(text)
    cues: list[Cue] = []
    for event in data.get("events", []):
        segs = event.get("segs") or []
        body = clean_caption("".join(str(seg.get("utf8", "")) for seg in segs))
        if not body:
            continue
        start = float(event.get("tStartMs", 0)) / 1000.0
        duration = float(event.get("dDurationMs", 0)) / 1000.0
        cues.append(Cue(start, start + max(duration, 0.0), body))
    return normalize_cues(cues)


def load_subtitles(path: str | Path) -> list[Cue]:
    """Đọc file phụ đề theo đuôi `.srt` / `.vtt` / `.json3` và chuẩn hoá."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".srt", ".vtt", ".json3"):
        raise ValueError(f"Không hỗ trợ định dạng phụ đề: {path.name}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".srt":
        return parse_srt(text)
    if suffix == ".vtt":
        return parse_vtt(text)
    return parse_json3(text)


def _close_open_ends(cues: list[Cue]) -> list[Cue]:
    """Điền end time cho cue thiếu `dDurationMs` bằng start của cue kế tiếp."""
    closed: list[Cue] = []
    for index, cue in enumerate(cues):
        if cue.end <= cue.start + 0.02 and index + 1 < len(cues):
            next_start = cues[index + 1].start
            end = next_start if next_start > cue.start + 0.02 else cue.start + 0.5
            cue = replace(cue, end=min(end, cue.start + 30.0))
        closed.append(cue)
    return closed


def normalize_cues(cues: list[Cue]) -> list[Cue]:
    """Chuẩn hoá timeline: sắp xếp, cắt phần caption cuộn lặp, gộp bản trùng.

    Với caption tự động kiểu "lớn dần" (event sau chứa lại chữ của event trước),
    phần chữ mới được tách ra và gán thời gian bắt đầu của chính event đó.
    """
    ordered = sorted((cue for cue in cues if cue.text.strip()), key=lambda c: c.start)
    out: list[Cue] = []
    for cue in ordered:
        text = cue.text
        if out:
            previous = out[-1]
            if text == previous.text:
                out[-1] = replace(previous, end=max(previous.end, cue.end))
                continue
            if text.startswith(previous.text):
                grown = text[len(previous.text):].strip()
                if not grown:
                    out[-1] = replace(previous, end=max(previous.end, cue.end))
                    continue
                text = grown
                cue = Cue(cue.start, cue.end, grown)
            elif cue.end <= previous.end + 0.02:
                # Cue nằm trọn trong cue trước nhưng khác chữ (hiếm) → bỏ.
                continue
        out.append(Cue(cue.start, cue.end, text))
    return _close_open_ends(out)


def _split_long_cue(cue: Cue, max_duration: float) -> list[Cue]:
    """Chia cue quá dài thành nhiều phần theo thời gian, chia chữ tỉ lệ theo phần."""
    duration = cue.end - cue.start
    parts = max(1, math.ceil(duration / max_duration))
    words = cue.text.split()
    chunks: list[Cue] = []
    for index in range(parts):
        start = cue.start + duration * index / parts
        end = cue.start + duration * (index + 1) / parts
        left = round(len(words) * index / parts)
        right = round(len(words) * (index + 1) / parts)
        text = " ".join(words[left:right]).strip()
        if text:
            chunks.append(Cue(start, end, text))
    return chunks


def merge_cues(
    cues: list[Cue],
    *,
    max_gap: float = 0.35,
    max_duration: float = 20.0,
    max_chars: int = 220,
    break_gap: float = 0.18,
) -> list[Cue]:
    """Gộp cue liền kề thành câu dài vừa phải để cắt clip.

    Gộp khi khoảng nghỉ ≤ `max_gap`, clip kết quả ≤ `max_duration` giây và
    ≤ `max_chars` ký tự. Dừng gộp nếu câu trước đã hết câu (dấu . ! ? …) và
    khoảng nghỉ ≥ `break_gap` — tránh dính hai câu vào một clip.
    """
    prepared: list[Cue] = []
    for cue in cues:
        if cue.end - cue.start > max_duration:
            prepared.extend(_split_long_cue(cue, max_duration))
        else:
            prepared.append(cue)

    groups: list[list[float | str]] = []
    for cue in prepared:
        text = cue.text.strip()
        if not text:
            continue
        if not groups:
            groups.append([cue.start, cue.end, text])
            continue
        start, end, current_text = groups[-1]
        gap = cue.start - float(end)
        merged = f"{current_text} {text}"
        sentence_done = str(current_text).rstrip().endswith(_SENTENCE_END)
        fits = (
            gap <= max_gap
            and cue.end - float(start) <= max_duration
            and len(merged) <= max_chars
        )
        if fits and not (gap >= break_gap and sentence_done):
            groups[-1] = [start, max(float(end), cue.end), merged]
        else:
            groups.append([cue.start, cue.end, text])

    return [Cue(float(start), float(end), str(text)) for start, end, text in groups]


def format_timestamp(seconds: float) -> str:
    """Đổi giây thành mốc SRT "HH:MM:SS,mmm"."""
    milliseconds = max(0, round(seconds * 1000.0))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_srt(cues: list[Cue]) -> str:
    """Xuất danh sách Cue thành nội dung SRT (UTF-8, kết thúc bằng dòng trống)."""
    lines: list[str] = []
    for index, cue in enumerate(cues, 1):
        lines.append(str(index))
        lines.append(f"{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines)

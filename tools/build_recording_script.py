"""tools/build_recording_script.py — kịch bản thu bổ sung v2 (A0 v13, Việc 9c).

Thay đổi so với v1:
  - Bỏ "đọc 2 lần/câu" (đọc 1 lần, dùng thời gian cho câu mới).
  - 16 khuôn câu ⇒ mỗi khuôn ≤ 20 % số câu.
  - 600 âm tiết hiếm × 2 câu (khuôn khác nhau) + 50 câu cầu nối lấy từ transcript corpus
    + 56 câu cặp tối thiểu ⇒ mục tiêu 60–90 phút.
  - Xuất `docs/rare_syllables_v1.txt`: âm tiết · số lần · 1–2 từ ghép thông dụng chứa nó (nếu có).

Chạy:
  .venv/bin/python tools/build_recording_script.py \
      --manifest data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl \
      --max-types 600 --out docs/recording_script_v1.md \
      --syllables-out docs/rare_syllables_v1.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CARRIERS = [
    "Hôm nay tôi muốn đọc từ {w} thật rõ ràng.",
    "Bạn thử nghe lại từ {w} một lần nữa nhé.",
    "Trong câu này có từ {w}, đọc chậm thôi.",
    "Từ {w} nằm ở giữa câu, không đọc quá nhanh.",
    "Tôi đã viết từ {w} lên bảng để bạn xem.",
    "Nếu đọc sai từ {w} thì đọc lại từ đầu.",
    "Chúng ta gặp từ {w} trong bài đọc hôm qua.",
    "Từ {w} nghe hơi lạ, nhưng rất dễ đọc.",
    "Mọi người cùng nhìn vào từ {w} trên màn hình.",
    "Tôi sẽ đọc từ {w} ba lần cho bạn nghe.",
    "Cái từ {w} này xuất hiện khá ít trong sách.",
    "Khi gặp từ {w}, bạn đừng ngắt giữa chừng.",
    "Bài học hôm nay có từ {w} ở ngay đầu câu.",
    "Tôi đánh dấu từ {w} bằng bút đỏ.",
    "Bạn có thể đọc từ {w} to hơn một chút.",
    "Cuối buổi, tôi sẽ kiểm tra lại từ {w}.",
]
MULTI = "Ba từ cần đọc chậm trong câu này là {a}, {b} và {c}."
PAIRS = [
    ("xanh", "xan"), ("ách", "át"), ("ếch", "ết"), ("lạnh", "lạn"), ("quanh", "quan"),
    ("kênh", "kên"), ("minh", "min"), ("móc", "mót"), ("cốc", "cốt"), ("múc", "mút"),
    ("kẹo", "kẻo"), ("boong", "bong"), ("khuých", "khuých"), ("oách", "oát"),
]
GUIDE = """## Cách thu

- Cùng micro, cùng khoảng cách như bộ dữ liệu cũ (khoảng 20 cm, chếch 30°).
- **TẮT RNNoise / noise suppression / echo cancellation** trong phần mềm thu.
- Ghi **WAV gốc 48 kHz, 24-bit** (không qua Zoom/Meet/điện thoại).
- Đọc kiểu **đọc đều** (không diễn cảm), nghỉ ngắn giữa các từ hiếm.
- **Mỗi câu đọc 1 lần**; nếu vấp thì đọc lại cả câu, đừng cắt giữa câu.
- Giữ im lặng 0,5 s ở đầu/cuối mỗi file; đặt tên `v1_0001.wav` … theo đúng số câu.

## Xử lý sau khi thu

- Giữ **WAV gốc**, đưa vào pipeline `audio_v2` với cùng bước chuẩn hoá (loudness, high-pass, gate).
- 50 câu cầu nối dùng để đo chênh kênh thu (micro/phòng/mức to) giữa corpus cũ và bản mới.
- **RNNoise**: chưa chốt — sau khi có câu cầu nối, so phổ bản thô vs bản qua RNNoise offline
  với corpus cũ, chọn bản khớp hơn.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-types", type=int, default=600)
    parser.add_argument("--bridge", type=int, default=50)
    parser.add_argument("--out", required=True)
    parser.add_argument("--syllables-out", default="docs/rare_syllables_v1.txt")
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    words_in_corpus: Counter[str] = Counter()
    bridge: list[str] = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        syllables = row.get("syllables", [])
        for syllable in syllables:
            counts[str(syllable.get("normalized") or syllable.get("raw") or "").lower()] += 1
        current: list[str] = []
        for syllable in syllables:
            current.append(str(syllable.get("normalized") or syllable.get("raw") or "").lower())
            if syllable.get("boundaries", {}).get("word_end"):
                word = " ".join(current)
                if word:
                    words_in_corpus[word] += 1
                current = []
        text = str(row.get("text") or "").strip()
        if 6 <= len(text.split()) <= 30:
            bridge.append(text)

    rare = [w for w, c in counts.most_common() if w.isalpha() and c < 10]
    rare.sort(key=lambda w: counts[w])
    rare = rare[: args.max_types]

    words_by_syllable: dict[str, list[str]] = defaultdict(list)
    for word, freq in words_in_corpus.most_common():
        parts = word.split()
        if len(parts) < 2:
            continue
        for part in set(parts):
            if len(words_by_syllable[part]) < 2:
                words_by_syllable[part].append(word)

    # 2 câu/khuôn khác nhau cho mỗi âm tiết hiếm
    sentences: list[str] = []
    for index, word in enumerate(rare):
        first = CARRIERS[(2 * index) % len(CARRIERS)].format(w=word)
        second = CARRIERS[(2 * index + 7) % len(CARRIERS)].format(w=word)
        sentences.extend([first, second])
    for index in range(0, len(rare), 3):
        triple = rare[index:index + 3]
        if len(triple) == 3:
            sentences.append(MULTI.format(a=triple[0], b=triple[1], c=triple[2]))

    # 50 câu cầu nối: lấy rải đều trong corpus
    stride = max(1, len(bridge) // max(1, args.bridge))
    bridges = bridge[::stride][: args.bridge]

    pair_sentences = [f"Tôi đọc cặp từ này: {a}, rồi {b}. Nghe lại lần nữa: {a}, {b}."
                      for a, b in PAIRS * 4]

    after = Counter(counts)
    for sentence in sentences:
        for word in sentence.lower().replace(",", " ").replace(".", " ").split():
            if word in counts:
                after[word] += 1

    syllables_count = [len(s.split()) for s in sentences + bridges + pair_sentences]
    seconds = sum(syllables_count) / 4.5
    covered = [w for w in rare if after[w] >= 3]
    still = [w for w in rare if after[w] < 3]

    lines = ["# Kịch bản thu bổ sung v1 (phủ âm tiết hiếm)", "",
             (f"- Sinh từ `{args.manifest}`: **{len(counts)} type**, **{len(rare)} âm tiết hiếm** "
              f"(< 10 lần) được chọn."),
             (f"- **{len(sentences)} câu phủ** (2 câu/âm tiết, khuôn khác nhau) + "
              f"**{len(bridges)} câu cầu nối** + **{len(pair_sentences)} câu cặp tối thiểu**."),
             (f"- Ước lượng: **{sum(syllables_count)} âm tiết** ≈ **{seconds/60:.0f} phút** đọc "
              f"(4,5 âm tiết/giây), đọc **1 lần**."),
             f"- Sau kịch bản: **{len(covered)}/{len(rare)}** âm tiết hiếm đạt ≥ 3 lần"
             + (f"; còn **{len(still)}** chưa đạt: {', '.join(still[:20])}." if still else "."),
             "", GUIDE, "## Câu phủ âm tiết hiếm", ""]
    idx = 0
    for sentence in sentences:
        idx += 1
        lines.append(f"{idx:04d}. {sentence}")
    lines += ["", "## Câu cầu nối (đo chênh kênh thu — đọc tự nhiên như bình thường)", ""]
    for sentence in bridges:
        idx += 1
        lines.append(f"{idx:04d}. {sentence}")
    lines += ["", "## Cặp tối thiểu (phân biệt phụ âm cuối / vần)", ""]
    for sentence in pair_sentences:
        idx += 1
        lines.append(f"{idx:04d}. {sentence}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    syllable_lines = ["# âm tiết\tsố lần trong train\ttừ ghép thông dụng (nếu có)"]
    for word in rare:
        compounds = ", ".join(words_by_syllable.get(word, [])) or "(không có)"
        syllable_lines.append(f"{word}\t{counts[word]}\t{compounds}")
    Path(args.syllables_out).write_text("\n".join(syllable_lines) + "\n", encoding="utf-8")

    print(f"{len(sentences)} câu phủ + {len(bridges)} cầu nối + {len(pair_sentences)} cặp · "
          f"{sum(syllables_count)} âm tiết · ~{seconds/60:.0f} phút · phủ {len(covered)}/{len(rare)} · "
          f"{out} · {args.syllables_out}")


if __name__ == "__main__":
    main()

"""tools/build_recording_script.py — sinh kịch bản thu bổ sung cho âm tiết hiếm (A0 v12, Việc 8c).

Vì sao: 62 âm tiết trong 40 câu test có < 10 lần trong train; "ách"/"treo" chỉ 1 lần ⇒ lỗi nghe
không sửa được bằng loss. Tool sinh kịch bản đọc phủ các âm tiết hiếm (mỗi âm tiết ≥ 3 lần),
kèm 50 câu cặp tối thiểu, ước lượng thời lượng (4,5 âm tiết/giây) và bảng phủ trước/sau.

Chạy:
  .venv/bin/python tools/build_recording_script.py \
      --manifest data/codelung/v32/train_clean_v2_dur_seg_slot4.jsonl \
      --texts examples/drift_texts.json --max-types 300 --out docs/recording_script_v1.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
- Mỗi câu đọc **2 lần**; nếu vấp thì đọc lại cả câu, đừng cắt giữa câu.
- Giữ im lặng 0,5 s ở đầu/cuối mỗi file; đặt tên `v1_0001.wav` … theo đúng số câu.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--texts", default="examples/drift_texts.json")
    parser.add_argument("--max-types", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for syllable in json.loads(line).get("syllables", []):
            counts[str(syllable.get("normalized") or syllable.get("raw") or "").lower()] += 1

    rare = [w for w, c in counts.most_common() if w.isalpha() and c < 10]
    rare.sort(key=lambda w: counts[w])
    rare = rare[: args.max_types]

    sentences: list[str] = []
    index = 0
    while index < len(rare):
        chunk = rare[index:index + 3]
        index += 3
        if len(chunk) == 3:
            sentences.append(MULTI.format(a=chunk[0], b=chunk[1], c=chunk[2]))
        else:
            sentences.append(CARRIERS[len(sentences) % len(CARRIERS)].format(w=chunk[0]))
    for word in rare:
        sentences.append(CARRIERS[len(sentences) % len(CARRIERS)].format(w=word))

    pair_sentences = [f"Tôi đọc cặp từ này: {a}, rồi {b}. Nghe lại lần nữa: {a}, {b}."
                      for a, b in PAIRS * 4]

    after = Counter(counts)
    for sentence in sentences:
        for word in sentence.lower().replace(",", " ").replace(".", " ").split():
            if word in counts:
                after[word] += 1

    syllables = [len(s.split()) for s in sentences + pair_sentences]
    seconds = sum(syllables) / 4.5
    covered = [w for w in rare if after[w] >= 3]
    still = [w for w in rare if after[w] < 3]

    lines = ["# Kịch bản thu bổ sung v1 (phủ âm tiết hiếm)", "",
             (f"- Sinh từ `{args.manifest}`: **{len(counts)} type**, "
              f"**{len(rare)} âm tiết hiếm** (< 10 lần) được chọn."),
             f"- **{len(sentences)} câu** + **{len(pair_sentences)} câu cặp tối thiểu**.",
             (f"- Ước lượng: **{sum(syllables)} âm tiết** ≈ **{seconds/60:.0f} phút** đọc "
              f"(4,5 âm tiết/giây; đọc 2 lần ⇒ ~{2*seconds/60:.0f} phút)."),
             f"- Sau kịch bản: **{len(covered)}/{len(rare)}** âm tiết hiếm đạt ≥ 3 lần"
             + (f"; còn **{len(still)}** chưa đạt: {', '.join(still[:20])}." if still else "."),
             "", GUIDE, "## Câu đọc", ""]
    for i, sentence in enumerate(sentences, 1):
        lines.append(f"{i:04d}. {sentence}")
    lines += ["", "## Cặp tối thiểu (phân biệt phụ âm cuối / vần)", ""]
    for i, sentence in enumerate(pair_sentences, 1):
        lines.append(f"{i:04d}. {sentence}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(sentences)} câu + {len(pair_sentences)} cặp · {sum(syllables)} âm tiết · "
          f"~{seconds/60:.0f} phút (2 lần: {2*seconds/60:.0f} phút) · "
          f"phủ {len(covered)}/{len(rare)} · ghi {out}")


if __name__ == "__main__":
    main()

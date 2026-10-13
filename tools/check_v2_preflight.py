"""ViFastTts v2 preflight: 5 kiểm tra trên M6 trước khi code tính năng.

Chạy:  python tools/check_v2_preflight.py --manifest-glob 'data/codelung/*_dur.jsonl' \
                                           --holdout data/codelung/holdout_clean_v2_dur.jsonl
Trước khi chạy, xác minh trên M6: tên manifest, soundfile đã cài,
chữ ký dct4_frames/idct4_frames (torch hay numpy).
Kết quả khác kỳ vọng -> DỪNG, báo số liệu.
"""
import argparse
import glob
import json
import math
from collections import Counter

SR, HOP = 24000, 256  # theo configs/small_codelung.yaml


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def mel_frames(row):
    """Số frame mel đúng như dataset dùng. Row segment: độ dài frame_range."""
    fr = row.get("frame_range")
    if fr:
        return int(fr[1]) - int(fr[0]), "exact"
    import soundfile as sf
    info = sf.info(row["audio_path"])
    n = info.frames if info.samplerate == SR else round(info.frames * SR / info.samplerate)
    # STFT center=True: T = 1 + len // hop
    return 1 + n // HOP, ("exact" if info.samplerate == SR else "approx_resample")


def classify(row):
    """Lặp lại đúng thứ tự nhánh của dataset.py:143-162."""
    N = len(row["syllables"])
    d = row.get("durations") or []
    g = row.get("gaps")
    lead = int(row.get("lead_frames") or 0)
    if len(d) != N:
        return "thieu_durations"
    T, mode = mel_frames(row)
    # dataset.py yêu cầu 0 < len(gaps): N=1, gaps=[] đi nhánh elif sum(d)==T, không gộp gap.
    if g is not None and len(g) > 0 and len(g) in (N - 1, N):
        if len(g) == N - 1:
            g, lead = list(g) + [0], 0
        if lead + sum(d) + sum(g) == T:
            return "forced"
        return f"tong_lech({mode})"
    if sum(d) == T:
        return "forced_khong_gap"
    # gaps có nhưng sai độ dài (vd N-2): dataset có thể lỗi shape thay vì fallback,
    # nên row này phải sửa manifest, không phải nhãn bẩn im lặng.
    return "gaps_sai_do_dai" if g else "khong_gap_tong_lech"


def check1(pattern):
    print("== 1. forced / lý do ==")
    for p in sorted(glob.glob(pattern)):
        c = Counter(classify(r) for r in load_rows(p))
        total = sum(c.values())
        bad = total - c["forced"] - c["forced_khong_gap"]
        print(f"{p}: {total} row, không forced {bad} ({100*bad/max(total,1):.2f}%) {dict(c)}")
    print("Kỳ vọng: không forced <= 1%. Lớn hơn -> dừng, kiểm mfa_to_durations/cut_segments.")


def check2():
    print("== 2. F1: spans vs durations trong batch train ==")
    print("Thủ công: dựng TtsDataset như train.py, lấy 1 batch, in durations/spans/lead,")
    print("xác nhận spans = durations + gaps (spans[0] += lead nếu lead>0) và")
    print("frame lặng thuộc span của âm tiết đứng trước (frame_positions chạy 0..1 trên cả lặng).")


def check3():
    import torch

    from vifasttts.models.vifasttts import LengthRegulator
    print("== 3. LengthRegulator: index + mask ==")
    lr = LengthRegulator()
    cases = {(2, 0, 3): [0, 0, 2, 2, 2], (0, 2, 3): [1, 1, 2, 2, 2], (2, 3, 0): [0, 0, 1, 1, 1]}
    # Giá trị slot = id+1 để phân biệt slot 0 với frame bị mask (=0).
    x = torch.arange(1, 4, dtype=torch.float32).view(1, 3, 1)
    for d, want in cases.items():
        # Batch 2: hàng 2 dài hơn để hàng 1 có padding t >= T thật sự.
        dur = torch.tensor([list(d), [2, 2, 3]])
        out, lengths, mask = lr(x.expand(2, 3, 1).contiguous(), dur)
        got = out[0, :, 0].tolist()
        T = int(lengths[0])
        idx = [int(v) - 1 for v in got[:T]]
        pad_ok = all(v == 0 for v in got[T:]) and not bool(mask[0, T:].any())
        ok = idx == want and pad_ok and bool(mask[0, :T].all())
        print(d, "index", idx, "pad", got[T:], "OK" if ok else f"SAI (kỳ vọng {want})")


def check4():
    import torch

    from vifasttts.data.features import dct4_frames, idct4_frames
    print("== 4. DCT: s0 == mean, công thức s_k khớp idct4_frames ==")
    torch.manual_seed(0)
    for n in (1, 2, 3, 4, 9, 17, 40):
        x = torch.randn(n, dtype=torch.float64)
        c = torch.as_tensor(dct4_frames(x), dtype=torch.float64)
        K = c.numel()
        s = c / math.sqrt(n)
        e0 = abs(float(s[0] - x.mean()))
        # Dựng bằng công thức B1 rồi so với idct4_frames (KHÔNG so với x: 4 hệ số mất thông tin khi n>4).
        pos = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        f = s[0] + sum(math.sqrt(2) * s[k] * torch.cos(math.pi * k * pos) for k in range(1, K))
        ref = torch.as_tensor(idct4_frames(c, n), dtype=torch.float64)
        e1 = float((f - ref).abs().max())
        ok = K == min(4, n) and e0 < 1e-6 and e1 < 1e-6
        print(f"n={n:3d} K={K} |s0-mean|={e0:.1e} |B1-idct|={e1:.1e}", "OK" if ok else "SAI")


def check5(path):
    print("== 5. holdout lead > 0 ==")
    rows = load_rows(path)
    leads = [int(r.get("lead_frames") or 0) for r in rows]
    pos = sorted(l for l in leads if l > 0)
    print(f"{path}: {len(rows)} row, lead>0: {len(pos)}",
          f"(p50={pos[len(pos)//2]}, max={pos[-1]} frame)" if pos else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-glob", required=True)
    ap.add_argument("--holdout", required=True)
    a = ap.parse_args()
    check1(a.manifest_glob); check2(); check3(); check4(); check5(a.holdout)

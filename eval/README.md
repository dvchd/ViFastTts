# `eval/` — tập eval CHỐT (tái lập số liệu)

Mọi con số trong `docs/ACOUSTIC_V2.md` §31–§39 chấm trên **các hàng đầu tiên** của manifest val/holdout.
Thư mục này **ghim** đúng các hàng đó + hash manifest, để máy/người khác chấm lại ra cùng số.

| Tập | Manifest | sha1 (14/10/2026) | Id chốt |
|---|---|---|---|
| val 300 (tất cả thước offline) | `data/codelung/val_clean_v2_dur_seg_slot4.jsonl` | `19b6e11756284ea86ee88933cc315b34011378b7` | `eval/val300_ids.txt` |
| holdout 20 (CER free) | `data/codelung/holdout_clean_v2_dur_slot4.jsonl` | `3ca7c79b0ba4ee9a66fb958ea28b3130da928fd0` | `eval/holdout20_ids.txt` |
| train (đối chiếu) | `data/codelung/train_clean_v2_dur_seg_slot4.jsonl` | `edbc058ecad37a9c7c50844f0bb8dcde8c5ee1a7` | – |
| **train fixgap (A2 §43)** | `data/codelung/train_clean_v2_dur_seg_slot4_fixgap.jsonl` | `fdb622940e0de9560a83050f0c8467cd34dea445` | – |
| **val300 fixgap (A2)** | `data/codelung/val_clean_v2_dur_seg_slot4_fixgap.jsonl` | `5ec5f139ce56…` (đầy đủ trong `artifacts/dataset_v2/verify_gaps_val_fixgap.json`) | – |
| ~~holdout20 fixgap (A2)~~ **KHÔNG DÙNG** | `data/codelung/holdout_clean_v2_dur_seg_slot4_fixgap.jsonl` | `aa418147f314…` | – |
| stats F0 | `data/codelung/f0_stats_v2.json` | `505604ebf0356560fb3348691668a687c2096c51` | – |

**Quy ước**: tool luôn lấy `--clips N` = **N hàng đầu** của manifest ⇒ `--clips 300` + manifest trên
= đúng tập trong `val300_ids.txt`. Nếu hash manifest khác ⇒ **đổi tập, số cũ không so được** —
checkpoint cũng ghi `val_manifest_hash` + `validate_batches` (từ 14/10/2026) để phát hiện việc này.

Chạy lại đủ bộ: xem `docs/EVAL_TOOLS.md` §4. Kiểm nhanh:
```bash
shasum data/codelung/val_clean_v2_dur_seg_slot4.jsonl   # phải khớp bảng trên
head -3 eval/val300_ids.txt
```

**A2 (§43)**: manifest `*_fixgap.jsonl` giữ **đủ gap** (kể cả gap đuôi) ⇒ Σgap/row 220,8→409,8,
phủ khối lượng pause 53,4%→**99,2%**, bất biến `T == Σdur+Σgap` 100%, `durations`/`seg_durations`
**không đổi** (seg hash không đổi). Val/holdout giữ nguyên ⇒ tập eval vẫn ghim như bảng trên.

**Holdout**: bench mặc định dùng holdout **CŨ** (`holdout_clean_v2_dur_slot4.jsonl`, clip-level 842 hàng) để
nối lịch sử CER; bản fixgap là segment-level (khác cấu trúc) ⇒ **không dùng** (§44, quyết định 2.5).

**Còn thiếu (đã biết)**: model `tone_ruler` (`artifacts/tone_path/tone_ruler.pt`) chưa được version-control
(train lại ~4 phút: `tools/tone_ruler.py --train`) và chưa ghi hash model vào JSON kết quả — xem
kiểm toán 14/10, việc #1/#10.

## Mốc (baseline) — `eval/baselines/` (§44)

| Tệp | sha256 | Dùng cho |
|---|---|---|
| `base40k_old.json` | `6cf65481655e657fa96bd33091b7a2f1c2e91d7d9c63b0d9c32c69ac7f33fdee` | CỔNG CỨNG (mel/dB/rough/tone_C/CER/F0 NGÃ @ val cũ) |
| `base40k_fixgap.json` | `7aaa9676c30f5c777a362b589bf44fc1721193bc0f4932c2ba4d4fe739f4908a` | mel_gap + syl_z + CỔNG MỤC TIÊU pause (val fixgap) |

Kiểm H1 (base 40k): `syl_z` val cũ **0,6612** → fixgap **0,6608** (chênh **0,1%**) · `gap_z` 1,9493 → 2,3862
⇒ mọi phần tăng của `mel_l1_z` tổng nằm ở khung gap (khung âm tiết không đổi) — §44.6.

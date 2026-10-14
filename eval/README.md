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
| **train fixtail (A2 §44.8)** | `data/codelung/train_clean_v2_dur_seg_slot4_fixtail.jsonl` | `965b4798b23266c6006be658dd1e93e13d8072ac` | – |
| **val300 fixtail (A2)** | `data/codelung/val_clean_v2_dur_seg_slot4_fixtail.jsonl` | `cdbbd659116ae9f948c56152be5b08ea7fa08115` | `eval/val300_ids.txt` |
| **val_calib (V11.A6 — hiệu chuẩn)** | `data/codelung/val_clean_v2_dur_seg_slot4_fixtail_calib.jsonl` | `147f4b7c117f6d54ee573fcc2fa537a0188fa89f` | hàng 301–1773 chia theo clip cha, ~nửa |
| **val_lock (V11.A6 — KHÓA)** | `data/codelung/val_clean_v2_dur_seg_slot4_fixtail_lock.jsonl` | `f6db096aa9edea1d9eca33171e00c592d065e31d` | chỉ dùng đánh giá CUỐI mỗi dòng; mỗi lần dùng ghi §44 |
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

## Mốc (baseline) — `eval/baselines/` (§44; V10.A3/A4)

Ghim sha256 theo **byte trong git**: `git show HEAD:<tệp> | shasum -a 256`. Tệp mốc chuẩn hóa (LF, `sort_keys`,
indent 2, dòng trống cuối) + `.gitattributes` `eval/baselines/*.json -text` ⇒ sha ổn định trên mọi máy checkout
(máy bật `core.autocrlf` cũng không đổi byte). Test `test_pin_readme_khop_byte_trong_git` khóa ở mọi lần test.

| Tệp | sha256 | val_manifest_hash | Trạng thái |
|---|---|---|---|
| `base40k_old.json` | `92e90dad406f661557d2a648c3a171bf8c9e6320bb659b261efcd50a88d475d8` | `19b6e11756284ea86ee88933cc315b34011378b7` (val CŨ) | **DÙNG** — CỔNG CỨNG (mel/dB/rough/tone_C/CER/F0 NGÃ @ val cũ) |
| `base40k_fixtail.json` | `a8b74d2ea48f0ebc58027f56282835df74bc0199b16e9f1ac0205ded46c14004` | `cdbbd659116ae9f948c56152be5b08ea7fa08115` (val fixtail; old `19b6e117…`) | **DÙNG** — mel_gap + syl_z + CỔNG MỤC TIÊU pause (val fixtail) |
| `base40k_fixgap.json` | `cf7708644f6396028452e746258a955bf848f3ea74521ffad3d20273bf024036` | `5ec5f139ce56a763d90af7de7a59956f712ee88c` (val fixgap) | **KHÔNG DÙNG** (thay bởi fixtail; bản trên đĩa: `artifacts/bench/legacy/`) |

Kiểm H1 (base 40k): `syl_z` val cũ **0,6612** → **fixtail 0,6621** (chênh **+0,14%**) · `gap_z` 1,9493 → **2,1529** (giá trị fixtail; bản fixgap cũ 2,3862 — KHÔNG dùng)
⇒ mọi phần tăng của `mel_l1_z` tổng nằm ở khung gap (khung âm tiết không đổi) — §44.6.

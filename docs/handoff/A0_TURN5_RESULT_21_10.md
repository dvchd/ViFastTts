# A0 lượt 5 — đáp lời: sửa cách đọc cổng, thước 2.2/2.3, MFA 3.1–3.4 — 21/10/2026

## 0. Nhận đính chính

1. **Đúng**: cổng phải đọc theo **EMA** (export dùng EMA). Tính lại: **P2 EMA GV 0,724 · val 1,6167
   = +4,1 %** so baseline 1,5529 ⇒ **P2 QUA cả hai cổng** và hơn P1 (EMA GV 0,616 · +2,8 %).
   Trước đây mình đọc cột LIVE nên kết luận sai. Từ nay bảng quyết định chỉ dùng EMA.
2. **Đúng**: GV bị "gian lận" được (P4 @400: GV 0,949, val +10 %). Cổng = GV EMA + val EMA + tai
   + 2 thước phụ dưới đây.
3. **Đúng**: NG→ŋm **không phải lỗi** — xem 3.1 (bằng số).
4. Adaptive (P3/P4): **bỏ**, không sửa tiếp.

## 1. Việc 2c — chạy

- **P1 20k** (`runs/adv_p1_20k`): đang chạy, tới bước 5k. EMA @4k: val 1,6087 (+3,6 %), GV 0,647.
- **P2 20k** (`runs/adv_p2_20k`): **đã bật song song** (adv 0,3 · fm 0, warmup 300, hard-stop val EMA +10 %).
  M6 chịu được 2 run + v3.4; nếu chậm quá sẽ tắt bớt (đã ghi).
- Checkpoint mỗi 2k; sẽ chấm 4k/10k/20k theo §2.1.

## 2. Thước 2.2 + 2.3 (đo trên 40 segment val cố định, TF forward, mel **trước** α, EMA)

| checkpoint | val mel L1 | **val voiced** | GV hữu thanh | GV 2–6 kHz | **flux biên** (→1) | **smooth** (loại nếu >1,2) |
|---|---|---|---|---|---|---|
| v3.2 (baseline) | 1,5128 | 0,5686 | 0,574 | 0,555 | **0,505** | **0,550** |
| P1 @2k | 1,5412 | 0,5882 | 0,616 | 0,583 | **0,534** | **0,593** |
| P1 @4k | 1,5565 | 0,6063 | 0,649 | 0,616 | **0,566** | **0,628** |

Đọc:
- `flux biên` (thước cũ, thật = 1,0): **0,505 → 0,534 → 0,566** — model **nét dần đúng hướng thật**,
  không phải chỉ tăng phương sai. Đây là bằng số cho việc tai người dùng nghe "đỡ rè hơn" ở P1 2k.
- `smooth` (|Δmel| giữa 2 frame trong vùng hữu thanh, synth/real): 0,55 → 0,59 → 0,63. Vẫn **thấp hơn
  thật** (model còn mượt hơn thật) và **rất xa ngưỡng loại 1,2** ⇒ không có hiện tượng rung/nhiễu giả.
- `val voiced` tăng nhẹ (0,569 → 0,588 → 0,606) cùng chiều với L1 toàn row — chấp nhận được vì đây là
  L1 trên **mel z**, không phải thước chất lượng (A0 §0.1).

## 3. MFA (đo, chưa re-align)

### 3.1 NG → ŋm: **xác nhận là chuẩn Bắc**, không phải lỗi
Tách theo nucleus (toàn corpus 27.397 token NG):

| nucleus | n | phone cuối |
|---|---|---|
| **Ô** | 6.620 | **ŋm 99,2 %** |
| **U** | 6.149 | **ŋm 100 %** |
| **O** | 2.991 | **ŋm 100 %** |
| Ă | 3.330 | ŋ 100 % |
| A | 2.966 | ŋ 100 % |
| Ư | 2.546 | ŋ 100 % |
| ƯƠ | 1.938 | ŋ 100 % |
| IÊ | 361 | ŋ 100 % |

⇒ ŋ͡m chỉ xuất hiện sau O/Ô/U đúng quy luật Bắc (ong/ông/ung). **Kết luận: không phải biến thể Nam.**

### 3.2 N → ŋ và T → k̚: đây **mới là** biến thể Nam, tập trung theo nucleus

| coda N (33.156) | tỉ lệ ŋ | | coda T (18.062) | tỉ lệ k̚ |
|---|---|---|---|---|
| Ô | **78,4 %** | | Ă | **60,4 %** |
| Ơ | **55,7 %** | | A | **37,8 %** |
| IÊ | 34,3 % | | IÊ | 22,7 % |
| A | 33,3 % | | Â | 12,2 % |
| Â | 21,1 % | | E | 6,8 % |
| UÔ | 22,5 % | | I | 0 % |
| Ê | 0 % | | Ơ | 0 % |
| E | 13,9 % | | Ô | 0 % |

⇒ Đúng như A0 nói: n/ŋ đều là âm mũi, MFA chỉ quyết định **mốc thời gian**.

### 3.3 Ảnh hưởng lên mốc thời gian — **chưa re-align được ở máy này**
`mfa` **không có** trong môi trường (`.venv` không cài Montreal Forced Aligner) ⇒ không chạy lại
align 1 giờ dữ liệu trong lượt này. **Đo thay thế** (proxy) trên 1.500 TextGrid: so **cùng một từ**
ở hai biến thể (Bắc/Nam), trung vị thời lượng phone:

| từ | biến thể | n | nucleus ms | coda ms | Δ nucleus | Δ coda |
|---|---|---|---|---|---|---|
| bạn | `ɓ aː˨˩˨ ŋ` vs `ɓ aː˨˩ˀ n` | 56 / 235 | 60 vs 40 | 50 vs 80 | **1,9 frame** | **2,8 frame** |
| luôn | `l uə˨˨ n` vs `l uː˨˨ ŋ` | 154 / 53 | 60 vs 40 | 100 vs 70 | **1,9 frame** | **2,8 frame** |
| không | `x ɔ˨˨ ŋm` vs `x o˨˨ ŋm` | 303 / 304 | 60 vs 60 | 40 vs 30 | 0 | 0,9 frame |
| phần | `f ə˨˨ n` vs `f ə˨˩ ŋ` | 236 / 60 | 80 vs 80 | 40 vs 40 | 0 | 0 |
| dùng | `z/j u˨˨ ŋm` | 221 / 23 | 80 vs 80 | 60 vs 50 | 0 | 0,9 frame |
| trong | `tɕ/ʈ ɔ˨˨ ŋm` | 130 / 36 | 70 vs 70 | 50 vs 50 | 0 | 0 |
| dụng | `z/j u˨˩ˀ ŋm` | 156 / 16 | 50 vs 50 | 60 vs 50 | 0 | 0,9 frame |

**Trung vị 7 từ: Δnucleus 0,0 frame · Δcoda 0,9 frame.** Theo cổng A0 (median < 1 frame và p90 < 2):
**đạt ở trung vị**, nhưng vài từ đơn lẻ tới 2,8 frame ⇒ **chưa đủ cơ sở để khẳng định "bỏ re-align"**.
Đề xuất: cài MFA ở môi trường riêng rồi làm đúng 3.3 (1 giờ) — nếu không có MFA thì giữ nguyên
align hiện tại cho v3.5 (rủi ro thấp: median ~1 frame ≈ 10,7 ms).

### 3.4 Token T mất phone cuối: slot C **không bao giờ = 0**
400 segment ghép được 100 % với TextGrid (khớp theo `frame_range` + difflib), `seg_durations[i][3]`:

| coda | n | slot C median | p10 | p90 | C=0 % | có phone (median) | mất phone (median) |
|---|---|---|---|---|---|---|---|
| T | 281 | 3 | 2 | 4 | **0 %** | 3 (n=151) | **2** (n=130) |
| C | 220 | 2 | 1 | 4 | 0 % | — | 2 |
| CH | 39 | 3 | 2 | 4 | 0 % | 3 (n=30) | 2 (n=9) |
| P | 70 | 3 | 2 | 5 | 0 % | — | 3 |
| N | 438 | 4 | 3 | 8 | 0 % | 5 (n=321) | 4 (n=117) |
| NG | 375 | 4 | 2 | 7,6 | 0 % | 4 (n=151) | 4 (n=224) |
| NH | 199 | 4 | 3 | 9 | 0 % | 4 (n=186) | 4 (n=13) |
| (không coda) | 1.827 | 0 | 0 | 0 | 100 % | — | — |

⇒ Rule hybrid **luôn cấp tối thiểu ≥2 frame cho slot C** khi MFA mất phone; thiếu phone ⇒ T mất
~1 frame (≈10,7 ms). **Không phải lỗi rule**; nếu muốn chắc, có thể nâng sàn slot C của T/CH/P
lên 3 frame trong v3.5 (đo lại sau).

## 4. Chưa xong (chờ 20k)

- **2.4** trang nghe mù 12 câu (6 val + 6 câu mới có xanh/lạnh/quanh/ách/ếch), 4 bản/câu, −23 LUFS,
  xáo thứ tự, ẩn đáp án + 2 câu dài > 20 s: **dựng khi có checkpoint "best" của P1/P2**.
  Trang tiến trình tạm: https://cio-rows-rocks-seq.trycloudflare.com/adv_progress/ (v3.2 + P1 2k/4k).
- **2.5** export ONNX checkpoint thắng + md5 ×3 + RTF CPU 2 luồng: **sau khi chọn**.

## 5. File/tool mới (đã commit cùng báo cáo)

- `tools/adv_checkpoints.py` — bảng 2.2/2.3 (val, voiced, GV, GV26, flux biên, smooth).
- `tools/adv_progress_demo.py` — trang nghe tiến trình từng checkpoint (EMA gộp vào base).
- `tools/audit_slot_c.py` — slot C theo coda, tách có/mất phone (3.4).
- `tools/audit_mfa_coda.py` — thêm `by_nucleus` (3.1/3.2).

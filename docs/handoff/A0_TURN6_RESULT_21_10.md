# A0 lượt 6 — Việc 4a (thước năng lượng) + bỏ α — 21/10/2026

## 0. Bỏ α (theo chỉ đạo người dùng)

- **Mặc định α = 0** ở mọi nơi: `src/vifasttts/cli.py` (1,5 → 0), `infer.py` (đã 0), `tools/steady_demo.py`,
  `tools/roughness_dashboard.py`, `tools/coda_realization.py`, `tools/creak_rarity_probe.py`,
  `tools/adv_energy_audit.py`, `tools/adv_progress_demo.py`.
- `chunking.py` và `export.py` **không có** tham số α (α chỉ áp trong `infer.py`/CLI/tool) — ghi nhận,
  không cần xoá. Sau khi chốt checkpoint sẽ xoá hẳn tham số khỏi `infer.py` + CLI + docs.
- Trang nghe tiến trình đã sinh lại **chỉ α=0**: `/adv_progress/` (P1) và `/adv_progress_p2/` (P2).
- 2.525 test pass, ruff sạch.

## 1. Việc 4a — spike năng lượng từng âm tiết (40 segment val, 683 âm tiết, TF, EMA)

Cách đo: mean log-mel **đã bỏ z-norm** (dB = 8,686 × log) trên frame của âm tiết, **trừ trung vị câu**
(độ to tương đối); spike = (synth − thật) > +3 dB; RMS dB của waveform sau Vocos cùng cách.

| checkpoint | spike α=0 | MAE dB α=0 | **RMS spike α=0** | (α=1,5: spike / RMS spike) |
|---|---|---|---|---|
| v3.2 | 0,186 | 2,589 | **0,078** | 0,177 / 0,108 |
| P1 @2k | 0,196 | 2,600 | 0,079 | 0,190 / 0,101 |
| P1 @4k | 0,208 | 2,624 | 0,084 | 0,205 / 0,107 |
| P1 @6k | 0,205 | 2,632 | 0,089 | 0,206 / 0,101 |
| P2 @2k | 0,189 | 2,613 | **0,123** | 0,190 / 0,110 |

### Kết luận H1/H2/H3

- **H1 (α cộng dồn) — ĐÚNG một phần, ở mức WAVEFORM**: trên mel, α gần như không đổi spike
  (0,186 → 0,177). Nhưng trên **RMS sau Vocos**, α=1,5 **tăng spike 0,078 → 0,108** ở v3.2 (+38 %)
  và giữ P1 ở 0,10–0,11 so với 0,079–0,089 khi α=0. Đây là cơ chế hợp lý cho "từ tự nhiên to lên"
  mà tai nghe thấy: trang cũ dùng α=1,5. ⇒ **bỏ α là đúng**.
- **H2 (GAN kéo lệch năng lượng) — YẾU**: spike tăng theo bước nhưng chỉ 0,186 → 0,196 → 0,208 → 0,205
  (1,05–1,12× v3.2), **không đạt ngưỡng 2×** ⇒ **chưa cần Việc 4b**. (P2 @2k spike mel 0,189 nhưng
  RMS spike 0,123 = cao nhất — theo dõi tiếp ở P2 4k/6k.)
- **H3 (nhiễu EMA) — KHÔNG**: 4k → 6k đi ngang (0,208 → 0,205).

### Top 10 spike — **giống hệt nhau ở mọi checkpoint** (di truyền từ v3.2, không phải do GAN)

| # | từ | v3.2 | P1 2k | P1 4k | P1 6k | P2 2k |
|---|---|---|---|---|---|---|
| 1 | qua | +21,2 | +22,3 | +22,7 | +22,7 | +23,1 |
| 2 | Ở | +17,7 | +19,0 | +19,3 | +19,1 | +21,1 |
| 3 | ét | +15,8 | +15,9 | +15,7 | +15,5 | +16,5 |
| 4 | Ở | +14,8 | +17,6 | +17,4 | +17,7 | +19,9 |
| 5 | ai | +14,7 | +14,7 | +14,3 | +14,5 | +15,1 |

⇒ Đây là các âm tiết ngắn/hư từ bị lệch chuẩn hoá (và có thể lệch biên align), **có từ trước GAN**.
Tai nghe "từ to lên" ở P1 4k nhiều khả năng là **α=1,5 trên mel đã nét hơn** + vài sự kiện hiếm,
không phải dịch chuyển năng lượng hệ thống.

## 2. Run

- P1 20k: bước 7,5k · EMA @7k val 1,6101 (+3,7 %), GV 0,635.
- P2 20k: bước 2,7k · EMA @2k val 1,6167 (+4,1 %), GV 0,724.
- Hard-stop spike (α=0 > 2× v3.2 trong 2 mốc liên tiếp): **chưa chạm** (ngưỡng 0,372; hiện 0,20–0,21).

## 3. Trang nghe (α=0)

- Tiến trình P1: https://cio-rows-rocks-seq.trycloudflare.com/adv_progress/
- Tiến trình P2: https://cio-rows-rocks-seq.trycloudflare.com/adv_progress_p2/
- Top spike từng checkpoint: `/energy/v32_ref/`, `/energy/p1_2k/`, `/energy/p1_4k/`, `/energy/p1_6k/`, `/energy/p2_2k/`

## 4. Bổ sung (người dùng nghe): "xoá" to dần ở P1 — P2 2k KHÔNG bị

Đo trên chính câu `tech2` ("… Chúng ta sẽ xoá cái phần này đi …"), WAV chế độ đọc đều α=0 đã có,
năng lượng **mệnh đề** chứa "xoá" so mệnh đề 1 (dB):

| checkpoint | c1 | c2 (chứa "xoá") | **c2 − c1** | đỉnh c2 |
|---|---|---|---|---|
| v3.2 | −29,1 | −28,4 | **+0,7** | −21,8 |
| P1 @2k | −28,2 | −26,9 | +1,2 | −20,6 |
| P1 @4k | −28,8 | −24,5 | **+4,3** | −13,8 |
| P1 @6k | −28,4 | −22,4 | **+6,0** | −12,4 |
| P1 @8k | −29,0 | −24,8 | +4,2 | −18,1 |
| **P2 @2k** | −25,4 | −25,6 | **−0,2** | −19,5 |

⇒ "Từ to lên" là **thật và đo được**, tăng theo bước ở P1 (H2 — GAN kéo lệch năng lượng), và **P2 @2k
không bị** (cân bằng như v3.2). Khớp đúng tai người dùng: P2 2k rõ hơn, ít rè hơn, ít to tiếng hơn.
⇒ Ứng viên hiện tại: **P2 @2k** (chờ P2 4k/6k để xem có trôi không; nếu trôi → chọn checkpoint sớm).
⇒ Việc 4b (energy envelope loss + low-band ×2) **có cơ sở** — ưu tiên khởi động từ P2 @2k.

## 5. Bổ sung (người dùng nghe): P2 @2k là bản tốt nhất hiện tại
Tai người dùng: **P2 2k rõ hơn P1, ít rè hơn, ít to tiếng hơn** — khớp bảng mục 4.

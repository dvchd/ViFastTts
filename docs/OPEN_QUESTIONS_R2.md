# Câu hỏi cần chuyên gia chốt (R2 → trước khi resume chuỗi nền)

> **Người gửi:** agent cầu nối. **Người chuyển:** chủ dự án.
> **Trạng thái hiện tại (quan trọng):**
> - Chuỗi nền đang **dừng an toàn**: shell `69421` ở trạng thái `Ts` (SIGSTOP), **không có tiến trình con**.
> - **Cache train đã xong 100% (11.424/11.424 file)**; cache val chưa chạy; `runs/vocoder_codelung_v2` **chưa tồn tại** → chưa mất giờ MPS nào.
> - Resume: `kill -CONT 69421` · Hủy: `kill 69421`. **Không có gì đang chạy** ngoài trạng thái chờ này.
> - Tham chiếu số đo đầy đủ: `docs/AI_REVIEW_ANSWERS_R1.md`, `docs/AI_REVIEW_ANSWERS_R2.md`.
>
> **Cách trả lời đề nghị:** mỗi câu ghi `A1: chọn (a)/(b), lý do…` — các ô có chữ **"Đề xuất"** là
> phương án agent cầu nối nghiêng về, chỉ cần xác nhận hoặc bác.

---

## PHẦN 1 — Phải chốt TRƯỚC KHI RESUME (để v2 không bị nhiễm eval hoặc chạy lại 20h)

### A. Chia lại val 16 video + lock

**A1. Phần clip CÒN LẠI của các video val (sau khi cap ~150 clip/video) xử lý thế nào?**
Bối cảnh: `pick_val_videos(16, seed=13)` = đúng 4 video val hiện tại + 12 video mới; 16 video này có
6.191 clip / 24,0h; cap 150/video → val = 2.235 clip.
- (a) **Loại hẳn khỏi train** (tách nghiêm ngặt theo video): train = 15 video còn lại = 7.024 clip · **27,7h**,
  nhưng train mới là **tập con của train cũ → cache đã ấm 100%**; sạch hoàn toàn cho eval.
- (b) **Giữ clip dư trong train**: train = 10.980 clip · ~43h, nhưng clip cùng video nằm trong train →
  eval bị nhiễm cùng video (CER có thể lạc quan); cần **cache bù ~1.280 clip** trước khi v2 chạy.
- (c) Giảm số video val (ví dụ 8) để bớt mất giờ train — cần tính lại số cụ thể.
> **Đề xuất: (a)** — vì mục đích của lock là eval sạch; 27,7h vẫn là dữ liệu vocoder đáng kể, và giữ
> được tính "không cần cache bù, không tốn thêm gì trước v2".

**A2. `split_lock.json` khoá ở mức nào?**
- (a) Chỉ **danh sách video val** + hash (như review đề xuất).
- (b) Video val + **danh sách clip val cụ thể** + hash (khoá cả mẫu eval, tái lập đúng 2.235 clip).
- Khi `manifest.jsonl` đổi (reparse/recut/overrides) thì lock cũ xử lý ra sao: giữ video, tự lấy lại clip
  của video đó theo cùng quy tắc cap? (đề xuất) hay báo lỗi bắt tạo lock mới?
> **Đề xuất: (b)** + quy tắc: lock giữ video; clip được tái tạo theo quy tắc cap cố định (sort theo id,
> lấy ≤150) và **assert hash clip list khớp** — lệch thì dừng, bắt `--force`.

**A3. Khi có lock, `reparse_overrides` / `split_manifest` làm gì?**
- (a) **Tôn trọng lock**: chỉ chia lại text/manifest, giữ nguyên video val; không có lock thì log cảnh báo.
- (b) Bắt buộc `--force` mới được chia mới; mặc định **từ chối chạy** nếu không có lock (an toàn nhất).
> **Đề xuất: (a)** cho `reparse_overrides` (vẫn cần cập nhật train/val theo text mới) — nhưng nếu
> người dùng truyền `--val-videos/--seed` khác lock thì **raise lỗi** thay vì âm thầm đổi.

**A4. Thứ tự vận hành sau khi chia lại (ai chạy gì)?**
- Nếu chọn A1(a): re-split → (không cần cache bù) → **resume chuỗi** (`cache val` sẽ tự tính cho val mới;
  phần lớn val clip đã có cache từ train, chỉ ~510 file mới).
- Nếu chọn A1(b): re-split → **chạy cache-features cho train trước** (~1.280 file, vài phút) → resume chuỗi.
> **Đề xuất: giữ nguyên shell SIGSTOP rồi resume** sau khi code xong; chỉ `kill` nếu muốn chạy từng bước bằng tay.

**A5. Có giữ 4 video val cũ trong 16 video mới không?**
`seed=13` cho thấy 16 video mới **chứa đủ 4 video val cũ** (không video val nào quay lại train).
> **Đề xuất: giữ nguyên seed 13 / 16 video** — kết quả đúng như mong muốn, không cần tham số mới.

### B. Log gradient từng thành phần (trước v2)

**B1. Lấy gradient theo tensor nào?**
- (a) `fake_crop` — waveform sau generator (đúng như review viết, rẻ vì không lan ngược qua G).
- (b) `mel_fake` — mel sinh (đắt hơn, không cần cho quy tắc đang bàn).
- Công thức xác nhận: `g_k = torch.autograd.grad(w_k * loss_k, fake_crop, retain_graph=True)[0].norm()`?
> **Đề xuất: (a)**, giữ `retain_graph=True` cho 3 thành phần đầu, cái cuối không cần.

**B2. Cột ghi vào `history_vocoder.csv` + tần suất?**
Hiện: `step,g_total,g_adv,g_mel,g_stft,d_loss` (g_adv/g_mel/g_stft là loss **chưa nhân weight**, g_total là
tổng có weight). Đề xuất mỗi 100 bước thêm 4 cột: `gr_adv,gr_fm,gr_mel,gr_stft` (norm đã nhân weight).
> **Đề xuất:** ghi từ bước 100; vẫn ghi loss như cũ. Header đổi — runner khối không đọc file này nên an toàn.

**B3. Ngưỡng hành động 5% / 60% dựa trên cơ sở nào?**
Số đo random-init của tôi (batch=1, crop 64, CPU): **adv ~0,00% · fm 0,02% · mel 60,2% · stft 39,7%**.
Nếu áp luật "adv+fm <5% → hạ mel_weight" **ngay từ đầu** thì sẽ hạ mel_weight trong khi D chưa ấm —
trái với chính lưu ý "sau 5k–10k bước" của review.
- (a) Chỉ đánh giá từ 5k–10k (đúng như review nói), dùng **tỉ lệ grad-norm**.
- (b) Dùng thêm tiêu chí phụ: chỉ hạ mel_weight nếu tỉ lệ vẫn <5% **và** g_adv không tăng sau 10k.
- Có tài liệu/kinh nghiệm nào cho ngưỡng 5/60 không, để ghi vào docs cho minh bạch?
> **Đề xuất: (a)+(b)**; chưa hạ gì ở 2k–5k đầu.

---

## PHẦN 2 — Chốt trước smoke acoustic (làm trong lúc v2 chạy, không chặn resume)

### C. F0 / Energy (không được phá cache)

**C1. Stats chuẩn hoá F0 đặt ở đâu để không đổi khoá cache?**
Khoá cache chứa nguyên mục `f0:` + `phonation:` + `mel:` (`data/cache.py:44`).
- (a) Mục top-level mới, ví dụ `f0_norm: {stats_path: ..., log: true}` — không nằm trong hash cache.
- (b) File stats riêng (kiểu `mel_stats.json`) + đường dẫn truyền qua CLI khi extract.
> **Đề xuất: (a)** — cùng tinh thần với `mel.normalize_path`, nhưng top-level để không phá cache.
> *(Lưu ý: stats F0 phải tính **chỉ trên frame hữu thanh**; ai tính — tool mới `compute_f0_stats.py`?)*

**C2. DCT 4 hệ số — xác nhận công thức & quy tắc loss?**
- Công thức đề xuất: `c_k = Σ z_t·cos(πk(t+0.5)/n)/n`, k=0..3, z = log-F0 đã z-score.
- Loss L1 trên 4 hệ số, **trọng số theo tỉ lệ frame hữu thanh**, **bỏ âm tiết < 0,3**.
- Xác nhận: c0 (trung bình) có cần thêm trọng số cao hơn như vài hệ TTS không, hay 4 hệ số ngang nhau?
- Chuẩn hoá DCT: giữ thang z-score hay nhân scale cho dễ train (ví dụ c1..c3 ×2)?
> Cần chuyên gia chốt 2 điểm này vì ảnh hưởng trực tiếp loss predictor.

**C3. Interface model/ONNX khi F0 đổi từ scalar → 4 hệ số?**
- `self.f0` predictor đổi output `[B,N]` → `[B,N,4]`; variance projector nhận thêm 3 chiều.
- Decoder dựng contour bằng IDCT + `frame_position` (ma trận cos, export ONNX được — đã kiểm ONNX hỗ trợ
  các op tương ứng? Cần xác nhận vì tôi chưa test export IDCT).
- ONNX output sẽ đổi: `f0` thành `[B,N,4]` (hoặc thêm output `f0_coeffs`); runtime `synthesize-onnx`
  hiện chỉ cần `mel` + `durations` nên **không vỡ**, nhưng tài liệu/hợp đồng output phải cập nhật.
> **Đề xuất:** giữ tên output `f0` nhưng đổi shape `[B,N,4]` + thêm metadata `f0_repr: "dct4"`;
> checkpoint cũ (scalar) không tương thích — chấp nhận, vì sẽ retrain sau MFA.

**C4. Energy: có z-score và/hoặc DCT không?**
Energy hiện là scalar/âm tiết từ log-power (đo được 4,49 — lớn thứ hai sau f0). Review đề xuất z-score.
- (a) Chỉ z-score scalar (rẻ, giữ F4).
- (b) DCT 2–4 hệ số như F0 (đắt hơn, contour năng lượng).
> **Đề xuất: (a)** cho lần smoke này; ghi nhận (b) cho v2.1.

### D. SIL (A3) — cần đặc tả đủ để code ngay sau MFA

**D1. SIL mã hoá thế nào trong 7 slot?** (spec A3 cũ nói "7 slot + tone NONE → vocab 70", nhưng chưa chốt giá trị)
- (a) Thêm giá trị `SIL` vào **onset** (24→25) và các slot còn lại NONE/NGANG.
- (b) Thêm một **trường thứ 8** (token type) — phá hợp đồng 7 ID, không nên.
- (c) SIL là âm tiết có onset `NONE` + nucleus đặc biệt? (phá tính "nucleus luôn tồn tại")
> **Đề xuất: (a)** — vocab 25/2/15/11/6 + boundary/punct; ID mới chỉ ở bảng onset; checkpoint cũ không
> tương thích (chấp nhận vì retrain).

**D2. Vị trí chèn tất định + dấu câu của SIL?**
- Đầu câu, cuối câu, sau `. ! ? … , ; :` (chốt danh sách); SIL nhận `punctuation=NONE`, boundary=SENTENCE/PHRASE?
- SIL cuối câu đặt trước hay sau dấu câu?
> Cần chốt để test round-trip text→IDs ổn định giữa train/lúc synthesize.

**D3. Duration SIL khi teacher forcing?**
- Dataset phải cho phép 0..n; `allocate_durations` (placeholder khi chưa có MFA) **không cấp frame cho SIL**.
- LengthRegulator đã chịu 0 (đã kiểm, torch + ONNX wrapper). Collate cần chấp nhận tổng duration bằng 0 với
  chuỗi toàn SIL? (thực tế khó xảy ra)

**D4. Export ONNX với SIL=0 — cách nào?**
Hiện `durations = round(exp(log_d)-1).clamp_min(1)` (`vifasttts.py:116`) chặn 0.
- (a) Thêm input `sil_mask` hoặc `duration_scale` vào graph; clamp riêng phần không-SIL.
- (b) Đưa `durations` thành input của graph (như wrapper tôi đã test) — runtime tự quyết định.
- (c) Chấp nhận SIL ≥ 1 frame trên ONNX (sai lệch nhỏ, không đẹp).
> **Đề xuất: (a)** kèm `duration_scale` cho `--speed` (gộp luôn việc review số 7). Cần chuyên gia xác nhận
> vì đây là thay đổi hợp đồng export.

**D5. Khoảng lặng giữa câu ≥150ms không có SIL:**
- (a) Loại clip khỏi acoustic (như review đề xuất).
- (b) Cắt thành segment ngay tại đó (giữ dữ liệu, phức tạp hơn).
- Ngưỡng 150ms có giữ không? Có cần thêm luật "ưu tiên cắt sau dấu câu" không?
> **Đề xuất: (b) nếu tỉ lệ clip bị loại khi thử MFA >5%; tạm thời code (a) trước cho đơn giản.**

**D6. Loss nào tắt tại SIL?**
f0/tone/vuv/energy mask SIL (đề xuất có); duration loss giữ (SIL học độ dài); mel loss giữ?
> **Đề xuất: mask f0/tone/vuv/phonation; giữ mel + duration.**

### E. Tone classifier đóng băng

**E1. `loss_weights.tone = 0` ngay bây giờ?**
Hiện `small_codelung.yaml` để 0,1 nhưng đo được loss ~0,0014 (vô dụng). Chuyên gia xác nhận set 0 **ngay**
cho lần acoustic tới (và giữ head trong graph cho tương thích ONNX)?

**E2. CNN phân loại thanh 6 lớp (đóng băng) — ai train, train trên gì?**
Cần: (i) biên âm tiết thật từ MFA để cắt mel; (ii) kiến trúc nhỏ (gợi ý: 3–4 lớp conv + pooling, <0,5M);
(iii) tập train/val riêng (không lẫn clip đánh giá); (iv) **đóng băng** rồi dùng làm perceptual loss trên
mel dự đoán — trọng số bao nhiêu? (gợi ý 0,05–0,2)
> Cần chuyên gia cho con số + quy trình, nếu không sẽ mất thêm một vòng thiết kế.

**E3. Classifier thứ hai để đánh giá: khác seed hay khác kiến trúc?**
Review nói "seed hoặc kiến trúc khác". Xác nhận chọn gì (đề xuất khác **kiến trúc** + seed) và ngưỡng đạt.

**E4. Chống "học để qua classifier":**
Ngoài việc tách classifier đánh giá, có cần augmentation (jitter biên, mask frame) hay giới hạn trọng số
perceptual loss? Cần chốt để tránh lặp lại vấn đề cũ ở tầng sâu hơn.

### F. Bộ metric + eval

**F1. Danh sách metric chốt?** (mel L1/bin, duration MAE ms bỏ SIL, tỉ lệ duration=1 frame, log-F0 RMSE
theo thanh, tone acc, CER) — xác nhận đủ/chưa; cài ở `tools/eval_metrics.py` chạy mỗi 10k bước?
**F2. 100 câu eval:** ai viết? tiêu chí (đủ 6 thanh, coda tắc, từ vay, câu ngắn/dài)? lưu ở đâu?
**F3. ASR đo CER:** model nào (Whisper bản nào? chạy MLX hay CPU?), chuẩn hoá text trước khi tính CER?
**F4.** `duration MAE` và `tỉ lệ 1-frame` có loại SIL đúng không? (nhất quán với D6)
**F5.** `log-F0 RMSE` tính trên frame hữu thanh (theo vuv) — có tách riêng thanh hỏi/ngã/sắc/nặng không?

### G. Việc nhỏ đã thống nhất
- G1. Sửa docs `best.pt` = bước 12.000 (không phải 8k).
- G2. `pyproject.toml`: `pythonpath = ["src", "."]` cho `pytest` trần.
- G3. `duration_scale` cho `--speed`: làm cùng đợt với D4 hay tách?

---

## PHẦN 3 — Tóm tắt 1 dòng cho từng việc đang chờ
1. **Chờ chốt A1** (loại/giữ clip dư video val) + A2/A3 (format lock) → rồi code lock + re-split.
2. **Chờ chốt B1–B3** → thêm log gradient vào v2 trước khi resume (việc rẻ, ~30 phút code + test).
3. Resume chuỗi: `kill -CONT 69421` (cache val → v2). Hoặc kill và chạy tay.
4. Song song v2: chốt C/D/E/F để code F0-DCT + SIL + tone classifier + metric trước smoke.

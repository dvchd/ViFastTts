# AI_REVIEW_ANSWERS_R3 — Phản hồi quyết định vòng 3 + kết quả test nền

> **Người trả lời:** agent cầu nối. **Snapshot:** HEAD `95076eb`; **tất cả luồng tính toán đã dừng**
> (shell chuỗi nền không còn tồn tại — "resume" sẽ đổi thành khởi động lại bằng tay); cache train đủ
> 11.424; `runs/vocoder_codelung_v2` chưa tạo. Các test dưới đây chạy trên script scratch, **không sửa
> repo** (đúng chỉ đạo "phân tích và test trước").
>
> Mã: đã xác nhận **toàn bộ A/B/C/D/E/F/G** theo quyết định của chuyên gia, gồm 4 điểm đổi
> (B1, C2, D1, D6). Dưới đây: (1) phân tích 4 điểm đổi kèm bằng chứng test; (2) tổng hợp test nền;
> (3) gap mới phát hiện cần bổ sung kế hoạch; (4) checklist triển khai.

---

## 1. Bốn điểm bạn đổi — phân tích + bằng chứng

### B1 — cả 4 thành phần đều `retain_graph=True`: **XÁC NHẬN, đã test đúng thứ tự train.py**

Test mô phỏng chính xác trình tự hiện tại (bước D `backward()` → `zero_grad` → bước G → 4×
`autograd.grad(w·loss, fake_crop, retain_graph=True)` → `g_loss.backward()`):

| Thành phần | loss (nhân weight) | \|grad theo `fake_crop`\| |
| --- | --- | --- |
| adv | 0,7218 | 9,52e-05 |
| fm | 0,0026 | 4,41e-03 |
| mel | 6,0938 | 53,39 |
| stft | 9,5302 | 53,79 |
| **tỉ lệ** | | **adv 0,00% · fm 0,00% · mel 49,8% · stft 50,2%** |

`g_loss.backward()` sau cùng chạy bình thường (**26/26 tensor tham số có grad**). Hai lưu ý của bạn
đều đúng: (i) `adv/fm` phải lấy từ forward D **không detach** của bước G; (ii) gradient mel/stft đi
qua bản copy CPU rồi về `fake_crop` — đường autograd của phép copy thiết bị hoạt động (test CPU; đường
MPS dùng cùng cơ chế `Tensor.cpu()`). Ghi chú thứ tự code: nên log grads **sau khi đủ 4 loss, trước
`g_loss.backward()`**; nếu muốn giữ NaN policy hiện tại thì grape grads trước hay sau finite-check đều
được (chi phí ~0).

### C2 — z-score từng hệ số DCT: **XÁC NHẬN, có số liệu thật — và đúng là rất cần**

Đo trên 200 clip train thật (193.747 frame hữu thanh; log-F0 chuẩn hoá mean 4,897 / std 0,205 log-Hz
→ **1 z ≈ 3,54 semitone**), DCT theo mảnh ~15 frame (tạm chia đều vì chưa có MFA):

| Hệ số | std đo được |
| --- | --- |
| c0 | **0,991** |
| c1 | 0,257 |
| c2 | 0,158 |
| c3 | 0,109 |

→ **c0/c1 = 3,9×, c0/c3 = 9,1×** — nếu L1 ngang nhau thì phần hình dạng contour bị át thật, đúng như
bạn nói; z-score từng hệ số bằng std train là cần thiết.

Sai số dựng lại contour khi giữ 4 hệ số (cùng dữ liệu): trung bình **0,117 z ≈ 0,41 semitone**,
p95 **0,362 z ≈ 1,28 semitone** → 4 hệ số đủ mịn cho thanh điệu.

Hai chi tiết kỹ thuật xác nhận:
- **`_frame_positions` hiện tại trả `t/n`** (n=4 → `[0, 0.25, 0.5, 0.75]`), **không phải** `(t+0.5)/n`
  (`[0.125, 0.375, 0.625, 0.875]`). Vì vậy **phải thêm helper mới cho IDCT** đúng như bạn yêu cầu;
  giữ nguyên hàm cũ cho decoder frame-position (hoặc thống nhất một hàm mới nếu không hồi quy).
- n<4: forward chỉ tính k<n; loss mask các hệ số thiếu; dựng lại dùng `k_used`. Công thức nghịch đảo
  bạn đưa đã được kiểm chứng số học khớp với forward `1/n` (sai số thuần do cắt cụt 4 hệ số).

### C3 — shape `[B,N,4]`, decoder nhận contour GT, test export: **ĐÃ TEST EXPORT — KHỚP TUYỆT ĐỐI**

Đã export một wrapper IDCT thuần (đúng các op: `Gather`, `Div`, `Cos`, `Mul`, `ReduceSum`) với
`durations=[[3,0,2]]` (có 1 âm tiết 0 frame):
- `contour` torch vs onnxruntime: **max|diff| = 0,0** (opset 18) → các op cần thiết đều export tốt.

Đề xuất kèm để khỏi phụ thuộc file lúc runtime: **bake mean/std log-F0 và std từng hệ số vào model
dưới dạng buffer** (nạp từ stats lúc train, lưu trong checkpoint, export thành hằng số trong graph) →
ONNX tự chứa; metadata ghi `f0_repr: "dct4"` + hash stats. Nếu bạn muốn runtime tự đọc file stats thì
nói rõ định dạng đường dẫn trong metadata.

### D1 — SIL 3 slot, vocab 71: **XÁC NHẬN, đã rà các điểm chạm code**

Toán: 25/2/16/11/7 + 4 + 6 = **71** ✓. Các file phải sửa (rà bằng grep, không sót theo cấu trúc size
hiện tại vì mọi nơi dùng `len(vocabs.*)`):

| Nhóm | Điểm chạm |
| --- | --- |
| Kiểu | `types.py` (`Tone.NONE`; `ParsedSyllable` giữ nguyên 5+2 slot) |
| Parser | `syllable.validate()` (nhánh SIL), `orthography` (biểu diễn canonical của SIL — xem câu hỏi §3.1) |
| Chèn SIL | `pipeline.py:61–71` (dấu câu → SIL sau âm tiết mang dấu), gộp SIL liền nhau |
| Vocab | `vocab.py` (3 bảng thêm SIL/NONE) |
| Phân tích | `g2p.syllable_to_phonemes` (guard), `segments.expand_syllable` (guard), `lexicon.strict_warnings`, `manifest.coverage_report` (tách SIL), `tools/parser_audit.py` (loại SIL khỏi matrix) |
| Model/loss | `losses.py` (tone ignore_index; mask f0/tone/phonation tại SIL — D6), `dataset.py` (durations 0, target SIL-0-frame — xem §3.2) |
| MFA | `mfa_build_corpus.py` không đổi (SIL do MFA tự thêm `sil/sp`); **cần tool TextGrid→durations mới** (§4.1) |
| Export/runtime | metadata vocab version; `duration_scale` (D4); bỏ `_stretch_mel` |

### D6 — giữ vuv/energy tại SIL: **ĐỒNG Ý, nhưng có 1 lỗ hổng cần chốt**

Nguyên tắc đúng, và ủng hộ độ sạch khoảng lặng. Nhưng SIL **duration = 0 frame** rất thường gặp
(đầu/cuối câu, dấu câu sát nhau): dataset hiện điền mặc định `energy = 0` khi không có frame, mà
**0 không phải im lặng** trên thang log-power (dải đo được của cache: **−13,82 … 0,05**). `vuv = 0`
thì đúng sẵn.
- **Cần bạn chốt (a)/(b):** (a) target energy cho SIL-0-frame = percentile thấp của frame im lặng
  (tính từ MFA, lưu hằng số vào stats); (b) mask luôn energy tại SIL-0-frame, chỉ giữ vuv.
- SIL có >0 frame: trung bình frame như hiện tại — không vấn đề.

---

## 2. Tổng hợp test nền (đã chạy, không sửa repo)

| # | Nội dung | Kết quả |
| --- | --- | --- |
| 1 | B1: 4×`autograd.grad(retain_graph=True)` + `backward()` đúng thứ tự train.py | OK, 26/26 tensor có grad; tỉ lệ adv/fm ~0% · mel 49,8% · stft 50,2% (random init) |
| 2 | C2: DCT-II `1/n` + nghịch đảo + cắt 4 hệ số | Công thức khớp; sai số contour thật 0,41 semitone (TB), p95 1,28 |
| 3 | C2: thang hệ số trên F0 thật | c0/c1 = 3,9× · c0/c3 = 9,1× → z-score từng hệ số |
| 4 | C2: `_frame_positions` | trả `t/n` → cần helper `(t+0.5)/n` mới cho IDCT |
| 5 | C3: export ONNX IDCT (Gather/Div/Cos/Mul/Sum), durations [3,0,2] | torch vs ORT **0,0** |
| 6 | D4: export ONNX quy tắc `where(is_sil,…)` + `duration_scale` | torch vs ORT **0,0**; SIL→0 frame, khác ≥1, scale áp đúng |
| 7 | (vòng 2) LengthRegulator duration=0 | torch OK; ONNX wrapper khớp 2,4e-7 |
| 8 | (vòng 2) DataLoader vocoder cache ấm | 4,2 ms/batch (w=6) vs 659 ms/bước |

---

## 3. Gap mới phát hiện — cần bổ sung vào kế hoạch

### 3.1. Biểu diễn `Orthography` của SIL
`compose_canonical()` và test song ánh (`test_spelling_bijection.py`) giả định mọi âm tiết dựng lại
được từ 5 phần. SIL không đến từ chữ viết. Cần chốt: (a) loại SIL khỏi invariant spelling + định nghĩa
`Orthography(onset="<sil>", …)`; hoặc (b) thêm dạng canonical `"<sil>"` chính thức. Ảnh hưởng test
round-trip D2.

### 3.2. Target cho SIL-0-frame (D6) — xem §1.D6, cần chốt (a)/(b).

### 3.3. Tool chuyển MFA → `durations` (mắt xích còn thiếu, chặn C2-target/D3/F1)
Chưa có tool nào đọc TextGrid. Đề xuất `tools/mfa_to_durations.py`:
- Map word interval → các âm tiết theo `labs` (dùng `syllable_index.json` của corpus builder).
- Map silence (`sil`/`sp`) → SIL theo luật D2 (vị trí) + D5 (≥150 ms → loại/cắt; <150 ms → gộp âm tiết trước).
- Quy đổi **biên tích luỹ**: `frame_b = round(t_b · sr / hop)` cho mọi biên, duration = hiệu, ép biên cuối = T.
- **Assert** `sum(durations) == T`, non-SIL ≥1; vi phạm → loại clip + log.
- Ghi `durations` vào manifest (tách train/val), report: số clip loại, phân bố độ dài phone, tỉ lệ phone 1 frame, tỉ lệ silence >150 ms.
- Test: TextGrid giả nhỏ + ca biên (SIL 0 frame, biên lệch làm tròn, clip dài).

### 3.4. `log(0)` khi tính log-F0
Clip toàn vô thanh → `log(0) = -inf`. Ngoài việc bị loại bởi voiced-ratio < 0,3, tool stats + dataset
nên clamp/lọc trước DCT để không sinh NaN.

### 3.5. Stats F0/Energy (C1/C4)
`tools/compute_f0_stats.py` (mới): đọc F0 cấp frame **từ cache** trên split train; mean/std log-F0
**chỉ frame hữu thanh**; std từng hệ số DCT; đề xuất lưu thêm std của energy z-score (C4) trong cùng
file hoặc tool `compute_energy_stats.py`. Config top-level `f0_norm:` / `energy_norm:` — **không** đụng
khoá cache (đã xác nhận `cache.py:44`).

### 3.6. Đối tượng train của classifier đánh giá (E3)
Bạn chốt classifier đánh giá "chạy trên audio sau vocoder". Nếu train nó trên **audio thật** rồi test
trên **audio sinh**, lệch domain. Đề xuất train trên **copy-synthesis** (mel thật → vocoder, train
split) để train/test cùng miền "audio sau vocoder". Xác nhận giúp.

### 3.7. Bộ clip eval cố định + lock_version
- 200 clip val cố định: chọn tất định + lưu `data/codelung/eval_clips_v1.json` + hash (như 100 câu).
- `lock_version` nên ghi **cả vào `resolved_config.json`/tên run** (không chỉ output eval) để truy vết
  checkpoint ↔ split ↔ metric.

### 3.8. Trạng thái "resume" đã đổi
Vì bạn đã dừng hết, shell SIGSTOP không còn → không còn `kill -CONT`. Khi code xong, chuỗi sẽ được
**khởi động lại bằng tay** (cache val → `train_vocoder_chunked`). Vì vậy cảnh báo "không sửa script
nền" tự hết hiệu lực; nhưng vẫn giữ nguyên tắc: chỉ chạy sau khi lock + gradient log đã code xong +
test xanh.

---

## 4. Checklist triển khai đề xuất (mỗi mục kèm test)

| # | Việc | File | Test |
| --- | --- | --- | --- |
| 1 | Split lock (video + cap + clip list + hash + version), cap rule sort-id/150 | `data/split.py` | lock idempotent; mismatch → raise; `--force` tăng version |
| 2 | `split_manifest` (cảnh báo khi chưa lock) / `reparse_overrides` (tôn trọng lock, chặn tham số lệch) | `tools/*` | tái tạo đúng clip list; không đổi video khi reparse |
| 3 | Re-split thật: val 2.235 clip (16 video, cap 150) · **train 7.024 clip/27,7h** · holdout 3.956 clip → `holdout_rest.jsonl` | data/codelung | đếm khớp; train ⊂ cache cũ (100% ấm); val cần ~510 file cache mới |
| 4 | Log gradient (4 cột) + `d_real_mean`/`d_fake_mean` vào `history_vocoder.csv` | `train.py` | test parse history; dry-run nhỏ |
| 5 | `mfa_to_durations.py` + assert + report | `tools/` | TextGrid giả + ca biên |
| 6 | `compute_f0_stats.py` (+energy) | `tools/` | stats ổn định; clip toàn vô thanh không NaN |
| 7 | SIL (types/pipeline/vocab/guards/loss mask/ignore_index/orthography) | nhiều file | round-trip text→IDs; mask; SIL-0-frame; coverage tách SIL |
| 8 | DCT4 + IDCT + `(t+0.5)/n` + `duration_scale` + export/runtime (bỏ `_stretch_mel`) | model/export/runtime/infer | test nền đã có; thêm test export↔torch↔ORT |
| 9 | `eval_metrics.py` + 200 clip + 100 câu + PhoWhisper/mlx-whisper | `tools/`, `eval/` | chạy trên model cũ để lấy baseline biết trước |
| 10 | Tone classifier loss + classifier đánh giá (E2/E3) | `tools/`, `models/` | ngưỡng ≥85% trước khi dùng |
| 11 | Docs: best.pt=12k; `pythonpath` pytest; ghi rõ 5%/60% là heuristic; ngưỡng nghiệm thu mới | docs | — |

**Thứ tự bạn chốt vẫn giữ:** (1) lock+re-split+gradient log+test → (2) khởi động lại chuỗi (val → v2)
→ (3) song song: MFA → converter → C/D → metric → E. Smoke chỉ khi đủ MFA+SIL+DCT+scale+metric và
`tone: 0`.

---

## 5. Câu hỏi cần chốt (ngắn — chỉ 5 điểm)

1. **D6:** SIL-0-frame → energy target theo (a) percentile thấp của im lặng, hay (b) mask energy? (vuv=0 giữ nguyên)
2. **3.1:** Orthography của SIL — loại khỏi invariant spelling hay định nghĩa canonical `"<sil>"`?
3. **3.6:** Classifier đánh giá (E3) train trên copy-synthesis (đề xuất) hay audio thật?
4. **C3:** bake stats F0/std vào buffer model (đề xuất, ONNX tự chứa) hay runtime đọc file stats?
5. **3.7:** `lock_version` ghi vào `resolved_config.json` của run nữa không (ngoài output eval)?

*Hết. Sau khi bạn trả lời 5 điểm này và ra lệnh, agent sẽ code theo checklist, chạy test, rồi mới đề
xuất khởi động lại chuỗi.*

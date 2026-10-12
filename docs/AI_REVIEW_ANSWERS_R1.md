# AI_REVIEW_ANSWERS_R1 — Trả lời vòng review 1 (kết quả xác minh trong repo)

> **Người trả lời:** agent cầu nối (đã đọc code, đo trực tiếp trên máy, chạy test).
> **Snapshot:** 2026-10-12, HEAD `fdc1405` — bao gồm `1bd127f` (rev.6: checkpoint sạch + demo theo mốc)
> và `fdc1405` (env MFA qua conda`.mamba-mfa`). Working tree sạch; `ruff` + `mypy` sạch;
> **2257 test pass** (`python -m pytest`). Có tiến trình nền đang chạy: cache feature
> (`cache/codelung`, ~7.980/13.215 file lúc soạn) → vocoder v2 chưa khởi động.
>
> Mọi kết luận dưới đây kèm **trích dẫn file:dòng** và **số đo thật**; chỗ nào chưa kiểm được thì
> ghi rõ "chưa kiểm được + cách kiểm".

---

## 0. Bảng tóm tắt 5 câu hỏi

| # | Câu hỏi | Kết luận | Bằng chứng chính |
| --- | --- | --- | --- |
| **Q1** | `tone_logits` tính từ tensor nào? | **XÁC NHẬN giả thuyết** — từ **encoder state** `x`; loss tone ≈ 0, acc 99–100% | `models/vifasttts.py:63,109,145`; đo trên checkpoint thật |
| **Q2** | F0 có chuẩn hoá trước L1? Hz hay log? | **XÁC NHẬN** — L1 trên **Hz thô**, không chuẩn hoá; đo được f0 loss ≈ 33 (lớn nhất sau A1) | `features.py:195–219`, `dataset.py:147`, `losses.py:60` |
| **Q3** | MFA version + topology, tối thiểu mấy frame/phone? | **BÁC BỎ một phần** — MFA **3.4.3** (conda-forge, vừa cài); topology mặc định **cho phép 1 frame/phone (10 ms)**, không phải 3 frame | `dictionary/mixins.py:98–99,730–737`; default beam 10/retry 40 |
| **Q4** | Vocoder mel L1 trên mel chuẩn hoá? Tỉ lệ gradient? | **XÁC NHẬN** mel L1 trên mel **chuẩn hoá**; đo random-init: **mel 60% · stft 40% · fm 0,02% · adv ~0%** (v2 chưa chạy nên chưa có log 2k bước) | `features.py:164–167`, `train.py:509–517`; đo gradient |
| **Q5** | Vocos export hop/ratio; iSTFT numpy khớp torch? | **BÁC BỎ giả thuyết** — Vocos trong repo **có upsample ×4** (hybrid iSTFTNet), metadata hop 256/ratio 4 **đúng**; ONNX vs torch 7e-9, numpy iSTFT 2,8e-9 | `istftnet.py:118–141,58–70`, `export.py:197–204`, `runtime/istft.py` |

---

## Q1 — `tone_logits` tính từ tensor nào?

**KẾT LUẬN: XÁC NHẬN.** Classifier nằm trên **encoder state**, và tone đã là input (slot 4) → nó chỉ
cần sao chép. Đo thật: loss → ~0 và accuracy → 100%.

**Bằng chứng code:**
- `self.tone_classifier = nn.Linear(h, 6)` — `src/vifasttts/models/vifasttts.py:63`
- `x = self.encoder(self.embedding(features), padding_mask)` — `models/vifasttts.py:109`
- `outputs["tone_logits"] = self.tone_classifier(x)` — `models/vifasttts.py:145` → **từ `x` (encoder), không phải decoder/mel**
- Input tone: `moved["tone_ids"] = moved["features"][..., 4]` — `train.py:69`
- Loss: `losses.py:65–68` (CE trên `tone_logits` vs tone ID lấy từ chính input)

**Bằng chứng đo** (3 clip val thật, code loss hiện tại, checkpoint của run cũ):

| Checkpoint | tone loss (unweighted) | tone accuracy trên val | ghi chú |
| --- | --- | --- | --- |
| `runs/small_codelung/best.pt` (step **12.000**) | **0,0096** | **99,3%** | weight 0,25 → đóng góp ~0,0024 |
| `runs/small_codelung/last.pt` (step 21.000) | **0,0014** | **100%** | gần như triệt tiêu |

**Lưu ý phụ:** `history.csv` chỉ có cột `step,train_total,train_mel,val_total` — **không có cột tone**
như câu hỏi giả định. Cross-check gián tiếp: `train_total − train_mel` (các loss còn lại, có trọng số)
= 41,1 (bước 12k) và 33,2 (bước 21k), khớp với đo trực tiếp (~36–39).

**Hệ quả đề xuất:** đặt classifier lên **mel dự đoán**, pool theo âm tiết bằng `durations`/`frame_mask`
đã có trong outputs (khả thi không cần đổi kiến trúc) — hoặc lên decoder state. Đây là thay đổi *thiết kế*,
cần chốt vì hiện loss tone vô nghĩa.

> Ghi chú kèm: `best.pt` thật ra ở **bước 12.000** (val 277,499) — không phải 8k như một số chỗ trong
> docs ghi; `val` thấp nhất trong `history.csv` là dòng 12.000.

---

## Q2 — F0 có được chuẩn hoá trước masked L1 không? Hz hay log?

**KẾT LUẬN: XÁC NHẬN giả thuyết** — F0 là **Hz thô**, không chuẩn hoá, L1 trực tiếp; và sau A1 nó
**lấn át tổng loss**.

**Bằng chứng code:**
- `estimate_f0_pyworld` trả `torch.from_numpy(f0).float()` — đơn vị **Hz** — `data/features.py:219`;
  `interpolate=True` nội suy tuyến tính qua đoạn vô thanh cho f0 (vuv giữ 0) — `features.py:200,216–218`.
- Dataset gom cấp âm tiết bằng **trung bình Hz**: `f0[i] = frame_f0[start:end].mean()` — `data/dataset.py:147`.
- Loss: `"f0": _masked_mean((outputs["f0"] - batch["f0"]).abs(), syllable_mask)` — `losses.py:60`
  (không log, không z-score, không mask theo vuv).
- Config `small_codelung.yaml` mục `f0:` chỉ có `method/f_min/f_max/interpolate` — **không có chuẩn hoá**.

**Bằng chứng đo** (cùng batch val, checkpoint cũ, code loss hiện tại):

| Thành phần (unweighted) | best.pt @12k | last.pt @21k | weight | nhận xét |
| --- | --- | --- | --- | --- |
| mel | 2,666 | 2,748 | 1,0 | đã mean/bin (A1) |
| duration | 0,040 | 0,018 | 1,0 | |
| **f0 (Hz)** | **33,846** | **33,321** | 1,0 | **chiếm ~87% tổng có trọng số** |
| vuv | 0,889 | 0,771 | 0,5 | |
| energy | 4,488 | 4,361 | 0,5 | log-power |
| tone | 0,0096 | 0,0014 | 0,25/0,1 | (xem Q1) |
| **TỔNG có trọng số** | **39,24** | **38,65** | | f0 ≈ 86–87% |

**Cảnh báo quan trọng (chưa thấy trong review):** run cũ dùng `autocorr` (không interpolate), nhưng
config **sắp chạy dùng `pyworld + interpolate: true`** → f0 mục tiêu ở vùng vô thanh là **giá trị nội suy
(khác 0)** trong khi vuv=0. Vì loss f0 không mask theo vuv, model sẽ bị ép khớp giá trị "bịa" này.
Cần chốt: (a) mask f0 loss theo `vuv`, hay (b) log-F0 + z-score toàn cục, hay (c) giữ nguyên nhưng giảm weight.

---

## Q3 — MFA version & topology phone, tối thiểu bao nhiêu frame/phone?

**KẾT LUẬN: BÁC BỎ giả thuyết "3 frame/phone là mặc định".** Máy đã có **MFA 3.4.3** (cài qua
micromamba/conda-forge tại `.mamba-mfa/` — lý do: `kalpy` không có wheel ARM64 cho pip). Đọc code
sinh topology của MFA 3.4.3: **topology mặc định cho non-silence cho phép tối thiểu 1 frame (~10 ms)**.

**Bằng chứng code MFA** (đường dẫn trong env vừa cài):
`.mamba-mfa/lib/python3.14/site-packages/montreal_forced_aligner/`:
- Mặc định số state: `num_silence_states: int = 5`, `num_non_silence_states: int = 3`
  — `dictionary/mixins.py:98–99`.
- Sinh chuyển trạng thái (`write_topology`): với mặc định `min_states=1, max_states=3`
  (`min_states != max_states`), **state đầu được chuyển trực tiếp tới các state `range(min_states, max_states+1)`
  = {1,2,3}**, trong đó state 3 là state kết thúc → **tối thiểu 1 frame phát âm**.
  — `dictionary/mixins.py:730–737` (xem thêm nhánh `min_states == max_states` → khi đó mới tối thiểu
  `num_states` frame, như 3 frame).
- Có hỗ trợ **topology tùy biến theo phone** qua file YAML (`min_states`/`max_states` per phone):
  `dictionary/multispeaker.py:214–240`.
- Beam mặc định của `mfa align`: **beam 10 / retry_beam 40** — `alignment/mixins.py:73–74`
  (nhỏ hơn nhiều so với `--beam 100 --retry_beam 400` mà review đề xuất; con số đó là của MFA 2.x).
  `lattice_beam = 6` — `alignment/base.py:97`.

**Còn thiếu (chưa kiểm được):** chưa tải acoustic/dictionary model nên **chưa chạy align thử 200 clip**
để đếm `unaligned` / `beam too narrow`. Khi có model, cách kiểm đúng theo yêu cầu review:
```bash
.mamba-mfa/bin/mfa align --help            # xem beam mặc định của đúng version
# chọn 200 clip nhanh nhất (âm tiết/giây cao nhất), align với --beam 10 --retry_beam 40
# đếm: số clip fail log, số phone < 3 frame, phân bố độ dài phone (ms)
```
*Caveat:* topology sinh ra phục vụ dictionary; nếu acoustic model tải sẵn có topo riêng thì cần xác nhận
lần align thật (kết quả log sẽ cho biết). Kết luận "min 1 frame" áp cho **đường dictionary/topology mặc định
của MFA 3.4.3**, không phải cho mọi model tải sẵn.

---

## Q4 — Vocoder mel L1 trên mel chuẩn hoá hay thô? Tỉ lệ gradient?

**KẾT LUẬN: XÁC NHẬN mel L1 tính trên mel ĐÃ CHUẨN HOÁ** (log-magnitude + z-score per-bin).
Run v2 **chưa khởi động** (cache còn chạy) nên chưa có log 2k bước đầu; tôi đo gradient bằng
diagnostic riêng theo đúng cấu hình sắp chạy.

**Bằng chứng code:**
- Dataset vocoder lấy mel qua `extract_features` → áp `normalize_path` — `data/features.py:164–167`
  (`mel = (mel - mean) / std`); `VocoderDataset` gọi `extract_features(row["audio_path"], self.config)`
  — `data/dataset.py:231–237`.
- Bước G: `g_mel = (mel_fake - mel_target).abs().mean()` với cả hai đều lấy từ cùng `mel_config`
  (`train.py:509–517`) → **cùng thang chuẩn hoá**.
- Trọng số sắp chạy: `mel_weight: 5.0`, `feature_matching_weight: 2.0`, `adversarial_weight: 1.0`,
  `stft_weight: 2.0` — `configs/small_codelung.yaml`.

**Đo gradient generator** (Vocos 12,07M + MPD/MRD ch=16, batch=1, crop 64, CPU, **random init**
→ chỉ đọc bậc độ lớn; mỗi thành phần backward trên graph mới):

| Thành phần | loss (nhân weight) | \|grad_G\| (nhân weight) | Tỉ lệ \|grad\| |
| --- | --- | --- | --- |
| adv | 0,687 | 3,4e-05 | **~0,00%** |
| fm | 0,0047 | 6,1e-03 | **0,02%** |
| mel | 6,306 | 17,76 | **60,2%** |
| stft | 8,930 | 11,72 | **39,7%** |

Diễn giải: ở random init, discriminator chưa học (logits std ≈ 0,035) nên adv/fm gần như không đóng góp;
**mel + stft chiếm 99,98% gradient**. Điều này **không ủng hộ** việc nâng `mel_weight` lên 10–15 ngay;
ngược lại cần **theo dõi gradient trong 2k bước đầu của v2** để xem adv/fm có lớn dần không.
Đề xuất: thêm log `|grad|` từng thành phần vào `history_vocoder.csv` (hiện chỉ có loss).

**Tham chiếu run cũ** (lite, mel log-power, weight 45): `history_vocoder.csv` chỉ có
`step,g_total,g_adv,g_mel,d_loss` — 3 dòng cuối: g_adv 0,46–1,44; g_mel 0,14–0,54; d_loss 1,14–1,47.
**Không so sánh trực tiếp được** với run mới (đơn vị mel đã đổi).

---

## Q5 — Vocos export: hop trong graph là 256 hay 64? Metadata ghi gì? iSTFT numpy khớp torch?

**KẾT LUẬN: BÁC BỎ giả thuyết.** Trong repo này `ISTFTNetVocos` **là hybrid**: backbone kiểu Vocos
(ConvNeXt) nhưng **vẫn upsample ×4 như iSTFTNetLite** (`self.up` là `ConvTranspose1d(stride=upsample_ratio)`),
nên graph xuất ra phổ có số frame = `T × upsample_ratio`, và metadata `hop_length = 256` (hop mel),
`upsample_ratio = 4` là **đúng và nhất quán** — không phải sót metadata của bản lite.

**Bằng chứng code:**
- `ISTFTNetVocos` dùng chung `_ISTFTNetBase.spectrogram_parts`: `self.out(self.post(self.up(self.net(self.pre(...)))))`
  — `models/istftnet.py:58–65`; `self.up = nn.ConvTranspose1d(..., stride=upsample_ratio)` — `models/istftnet.py:130–135`.
- Metadata: `hop_length = config["mel"]["hop_length"]`, `upsample_ratio` — `export.py:197–204`.

**Đo thực** (model Vocos nhỏ 16 hidden, mel T=8, n_fft 256, hop 64, ratio 4):
| Kiểm tra | Kết quả |
| --- | --- |
| Metadata đọc lại từ graph | `n_fft=256, hop_length=64 (hop mel của config test), sample_rate=24000` |
| Số frame phổ | 32 = T×4 → **có upsample ×4** |
| Spec ONNX vs spec torch | `max|diff| real = 1,34e-7`, `imag = 1,19e-7` |
| Waveform ONNX vs waveform torch | `max|diff| = 7,0e-9` |
| numpy iSTFT vs torch iSTFT (cùng spec) | `max|diff| = 2,8e-9` (tương đối 1,8e-7); test `tests/test_runtime_istft.py` atol 1e-4 với tín hiệu thật cũng xanh |

→ Đường export vocoder + iSTFT numpy **khớp torch ở mức số học floating-point**; không có gì phải sửa
cho Q5. (Với config thật `hop_length` metadata sẽ là 256.)

---

## Xác nhận nhanh các điểm khác của vòng review

| # | Điểm review | Kết quả xác minh |
| --- | --- | --- |
| 1 | Tone classifier vô dụng | **Đúng** — xem Q1 (đo: loss 0,0014–0,0096; acc 99–100%) |
| 2 | Chưa có SIL → silence bị dồn vào âm tiết kề | **Đúng** — không có `SIL`/`<sil>` trong `src/`; vocab vẫn 68 ID (`vocab.py:31–41`). Chưa có chính sách gán silence. |
| 3 | Quy đổi frame MFA (10 ms) → hop 256 (10,67 ms) | **Chưa có code nào làm việc này** (MFA chưa từng chạy). Đề xuất "quy đổi biên tích luỹ + assert sum(durations)==T" là hợp lý và **chưa tồn tại** — nên ghi thành yêu cầu kỹ thuật trước khi smoke. |
| 4 | Topology 3 frame/phone của MFA | **Một phần sai cho MFA 3.4.3** — xem Q3 (default cho phép 1 frame; beam mặc định 10/40). |
| 5 | F0 trung bình làm mất contour; DCT nên bắt buộc | **Đúng** — không có DCT trong `src/`; F0 là trung bình Hz/âm tiết (`dataset.py:147`). Xem thêm cảnh báo interpolate-vs-vuv ở Q2. |
| 6 | Val loss 274→305 không so sánh được | **Đúng và đã xác nhận nguyên nhân:** `runs/small_codelung/resolved_config.json` **không có** `log_magnitude`/`normalize_path` (mel cũ log-power) và thời điểm đó mel loss là **sum-over-bins**: `history` ghi train_mel 217–229, trong khi code hiện tại `mean_over_bins=True` cho mel ≈ 2,67. → Cần bộ metric chuẩn (mel L1/bin, duration MAE ms, F0 RMSE, CER). |
| 7 | `--speed` lệch giữa 2 đường | **Đúng** — PyTorch: chia duration rồi forward lại (`infer.py:111–112`); ONNX: `_stretch_mel` nearest-frame (`onnx_engine.py:171–176`). Giải pháp `duration_scale` là khả thi (thêm 1 input scalar vào graph acoustic, áp trước `round(...)`). |
| 8 | mel_weight=5.0 có thể thấp | Chưa kết luận được từ log v2 (chưa chạy). Đo random-init cho thấy mel+stft đã chiếm ~100% gradient → **không nên nâng mù**; cần log 2k bước đầu (xem Q4). |
| 9 | Val chỉ 4 video | **Đúng** — đếm thực: 1.791 clip / **4 video** (972 · 437 · 322 · 60). |
| 10 | Sửa `pytest` trần bằng `pythonpath = ["src", "."]` | **Đúng** — 4 module import `tools.*` fail khi chạy `pytest` trần; `python -m pytest` qua 2257. Sửa 1 dòng trong `pyproject.toml`. |

### Cập nhật hạ tầng MFA (mới so với hồ sơ)
- Env conda riêng tại `.mamba-mfa/` (micromamba; Python 3.14; `mfa 3.4.3`; conda-forge).
- Lý do: **không có wheel pip ARM64 cho `kalpy`** → pip `.venv-mfa` bị bỏ; `.gitignore` đã thêm `.mamba-mfa/`.
- Chưa tải acoustic/dictionary model; corpus `data/codelung/mfa/` (wavs/labs/lexicon/phones 121) đã sẵn sàng.

---

## Đề xuất thứ tự xử lý (rút từ kết quả trên)

1. **Chốt chính sách SIL + silence** trước khi train acoustic với `durations` MFA (review #2) — có thể
   bắt đầu bằng phương án "gán silence vào âm tiết mang dấu câu/SIL ảo" để đo được ngay.
2. **Chuẩn hoá F0** (log + z-score, hoặc rất ít nhất mask theo vuv) + quyết định `interpolate: true/false`
   cho pyworld; hiện f0 chiếm ~87% tổng loss sau A1 (Q2).
3. **Sửa tone classifier** đo trên mel dự đoán pool theo duration (Q1) — mở lại quyết định F3 có bằng chứng.
4. **Quy đổi frame MFA theo biên tích luỹ + assert** tổng (review #3).
5. **Bộ metric đánh giá độc lập loss** (mel L1/bin, duration MAE ms, F0 RMSE theo thanh, CER) — bắt buộc
   trước smoke (review #6, A1).
6. **`duration_scale` trong graph ONNX** để đồng nhất `--speed` (review #7).
7. **Log gradient từng thành phần** trong 2k bước đầu của v2 (Q4) trước khi chỉnh `mel_weight`.
8. **Chia lại val 12–20 video** (review #9) + `pythonpath` cho pytest (review #10) — hai việc rẻ.
9. Sau khi có model MFA: align thử 200 clip nhanh nhất, đếm unaligned/phone dài ≤2 frame (Q3) trước khi
   tin ngưỡng ">90% clip giữ".

---

*Trả lời bởi agent cầu nối. Nếu cần xác minh thêm (ví dụ sau khi v2 chạy 2k bước, hoặc sau align thử MFA),
gửi lại câu hỏi theo định dạng §0.4 của `AI_REVIEW_BRIEF.md`.*

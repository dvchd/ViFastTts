# V18.40 — Agent 2: xác nhận artifact ONNX v1837, không training

> **Cập nhật v18.41:** frontend OOV đã được kiểm và sửa (`OnnxAcousticModel.features_from_text`
> mặc định `resolve_oov=True`); xem `V1841_agent2.md`. Kết quả parity số học và ngoại lệ
> spectrum ở tài liệu này giữ nguyên.

> **Trạng thái: DỪNG SAU V18.40, CHỜ ĐỌC.** Không training, không đổi production,
> không chuyển ONNX. Kết quả: **acoustic/front-end/end-to-end đạt**; **vocoder 11/12**
> do một bin phổ vượt tolerance tuyệt đối đã đăng ký trước; **chưa tuyên bố sẵn sàng chuyển**.
>
> **Commit gốc của vòng**: `266f5a9` · **Tệp đặc tả**: yêu cầu A0 → Agent 2 trong chat
> (không có `V1840_agent0.md`) · **Test**: clean worktree 36 passed; working tree full 2.556 passed;
> `ruff` sạch.

---

## 1. Khóa v18.39

- Giữ acoustic **P2 EMA**.
- Giữ crop fixed vì **loại padding**, không tuyên bố cải thiện chất lượng.
- Đóng audit UTMOS; protocol seeded 5 seed là protocol hiện hành khi cần.
- Không chấm lại toàn bộ benchmark lịch sử.
- Không training, không đổi loss/kiến trúc, không chấm tai.

---

## 2. Bộ 12 input (chọn trước khi chạy)

`artifacts/bench/v1840_onnx_parity/inputs.json`:

- **10 row manifest**: 4 short + 3 mid + 3 long, phủ `,` `.` `?` và không dấu câu
  (lấy từ bộ pause 12 câu v18.37, bỏ 1 mid + 1 long).
- **1 âm tiết**: `Chào.`
- **1 custom reading**: `VBA.` → frontend đọc `vi be a.` (3 âm tiết).

Mỗi input ghi text, `n_syllables`, IDs 7 thành phần, SHA256 của features.
**Không dùng dữ liệu để chọn checkpoint**; không chấm BAK/UTMOS.

---

## 3. Tiêu chí đăng ký trước

Khai báo tại `CRITERIA` trong `tools/v1840_onnx_parity.py`, trước khi chạy:

| Tầng | Tiêu chí |
|---|---|
| Acoustic | duration/slot/lengths **exact**; mel max ≤ 2e-3, mean ≤ 2e-4; các output liên tục khác max ≤ 2e-3; finite |
| Vocoder | length exact; spectrum real/imag max ≤ 2e-2, mean ≤ 2e-3; waveform max ≤ 5e-4, RMSE ≤ 5e-5 |
| End-to-end | length exact; max ≤ 2e-3, RMSE ≤ 2e-4; RMS ratio 0,99–1,01; clipping_extra ≤ 0 |
| Chung | frontend features exact; provenance/mel hash khớp; lặp cùng runtime exact |

Không đổi tiêu chí sau khi xem kết quả.

---

## 4. Tầng A — Acoustic

- 12 input × 3 cấu hình `(speed, gate)` = **36/36 pass**:
  `(1,0; 0,5)`, `(0,75; 1,01)`, `(1,25; 0,5)`.
- Cùng `features`, `speed`, `gate_threshold`; PyTorch P2 EMA vs ONNX `p2ema_acoustic.onnx`.
- **Discrete exact**: `durations`, `slot_durations`, `lengths` — tất cả 36/36.
- Sai số liên tục lớn nhất qua 36 phép:

| Output | max abs |
|---|---:|
| `mel` | **1,53e-05** |
| `f0` | 4,77e-07 |
| `log_duration` | 1,91e-06 |
| `pause_gate` | 1,25e-06 |
| `vuv_logits` | 3,58e-06 |
| `energy` | 9,54e-07 |
| `phonation` | 1,91e-06 |
| `tone_logits` | 9,54e-06 |

Không NaN/Inf; shape khớp. **Tầng A đạt.**

---

## 5. Tầng B — Vocoder

- Cùng mel tensor lấy từ acoustic PyTorch (speed 1, gate 0,5), 12 input.
- So `spectrogram_parts` (real/imag) trước iSTFT và waveform sau iSTFT:
  PyTorch pretrained vs ONNX `vocos_pretrained_vocoder.onnx`.
- **11/12 pass**; 1 input fail criterion spectrum:
  `2b65WyiFC9k_0030__s03`.
  - `spectrum_real` max abs **0,02336** tại bin `(20, 66)` (ref −4,472832, got −4,449474;
    magnitude ~6,31; sai tương đối ~0,52%);
  - `spectrum_imag` max abs **0,01777** tại `(21, 66)` (ref −3,064308, got −3,082081;
    magnitude ~4,35; ~0,58%);
  - mean abs real/imag ~**9,8e-06 / 9,7e-06**; p99.99 ~0,0070 / 0,0048.
  - Waveform cùng input: length **52.224 = 52.224**, max abs **3,23e-04** (≤5e-4),
    RMSE **7,25e-06**, RMS ref/got **0,037166 / 0,037166**, peak **0,264139 / 0,264139**,
    clipping 0/0.
- 11 input còn lại pass; waveform max toàn bộ ≤ **3,23e-04**, RMSE ≤ **7,25e-06**,
  spectrum mean ≤ ~1,7e-05.
- **Kết luận tầng B**: không phải mismatch provenance; là **một outlier số học thưa**
  ở spectrum, vượt ngưỡng tuyệt đối 2e-2, trong khi waveform và end-to-end vẫn đạt.
  **Không sửa runtime**, không đổi tiêu chí sau khi thấy kết quả.

---

## 6. Tầng C — End-to-end

- Cùng **chunk mode** cho cả hai runtime (không so sentence/steady để tránh khác biệt
  trim của ONNX); cùng `chunk_text(max_chars=400)`, `pause_seconds`, fade, PCM16,
  `target_lufs=None`, `peak_normalize=False`, α=0, `gate=0,5`.
- **12/12 pass**:
  - length exact cả 12;
  - max abs lớn nhất **2,14e-04** (≤2e-3);
  - RMSE lớn nhất **7,74e-06** (≤2e-4);
  - RMS ratio **0,999998–1,000001**;
  - clipping_extra **0**;
  - sample rate 24.000 cả hai.
- **Frontend**: 12/12 features exact (cùng chunk/parse). Lưu ý: ONNX
  `features_from_text` không truyền `resolve_oov=True`; bộ này không có OOV nên không lộ
  khác biệt. Đây là **rủi ro chưa kiểm** cho text OOV, không phải lỗi parity đã thấy.
- **Tất định**: chạy lại cùng input trên cùng runtime — PyTorch **12/12 exact**,
  ONNX **12/12 exact**.

---

## 7. Metadata / provenance

`artifacts/bench/v1840_onnx_parity/result.json` + `v1837_provenance.json`:

- Acoustic ONNX = P2 EMA: **180/181 tensor exact**, max abs **1,57e-05** (`f0_log_mean`);
- Vocoder ONNX = Vocos pretrained: **63/63 tensor exact**, max abs **0**;
- `mel_hash` cả hai = `e065776b45b74b874c483dfdc68c1f9c5e45918b`;
- acoustic `checkpoint_sha256` = `44dd0da6…` (P2);
- vocoder `checkpoint_sha256` = `pretrained_sha256` = `97ec976a…`;
- `stats_sha256` = `cab53eb6…`; `export_version=v18.37`;
- SHA file khớp `v1837_provenance.json`;
- ONNX cũ giữ nguyên: `v32_adv_p2_2k.onnx` + `v32_adv_p2_2k_vocoder.onnx` (vocoder E4-C).

---

## 8. Kết luận & chuyển đổi

- **Acoustic + frontend + end-to-end đạt**.
- **Vocoder 11/12**: một bin spectrum vượt **tiêu chí tuyệt đối đã đăng ký**;
  không có bằng chứng mismatch nguồn, nhưng **chưa đủ để tuyên bố “sẵn sàng chuyển ONNX”**
  theo luật đã đặt trước.
- **Không chuyển production** trong vòng này; **không sửa runtime**.
- Nếu Agent 0 muốn chấp nhận tiêu chí **tương đối** cho spectrum (ví dụ sai tương đối
  ≤1% tại bin có magnitude >1), đó là quyết định tiêu chí ở vòng sau — **không đặt sau
  để hợp thức hóa kết quả này**. Bằng chứng hiện tại: bin lỗi ~0,5% tương đối;
  waveform/end-to-end đạt.

**Lệnh chuyển (chưa chạy):**
```bash
vifasttts synthesize-onnx \
  --acoustic artifacts/onnx/v1837/p2ema_acoustic.onnx \
  --vocoder artifacts/onnx/v1837/vocos_pretrained_vocoder.onnx \
  --text '<text>' --output out.wav
```

**Rollback:**
```bash
# PyTorch P2 + Vocos pretrained
vifasttts synthesize --config configs/small_codelung_v32.yaml \
  --checkpoint artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt \
  --text '<text>' --output out.wav

# ONNX cũ (chỉ lịch sử; vocoder cũ là E4-C, KHÔNG phải pretrained)
vifasttts synthesize-onnx \
  --acoustic artifacts/onnx/v32_adv_p2_2k.onnx \
  --vocoder artifacts/onnx/v32_adv_p2_2k_vocoder.onnx \
  --text '<text>' --output out.wav
```

---

## 9. Test / môi trường

### Clean worktree
- Worktree `266f5a9` tại `/private/tmp/opencode/v1840_clean`, symlink `data/codelung`,
  `cache`, `artifacts`, `runs`; `PYTHONPATH=<worktree>/src`.
- Relevant tests:
  `test_export_v2.py`, `test_v2_slot5.py`, `test_onnx_runtime.py`,
  `test_runtime_istft.py`, `test_vocos_generator.py`, `test_eval_text.py`
  → **36 passed**.
- Lần chạy đầu thiếu symlink `data/codelung` → 1 test custom reading fail; sau khi symlink
  đúng, 36 passed. Không sửa code vì lỗi này.

### Working tree
- `pytest -q` → **2.556 passed**; `ruff check .` sạch.
- Working tree có 2 file test user chưa stage; **không stage/revert**.
- Không dùng worktree sạch cho full suite; full suite chạy trên working tree.

### Artifact / commit
- Artifact parity: `artifacts/bench/v1840_onnx_parity/{inputs,result}.json` + WAV.
- Tool: `tools/v1840_onnx_parity.py`.
- Không đổi production/ONNX cũ/checkpoint gốc.
- Commit hồ sơ sau cùng ở HEAD; `V1840_agent2.md` là tài liệu bàn giao.

---

## 10. Chưa làm / giới hạn

- Không chấm BAK/UTMOS; không benchmark tốc độ/RAM.
- Không thử OOV text trên ONNX (khác `resolve_oov` tiềm ẩn).
- Không thử steady/sentence mode trên ONNX (ONNX không có sentence-gap; steady khác trim);
  đây là khác biệt runtime đã biết, không gộp vào sai số export.
- Không sửa tiêu chí spectrum sau kết quả; không chuyển production.
- Không kết luận artifact vocoder “sai” — chỉ kết luận **chưa đạt tiêu chí tuyệt đối đã đăng ký**.

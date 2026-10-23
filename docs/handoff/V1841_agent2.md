# V18.41 — Agent 2: đóng parity số học + frontend OOV trước chuyển ONNX

> **Trạng thái: DỪNG SAU V18.41, CHỜ ĐỌC.** Không training, không đổi production,
> chưa chuyển ONNX. Acoustic/end-to-end/parity số học giữ nguyên v18.40;
> vocoder 11/12 với **ngoại lệ spectrum được A0 chấp nhận có phạm vi**.
>
> **Commit gốc của vòng**: `1932553` · **Commit bàn giao**: `44ebe7c` · **Tệp đặc tả**:
> yêu cầu A0 → Agent 2 trong chat (không có `V1841_agent0.md`).

---

## 1. Khóa v18.40 / ngoại lệ A0

- Giữ acoustic **P2 EMA**, crop fixed vì loại padding.
- Giữ nguyên kết quả: acoustic/end-to-end đạt; vocoder 11/12.
- **Không sửa ngưỡng spectrum, không sửa runtime** để ép pass.
- A0 chấp nhận **ngoại lệ triển khai có phạm vi** cho outlier spectrum:
  provenance đúng; waveform và end-to-end đạt; không clipping.
- Đây là ngoại lệ **quyết định sau khi xem kết quả**, không phải vượt toàn bộ
  tiêu chí đăng ký trước; **không suy rộng** cho mọi input/backend.
- Chưa chuyển production trong vòng này.

Bằng chứng v18.40:
- acoustic 36/36; frontend 12/12; end-to-end 12/12;
- vocoder 11/12; outlier `2b65WyiFC9k_0030__s03` spectrum real max 0,02336 > 0,02,
  mean ~1e-05, waveform max 3,23e-04, end-to-end 9,16e-05;
- `/artifacts/bench/v1840_onnx_parity/result.json`.

---

## 2. Frontend OOV — phát hiện và sửa

### 2.1 Khác biệt source

- PyTorch production:
  - `Synthesizer._render` gọi `parse_text(..., resolve_oov=True)`
    (`src/vifasttts/infer.py`);
  - `iter_synthesize_sentence` cũng `parse_text(..., resolve_oov=True)`.
- ONNX cũ:
  - `OnnxAcousticModel.features_from_text` gọi `parse_text(text, dialect, options)`
    **không truyền `resolve_oov`** ⇒ mặc định `False`;
  - không có comment/doc nói đây là chủ đích; CLI/ONNX dùng cùng frontend.
- Kết luận: **bỏ sót ngoài chủ đích**, không phải thiết kế.
  - Custom reading vẫn được áp trước normalize ở cả hai (không phụ thuộc cờ).
  - Normalization/dialect/vocab giống nhau.
  - Khác biệt chỉ ở token **không parse được**: PyTorch respell OOV, ONNX cũ fallback.

### 2.2 Sửa

`src/vifasttts/runtime/onnx_engine.py`:

```python
def features_from_text(..., resolve_oov: bool = True):
    items = parse_text(..., resolve_oov=resolve_oov)
```

- Mặc định **True** = khớp đường PyTorch production.
- Không sửa trọng số, không export lại model, không thêm quy tắc đọc mới.
- Caller cũ có thể truyền `resolve_oov=False` nếu muốn hành vi legacy.

### 2.3 Bộ kiểm tra cố định

`artifacts/bench/v1841_frontend_oov/result.json`:

| Kind | Text | PyTorch items | Old ONNX | New ONNX |
|---|---|---:|---|---|
| custom_reading | `VBA.` | 3 (`vi be a`) | equal | equal |
| english_fallback | `OpenAI.` | 4 (`ô pần ây ai`) | equal | equal |
| oov_name_short | `John.` | 4 (`giây o hát en`) | **mismatch** | equal |
| oov_upper | `XYZ.` | 3 (`ích i dét`) | **mismatch** | equal |
| unresolved | `xqzzt.` | 1 `FALLBACK_REQUIRED` | equal | equal |
| number_punctuation | `Năm 2026, giá 15,5%.` | 15 | equal | equal |
| control_vi | `Xin chào các bạn.` | 4 | equal | equal |
| dialect_south | `1500.` | 4 (`một ngàn...`) | equal | equal |
| empty | `...` | 0 | — | **ValueError** đúng |

Tổng:
- valid cases: **8/8**;
- old ONNX equal: **6/8** (mismatch `John`, `XYZ`);
- new ONNX equal: **8/8**;
- smoke end-to-end 8/8: finite, samples >0, không crash;
- empty: `ValueError: Không có âm tiết nào để tổng hợp` (không crash im lặng).
- Unresolved `xqzzt` vẫn `FALLBACK_REQUIRED`, **không bị bỏ token**; cả hai đường giống nhau.

Smoke samples: `VBA` 30.464 · `OpenAI` 34.048 · `John` 34.816 · `XYZ` 26.624 ·
`xqzzt` 18.432 · number 90.624 · control 30.720 · dialect 43.008 (24 kHz).

---

## 3. Test

### 3.1 Test mới

`tests/test_onnx_frontend.py` — **9 passed**:
- 6 case frontend khớp PyTorch `resolve_oov=True`;
- dialect Nam;
- default phải resolve (và `resolve_oov=False` khác với `XYZ`);
- text không có âm tiết → `ValueError`.

### 3.2 Working tree

- `pytest -q` → **2.565 passed**.
- `ruff check .` sạch.
- Working tree có 2 file test user chưa stage; **không stage/revert**.
- Không training, không chấm MOS/ASR, không benchmark lớn.

### 3.3 Clean worktree

Sau commit bàn giao, chạy lại trên worktree sạch của đúng commit:
`test_onnx_frontend.py`, `test_onnx_runtime.py`, `test_export_v2.py`,
`test_v2_slot5.py`, `test_runtime_istft.py`, `test_vocos_generator.py`,
`test_eval_text.py`.
Kết quả tại `44ebe7c`: **45 passed**. Worktree dùng symlink
`data/codelung`, `cache`, `artifacts`, `runs` — dependency dữ liệu ngoài repo.

---

## 4. Kết luận chuyển ONNX

- **Frontend nhất quán** sau sửa: valid 8/8, smoke 8/8.
- **Sẵn sàng chuyển ONNX trong phạm vi chunk mode đã kiểm tra**, với:
  - acoustic P2 EMA; vocoder Vocos pretrained;
  - ngoại lệ spectrum `2b65WyiFC9k_0030__s03` được A0 chấp nhận có phạm vi;
  - không suy rộng cho mọi input/backend.
- **Chưa tự chuyển production** trong vòng này.

### Config ứng viên

`configs/onnx_candidate_v1841.json` — `status: candidate_not_active`:
- acoustic `artifacts/onnx/v1837/p2ema_acoustic.onnx`;
- vocoder `artifacts/onnx/v1837/vocos_pretrained_vocoder.onnx`;
- `mel_hash`/stats/checkpoint hash;
- runtime CPU, `VIFASTT_INTRA_THREADS=2`, chunk mode;
- ngoại lệ spectrum + ghi chú “không phải pass toàn bộ”.

### Lệnh chuyển (chưa chạy)

```bash
VIFASTT_INTRA_THREADS=2 vifasttts synthesize-onnx \
  --acoustic artifacts/onnx/v1837/p2ema_acoustic.onnx \
  --vocoder artifacts/onnx/v1837/vocos_pretrained_vocoder.onnx \
  --text '<text>' --output out.wav --provider cpu
```

### Rollback

```bash
# Về đúng production đã chọn: P2 EMA + PyTorch Vocos pretrained
VIFASTT_INTRA_THREADS=2 vifasttts synthesize \
  --config configs/small_codelung_v32.yaml \
  --checkpoint artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt \
  --text '<text>' --output out.wav
```

ONNX cũ (`v32_adv_p2_2k*.onnx`) chỉ là **artifact lịch sử**; vocoder cũ là E4-C,
**không gọi là rollback tương đương** production P2 + pretrained.

---

## 5. Giới hạn

- Chỉ kiểm **chunk mode**; chưa kiểm sentence/steady mode trên ONNX
  (ONNX không có sentence-gap; steady khác trim).
- OOV `John` được đánh vần theo `resolve_oov_token` hiện hành — đây là hành vi production,
  không phải quy tắc mới; chưa đánh giá chất lượng phát âm tên riêng.
- Không suy rộng ngoại lệ spectrum cho mọi input/backend/provider.
- Không chấm BAK/UTMOS/ASR; không benchmark tốc độ/RAM.
- Không sửa production, không chuyển ONNX.

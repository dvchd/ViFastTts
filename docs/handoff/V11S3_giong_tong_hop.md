# Dữ liệu đặc thù: thu giọng thật + sinh giọng bằng AI (đánh giá 14/10/2026)

> Trả lời yêu cầu: *"nếu thiếu dữ liệu đặc thù, dùng giọng mình kết hợp model AI tạo giọng
> chất lượng cao để tạo các đoạn đọc ít dữ liệu, đặc biệt, trường hợp biên, dữ liệu chưa bao phủ"*.

## 1. Dữ liệu thiếu gì (đo được, không phải đoán)

Phân tích **222.737 âm tiết** của tập train (`train_clean_v2_dur_seg_slot4_fixtail`):

| Nhóm | Tỉ lệ | Ô hiếm nhất |
|---|---|---|
| NGÃ | **4,5%** (10.113) | `M:NGÃ` = **2 lần** · `W:NGÃ` 56 · `NH:NGÃ` 84 |
| NẶNG | 9,7% | `M:NẶNG` 328 · `M:HỎI` 336 · `NH:NẶNG` 417 |
| coda tắc p/t/c/ch | 1,1–5,0% | SẮC-tắc là ô tệ nhất (§42.1: mel +13,8%) |
| câu HỎI cuối câu | ít | model **ngã giọng** cuối câu hỏi (§42.3: thật +1,6 vs model −238,5 c/s) |

⇒ Đúng các nhóm model yếu nhất **trùng** các nhóm dữ liệu mỏng nhất. Đây là gốc của "rè ngã".

## 2. Ba nguồn dữ liệu bù, xếp theo độ tin cậy

| Nguồn | Chất lượng cho TRAIN | Chi phí | Ghi chú |
|---|---|---|---|
| **① Bạn thu 60 câu** (`resources/cau_thu_am_v1.md`) | **cao nhất** — giọng thật, phòng thật, không artefact | ~40 phút đọc | Đây là "chuẩn vàng"; 60 câu ≈ 8–10 phút audio, đủ dày ô hiếm |
| ② Sinh bằng AI clone từ giọng bạn | trung bình — **có artefact của bộ sinh** | 0đ (self-host) | Chỉ nên dùng sau **ablation đo được**; lọc nghiêm |
| ③ Thu thêm nguồn ngoài (VLSP/VAIS…) | trung bình | giấy phép phức tạp | Đã khảo sát 14/10: không có bộ "mở + nữ/nam Bắc + ≥5 h" |

## 3. Model sinh giọng (khảo sát 14/10/2026)

| Model | Clone | Chạy trên Mac M6? | Giấy phép | Ghi chú |
|---|---|---|---|---|
| **VieNeu-TTS-v2** | 3–5 s audio mẫu | ✅ MPS + CPU (GGUF/ONNX) | mở (repo KinX-ai) | 10.000+ h Anh–Việt, 24 kHz — **hợp máy ta nhất** |
| F5-TTS-ViVoice (`hynt/…`) | ~10 s | ⚠️ cần GPU NVIDIA (MPS rất chậm) | **CC-BY-NC-SA-4.0** (phi thương mại) | 1.000 h Việt, chất lượng cao |
| XTTS-v2 (Coqui) | ~10 s | ✅ nhẹ | cộng đồng duy trì (Coqui đã đóng) | chất lượng "khá", hợp demo |
| vi-F5-TTS (`danhtran2mind`) | ~10 s | ⚠️ | MIT | 100 h, tác giả tự nêu hạn chế |

**Điểm chung phải nhớ**: clone chỉ tốt nếu **audio mẫu sạch** (không nhạc nền/vang) và
**transcript mẫu khớp chính xác** — lệch vài chữ là giọng méo.

## 4. Rủi ro khi lấy audio AI làm dữ liệu TRAIN (bài học đã trả giá)

- §41 (GTA Vocos): train trên audio/mel *sinh ra* ⇒ **học luôn artefact**: HNR −1,7 dB, RMS +16%,
  độ gồ ghề **mượt hơn** (ngược mục tiêu) ⇒ **trượt cổng, xếp lại**.
- §39: phần "ồn/rè" nằm ở **mel dự đoán**, không ở vocoder ⇒ thêm dữ liệu sinh bởi model khác
  sẽ **đổi phân bố** mà ta không kiểm soát được.
- Nguyên tắc rút ra: **audio AI chỉ vào train khi có ablation thắng cổng** (CER + F0 theo thanh +
  creak + độ nét), và **tỉ lệ nhỏ** (≤10–15% giờ), có lọc tự động (ASR vòng tròn + kiểm HNR).

## 5. Đề xuất thứ tự (làm ngay được)

1. **Bạn thu 60 câu** (file `resources/cau_thu_am_v1.md`) — mình chuẩn hoá 24 kHz, cắt câu,
   align, rồi **thêm vào train với trọng số cao** cho đúng ô hiếm. Đây là đòn chắc ăn nhất.
2. Song song: mình dựng thử **VieNeu-TTS-v2** (MPS) clone từ chính 60 câu đó → sinh thêm
   ~200 câu biên → **đo** (ASR vòng tròn + HNR + creak) → chỉ giữ câu đạt; dùng làm **tập test
   căng** trước, train sau (kèm ablation).
3. Nếu cần giọng khác/nữ Bắc: vẫn phải **tự thu 10–20 h** (khảo sát 14/10 không có bộ mở phù hợp).

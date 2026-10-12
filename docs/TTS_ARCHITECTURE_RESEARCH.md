# Khảo sát kiến trúc TTS cho ViFastTts

> **Trạng thái triển khai (11/10/2026):** các đề xuất trong tài liệu này đã được hiện thực
> một phần: ✅ Segment Expander (§9.4, `frontend/segments.py`), ✅ phonation extraction +
> conditioning (§9.7, `data/phonation.py`, `model.use_phonation`), ✅ vocoder training tham
> chiếu (§9.9, `train-vocoder`), ✅ quantization FP16/INT8 động (§9.12), ✅ ONNX acoustic
> với độ dài động. ⬜ Còn lại: aligner thật (MFA/forward-sum), discriminator/spectral loss
> production, ONNX vocoder, calibration INT8 đầy đủ. Luồng as-built: `docs/MODEL_FLOW.md`;
> trạng thái tổng: `docs/PROJECT_STATUS.md`.

**Mục đích:** rà soát các họ kiến trúc TTS phổ biến (AR, NAR, end-to-end VAE/GAN, flow matching, diffusion, vocoder) và kết luận cho bài toán riêng của ViFastTts: **một giọng, một phương ngữ cho mỗi checkpoint, đầu vào là cấu trúc âm tiết `ONSET+MEDIAL+NUCLEUS+CODA+TONE`, không speaker/dialect/style id, non-autoregressive, deterministic, 24 kHz, ONNX là định dạng triển khai, có baseline iSTFTNet**.

**Cách khảo sát:** mỗi kết luận dưới đây dựa trên bài báo/repo gốc. Các liên kết đã được kiểm tra trực tiếp bằng công cụ tải trang (webfetch) được đánh dấu ✅ trong mục 11; các liên kết chỉ xác minh được qua kết quả tìm kiếm (trích đúng tiêu đề/abstract) được đánh dấu 🔎. Không có benchmark nào trong tài liệu này được tự đặt ra; các con số đều là con số do chính bài báo/repo công bố và được ghi rõ nguồn.

---

## 1. Tiêu chí cho bài toán (một giọng, deterministic, ONNX, tốc độ, dữ liệu vừa phải)

### 1.1 Ràng buộc cứng (hard constraints)

| Mã | Ràng buộc | Hệ quả kiến trúc |
| --- | --- | --- |
| C1 | Mỗi checkpoint = một người nói + một phương ngữ, không `speaker_id`/`dialect_id`/`style_id`/audio tham chiếu | Không cần speaker encoder, style encoder, embedding điều kiện; chất giọng nằm trong trọng số. Loại bỏ mọi kiến trúc zero-shot/voice-cloning ở runtime. |
| C2 | Đầu vào là cấu trúc âm tiết (5 trường + boundary/punctuation), không phải chuỗi ký tự thô | Frontend giải bằng luật; mạng chỉ học ánh xạ ngữ cảnh âm tiết → âm thanh. Encoder cấp âm tiết + expander cấp thành phần (onset/medial/nucleus/coda). |
| C3 | Non-autoregressive | Loại bỏ mọi decoder sinh tuần tự theo frame (Tacotron 2, Transformer TTS, codec LM). Không có attention tự hồi quy ⇒ giảm mạnh nguy cơ lặp/nuốt/dừng sớm. |
| C4 | Deterministic: cùng input + cùng `speed` ⇒ cùng output (byte-identical trên cùng phần cứng/numerics) | Không dùng sampling, không dropout khi suy luận, không stochastic duration predictor, không VAE latent sampling, không diffusion/flow nhiều bước có nhiễu. |
| C5 | ONNX là định dạng triển khai chính (CPU, ARM, Apple Silicon, NVIDIA) | Ưu tiên op phổ thông (`Conv`, `ConvTranspose`, `MatMul`, `LayerNorm`, `GELU`/`SiLU`, `Softmax`). Tránh op hiếm/custom, tránh op phụ thuộc trạng thái (KV-cache, `Loop`, `Scan`), cảnh giác với `STFT`/`ISTFT`. |
| C6 | Suy luận nhanh | Số lần forward cố định và nhỏ (1 lần cho acoustic + 1 lần cho vocoder). Tránh NFE (number of function evaluations) lớn. |
| C7 | Dữ liệu vừa phải (5–50 giờ, một người đọc) | Kiến trúc không đòi hỏi pretraining quy mô lớn (LLM/codec LM/SLM discriminator). Mô hình ~30–80M tham số là hợp lý. |
| C8 | 24 kHz | Cấu hình `n_fft=1024`, `hop=256`, `n_mels=100` (đúng như `configs/base_24k.yaml`). |
| C9 | Có baseline iSTFTNet | Vocoder mục tiêu iSTFTNet; HiFi-GAN giữ làm baseline đối chứng. |
| C10 | Đọc đúng > không lặp/nuốt > thanh điệu > chất giọng > tự nhiên > biểu cảm | Kiến trúc phải có cơ chế ràng buộc duration và tone rõ ràng, không phó mặc cho attention. |

### 1.2 Định nghĩa tốc độ dùng trong tài liệu

`RTF = thời_gian_sinh / thời_lượng_audio` (RTF < 1 là nhanh hơn thời gian thực). Cần đo tối thiểu: cold start, warm start, batch 1/4/8, độ dài 3 s / 10 s / 30 s, RAM/VRAM (mục 28 của `DESIGN_V2.md`).

### 1.3 Ngân sách tốc độ suy ra cho từng khối

Với `hop=256` ở 24 kHz, mỗi giây audio ≈ 93,75 frame mel. Muốn RTF ≈ 0,05 trên CPU desktop (mục tiêu trong `DESIGN_V2.md`), tổng chi phí 1 giây audio phải ≤ 50 ms. Ước lượng bậc độ lớn: encoder Conformer nhẹ cấp âm tiết ~ vài ms, decoder mel ~ vài ms chục, vocoder chiếm phần lớn ngân sách. Điều này giải thích trực tiếp vì sao các họ nhiều bước (diffusion/flow) và các họ codec LM bị loại.

---

## 2. Họ mô hình hồi quy tự động (AR)

### 2.1 Tacotron 2 và Transformer TTS

- **Tacotron 2** (Shen et al., 2017): seq2seq RNN + location-sensitive attention dự đoán mel, rồi vocoder WaveNet. Chất lượng vượt trội thời điểm đó nhưng suy luận tuần tự theo frame, dễ lặp/nuốt, không ổn định với câu dài (vấn đề này được chính FastSpeech 1 (2019) nêu ra khi mô tả động cơ thiết kế: "some words are skipped or repeated").
- **Transformer TTS** (Li et al., 2019): thay RNN bằng self-attention; huấn luyện nhanh hơn Tacotron 2 khoảng 4,25×, MOS 4,39 so với 4,44 của bản ghi thật; nhưng **vẫn là AR decoder**.

Với ViFastTts, hai mô hình này vi phạm C3 (và gián tiếp C4/C6: độ dài output do mô hình tự quyết nên không thể đảm bảo output khớp input, không thể ràng buộc duration theo segment). ONNX export được nhưng cần vòng lặp/decoder cache rời rạc, thường phải chia graph và driver bên ngoài can thiệp — đi ngược tinh thần triển khai một graph.

### 2.2 Codec language models (VALL-E, VALL-E 2, VoiceCraft, CosyVoice 1/2, XTTS-v2, Fish Speech)

Đây là xu hướng 2023–2025 của "foundation TTS": rời rạc hóa audio thành token rồi dùng LM sinh token.

| Mô hình | Cơ chế | Điểm mạnh báo cáo | Vì sao không phù hợp ViFastTts |
| --- | --- | --- | --- |
| **VALL-E** (2023) | Codec (EnCodec) + 2 tầng LM (AR theo thời gian, NAR cho codebook còn lại); pretrain 60.000 giờ | Zero-shot với 3 s audio mẫu; vượt SOTA zero-shot về tự nhiên/giống người nói | Cần dữ liệu + compute cực lớn; sampling stochastic; phụ thuộc audio tham chiếu (vi phạm C1) |
| **VALL-E 2** (2024) | VALL-E + Repetition Aware Sampling + Grouped Code Modeling | Lần đầu đạt "human parity" zero-shot; ổn định khi câu nhiều lặp từ; giảm STT (64→16) và tăng tốc inference | Vẫn AR + sampling; dữ liệu lớn; không deterministic |
| **VoiceCraft** (2024) | Token infilling Transformer decoder, delayed stacking | SOTA speech editing + zero-shot TTS trên dữ liệu "in the wild" | Infilling = sinh tuần tự có điều kiện; cần prompt audio |
| **CosyVoice 1/2** (2024) | Token ngữ nghĩa giám sát + LM + flow matching; CosyVoice 2 dùng FSQ token, backbone LLM có sẵn, flow matching nhân quả theo chunk | CosyVoice 2: streaming, độ trễ thấp, chất lượng "gần như không mất mát" khi streaming | LM sinh token AR + flow matching nhiều bước ⇒ không deterministic, RTF cao, cần LLM; ngoài phạm vi một giọng |
| **XTTS-v2** (2024) | Kế thừa Tortoise (AR decoder) + nhiều cải tiến; huấn luyện 16 ngôn ngữ (card mô hình ghi hỗ trợ 17 ngôn ngữ) | Zero-shot đa ngôn ngữ, cloning ~6 s audio | AR; cần reference audio; không phù hợp ONNX nhẹ, không deterministic |
| **Fish Speech** (2024) | Dual-AR: "slow AR" theo thời gian + "fast AR" theo độ sâu codebook; GFSQ; bỏ G2P (dùng LLM trích đặc trưng ngôn ngữ) | Ổn định sinh codebook, đa ngôn ngữ, cloning | Dual-AR = 2 tầng tuần tự; cần LLM backbone; không deterministic |

Nhận xét chung nhóm codec LM:

1. **Chất lượng đến từ dữ liệu và tham số**, không từ cơ chế: VALL-E pretrain 60k giờ; VieNeu-TTS v3 Turbo (bản mở của VieNeu) báo cáo 10.000+ giờ huấn luyện song ngữ. Một người đọc 5–50 giờ không thể khai thác lợi thế này.
2. **Sinh token là rời rạc + stochastic**: mọi kỹ thuật sampling (nucleus, repetition-aware) đều phá vỡ C4. Có thể dùng greedy để "deterministic hóa", nhưng chất lượng greedy của codec LM thấp hơn sampling và vẫn dễ lặp.
3. **ONNX có thể export** (VieNeu v3 Turbo chạy torch-free trên ONNX Runtime CPU, RTF ≈ 0,5–0,62 fp32, ≈ 0,35 int8 theo README repo) nhưng **RTF CPU kém xa mục tiêu ViFastTts** và độ trễ streaming chỉ đạt ~115 ms nhờ GPU + CUDA graph.
4. Không kiểm soát được cấu trúc âm tiết: đây là điểm mấu chốt — ViFastTts muốn mỗi `ONSET+MEDIAL+NUCLEUS+CODA+TONE` được mô hình hóa tường minh, còn codec LM coi phát âm là hệ quả gián tiếp của LM.

**Kết luận mục 2:** toàn bộ họ AR bị loại khỏi đường triển khai ViFastTts. Chỉ nên dùng chúng làm mốc tham chiếu chất lượng (và có thể dùng để tạo dữ liệu synthetic nếu cần).

---

## 3. Họ mô hình phi hồi quy (NAR / FastSpeech)

Đây là họ gần nhất với mục tiêu ViFastTts.

### 3.1 FastSpeech 1 (Ren et al., NeurIPS 2019)

- Cấu trúc: Feed-Forward Transformer (FFT) + **length regulator** mở rộng chuỗi phoneme theo duration; duration lấy từ **teacher AR** (attention alignment) và distill mel từ teacher.
- Kết quả báo cáo: nhanh hơn AR Transformer TTS **270×** ở sinh mel và **38×** end-to-end; gần như loại bỏ lặp/nuốt; điều khiển tốc độ trơn tru.
- Hạn chế: pipeline teacher–student phức tạp; duration từ teacher kém chính xác; mel distill mất thông tin ⇒ chất lượng bị chặn trên.

### 3.2 FastSpeech 2 / 2s (Ren et al., ICLR 2021) — nền tảng quan trọng nhất

- Bỏ distillation: huấn luyện trực tiếp với **mel ground-truth**.
- **Variance adaptor**: duration + pitch + energy (và có thể thêm các biến thể khác), lấy từ waveform (duration từ forced aligner như MFA) làm điều kiện khi train; khi suy luận dùng giá trị do predictor dự đoán.
- Pitch biểu diễn bằng spectrogram (continuous wavelet transform) để dễ dự đoán contour.
- **FastSpeech 2s**: bỏ mel trung gian, sinh thẳng waveform (đầu ra "end-to-end").
- Kết quả báo cáo: giảm **3,12×** tổng thời gian huấn luyện so với FastSpeech; tăng tốc sinh waveform **47×** (FS2) và **51×** (FS2s) so với Transformer TTS; chất lượng có thể vượt AR.

Vì sao quan trọng với ViFastTts:

- Mọi thứ đều feed-forward, độ dài output do duration quyết định ⇒ **deterministic** và **ràng buộc được** (C3, C4, C10).
- Variance adaptor chính là cơ chế để đưa tri thức thanh điệu/ngữ điệu vào mà không cần style embedding ⇒ khớp C1 (mỗi checkpoint học "giọng trung bình" của người đọc).
- ONNX thân thiện: encoder/decoder là chồng `Linear/Conv/Attention`, length regulator chỉ là `repeat_interleave`/`Gather` hoặc `MatMul` one-hot — tất cả đều là op cơ bản.

### 3.3 FastPitch (Łańcucki, 2020)

FastSpeech + điều kiện hóa theo contour F0 (dự đoán cả average pitch và contour). Đây là tiền lệ trực tiếp cho "tone-conditioned F0" của ViFastTts: pitch được tách khỏi nội dung, do đó có thể dự đoán/điều khiển riêng. Fully parallel, deterministic.

### 3.4 Parallel Tacotron (Google, ICASSP 2021) và DelightfulTTS (Microsoft, Blizzard 2021)

- **Parallel Tacotron**: NAR + **residual encoder VAE** để nới lỏng ánh xạ một–nhiều, **iterative spectrogram loss** (tinh chỉnh lặp đầu ra) và **lightweight convolution** bắt ngữ cảnh cục bộ; kết quả ngang chất lượng baseline AR với tốc độ tốt hơn đáng kể. Hạn chế: VAE ở acoustic model nghĩa là có latent (thường cần mean ở suy luận — nhưng đã có tiền lệ dùng deterministic mean); iterative loss cần thêm bước.
- **DelightfulTTS**: hệ thống NAR cho Blizzard 2021 (Microsoft), dùng attention-based duration + mô hình hóa prosody phân cấp, chất lượng cao nhưng không phải thiết kế cho ONNX nhẹ và không có ablation "single speaker 24 kHz edge".

Với ViFastTts: **mượn ý tưởng** (residual prosody, iterative/aux loss) nhưng không chọn VAE latent làm đường chính, vì chúng ta ưu tiên deterministic (C4) và đơn giản để quantize (C5).

### 3.5 Vì sao họ NAR hợp ONNX nhất

- Các phép biến đổi đều tĩnh theo trục thời gian sau khi duration được tính bằng phép toán tensor (không có vòng lặp rời rạc).
- Số lần forward cố định = 2 (acoustic + vocoder) hoặc 1 nếu gộp.
- Không có phụ thuộc giữa các bước decode ⇒ lượng tử hóa INT8 cho toàn mạng chỉ còn vấn đề số học, không có sai số tích lũy từ mẫu.
- Hỗ trợ trực tiếp "speed scaling" bằng cách chia duration trước khi expand — một phép chia scalar, có thể đưa vào graph hoặc xử lý ở runtime.

---

## 4. Họ end-to-end VITS/GAN/VAE

### 4.1 VITS (Kim et al., ICML 2021)

- Một mạng duy nhất text → waveform: **conditional VAE** + normalizing flow + **adversarial training**; dùng **Monotonic Alignment Search (MAS)** để tự học alignment (không cần aligner ngoài).
- **Stochastic duration predictor** (flow trên duration) để tái hiện nhịp điệu "người thật"; kết quả trên LJSpeech: MOS tương đương bản ghi thật, vượt các hệ 2 tầng.
- Điểm mấu chốt cho ViFastTts: **stochastic duration + latent sampling phá vỡ C4** (tài liệu HuggingFace Transformers mô tả đúng: "Due to the stochastic nature of the duration predictor, the model is non-deterministic, and thus requires a fixed seed"). Có thể chuyển sang "deterministic duration + lấy mean của prior", nhưng khi đó ta đang tự xây một biến thể — lúc này lợi ích "end-to-end" giảm đi trong khi độ phức tạp (flow, discriminator, posterior encoder) vẫn còn nguyên.
- Tuy nhiên VITS có ONNX chạy thật: **Piper** (Rhasspy) là hệ VITS huấn luyện single-speaker, export sang ONNX, chạy trên Raspberry Pi. Vậy "VITS + ONNX" khả thi — nhưng Piper phải vô hiệu hóa phần stochastic và chấp nhận ràng buộc chất lượng.

### 4.2 VITS2 (Interspeech 2023)

Cải thiện VITS: (i) thay khối trong flow bằng Transformer, (ii) thêm discriminator + **adversarial learning cho duration predictor**, (iii) giảm phụ thuộc chuyển phoneme. Kết quả: tự nhiên hơn, hiệu quả huấn luyện/suy luận tốt hơn. Đây là bằng chứng rằng **duration predictor nên được học có giám sát/đối kháng ở cấp phân bố**, và là gợi ý cho ViFastTts (thêm adversarial loss phụ cho duration/decoder nếu cần).

### 4.3 JETS (Interspeech 2022)

FastSpeech 2 + HiFi-GAN **huấn luyện chung** end-to-end (acoustic decoder → mel → vocoder trong cùng vòng lặp, có discriminator). Vì là FastSpeech 2, JETS **deterministic và ONNX-friendly**, đồng thời tránh lệch pha train/inference giữa mel dự đoán và mel thật. Đây là **kiến trúc tham chiếu số 1 cho giai đoạn joint fine-tuning của ViFastTts** (tương ứng "Giai đoạn 5" trong `TRAINING.md`).

### 4.4 StyleTTS 2 (NeurIPS 2023) và Kokoro-82M

- **StyleTTS 2**: style là biến latent ngẫu nhiên được lấy mẫu bằng **style diffusion** (không cần audio tham chiếu), huấn luyện đối kháng với **SLM (WavLM)** làm discriminator; duration được mô hình hóa vi phân end-to-end. Kết quả: vượt chất lượng bản ghi trên LJSpeech single-speaker, ngang bằng trên VCTK.
- Điểm quan trọng: **decoder của StyleTTS 2 chính là iSTFTNet**. Và **Kokoro-82M** là bản triển khai thực tế: card mô hình ghi rõ kiến trúc = **StyleTTS 2 + ISTFTNet** ("Decoder only: no diffusion, no encoder release"), 82M tham số, 24 kHz, Apache-2.0, huấn luyện trên "vài trăm giờ" dữ liệu sạch với chi phí ~1.000 giờ A100 (~1.000 USD). Kokoro chạy ONNX phổ biến trong cộng đồng (kokoro-onnx, onnx-community/Kokoro-82M-v1.0-ONNX, sherpa-onnx) và RTF CPU xấp xỉ/nhanh hơn thời gian thực.

Với ViFastTts, Kokoro là **bằng chứng mạnh nhất** cho luận điểm: một mô hình NAR nhỏ + decoder kiểu iSTFTNet + deterministic + ONNX là đủ để đạt chất lượng "tiệm cận mô hình lớn" trong điều kiện một giọng/dữ liệu vừa phải. Phần style diffusion của StyleTTS 2 bị loại khỏi thiết kế của ta (C1/C4), nhưng phần decoder/frequency-domain vocoding thì giữ.

### 4.5 NaturalSpeech 2 / 3 — dùng làm tham chiếu, không dùng làm thiết kế

- **NaturalSpeech 2** (2023): neural codec (RVQ) + **latent diffusion** + speech prompting; chất lượng/tự nhiên cao, zero-shot cả singing.
- **NaturalSpeech 3** (2024): **FACodec** (codec phân rã thuộc tính thành content/prosody/detail) + **factorized diffusion** cho từng không gian thuộc tính; đạt chất lượng người thật zero-shot trên dữ liệu đa người nói. Lưu ý kỹ thuật: FACodec hoạt động ở **16 kHz** (hop 200 mẫu; 6 codebook) — theo Q&A của repo `lifeiteng/naturalspeech3_facodec`.

Hai hệ này khẳng định 2 ý tưởng mà ViFastTts đã có sẵn trong thiết kế: (1) **tách nhân tố phát âm / prosody / chi tiết** và (2) **điều kiện hóa tường minh thay vì để mạng tự đoán**. Nhưng bản thân hệ (codec + diffusion, dữ liệu hàng chục nghìn giờ) vi phạm C6/C7/C4/C5 nên chỉ dùng làm tham chiếu thiết kế.

---

## 5. Họ flow matching và diffusion

### 5.1 Diffusion cho TTS

| Mô hình | Cơ chế | Ghi chú cho ViFastTts |
| --- | --- | --- |
| **Grad-TTS** (2021) | Score-based decoder sinh mel từ nhiễu, dùng MAS để align; SDE cho phép **điều chỉnh đánh đổi chất lượng/tốc độ** bằng số bước | Chất lượng tốt nhưng NFE lớn; stochastic; không deterministic nếu không cố định nhiễu |
| **Diff-TTS** (2021) | Diffusion trên mel, non-AR, dùng accelerated sampling | Báo cáo 28× nhanh hơn thời gian thực trên 2080Ti — nhưng trên CPU edge vẫn là nhiều bước |
| **E3 TTS** (ASRU 2023) | Diffusion **trực tiếp trên waveform**, không có mel/alignment trung gian; hỗ trợ edit zero-shot | Bỏ mel trung gian ⇒ mất khả năng debug/kiểm soát và giám sát cấp frame; ngoài triết lý "mel dễ quan sát" của ViFastTts |

### 5.2 Flow matching

| Mô hình | Cơ chế | Ghi chú |
| --- | --- | --- |
| **Matcha-TTS** (ICASSP 2024) | Encoder-decoder dùng **OT-CFM**, decoder U-Net 1D + Transformer (không dùng 2D conv như Grad-TTS); dùng MAS + prior loss, học nói mà **không cần aligner ngoài** | Nhỏ gọn, "memory footprint nhỏ nhất" trong so sánh của bài báo, tốc độ ngang mô hình nhanh nhất trên câu dài, MOS cao nhất trong listening test của họ. Export ONNX tồn tại trong thực tế (sherpa-onnx hỗ trợ Matcha-TTS), nhưng vẫn cần N bước Euler ⇒ RTF phụ thuộc N |
| **E2 TTS** (SLT 2024) | Fully NAR, text điền vào chuỗi như iTTS "filler", flow matching trên mel; **không cần duration predictor, không cần alignment**; human-level naturalness + SOTA speaker similarity | Cần audio prompt (zero-shot) ⇒ vi phạm C1; không kiểm soát alignment tường minh |
| **F5-TTS** (2024/2025) | NAR, flow matching với **Diffusion Transformer (DiT)**; không duration modeling, không phoneme alignment; hỗ trợ zero-shot | Chất lượng cao nhưng là mô hình zero-shot cỡ lớn, huấn luyện cần GPU; ở ViFastTts chỉ nên coi là mốc tham chiếu hoặc công cụ tạo dữ liệu |

### 5.3 Đánh giá khả năng "deterministic + ONNX" của flow/diffusion

- Về mặt toán học, ODE flow matching **có thể chạy deterministic**: cố định nhiễu đầu vào (seed), cố định số bước và solver Euler ⇒ output lặp lại được. Nhưng: (i) đó là "deterministic theo cấu hình", không phải bất biến theo thiết kế; (ii) chi phí = N × (forward của vector field), thường N = 8–32 bước; (iii) các bước là tuần tự nên không song song hóa được.
- Bằng chứng thực tế gần ViFastTts nhất: **VieNeu-TTS v3 Nano** là mô hình flow matching 48M tham số, export ONNX, chạy CPU torch-free: RTF ≈ 0,22 ở 16 bước và ≈ 0,11 ở 8 bước (24 kHz), chất lượng thấp hơn bản "Turbo" và bị đánh giá yếu ở phần tiếng Anh. Đây là minh chứng: flow matching **có thể** chạy edge ONNX, nhưng **RTF CPU vẫn thua một feed-forward 1 bước** ở cùng ngân sách chất lượng.
- Đối chiếu: các mô hình NAR feed-forward (FastSpeech2/JETS/VITS-deterministic/Kokoro) đạt RTF thấp hơn với 1 forward duy nhất.

### 5.4 Xu hướng 2023–2026 (foundation TTS) và kết luận riêng cho ViFastTts

Xu hướng quan sát được qua các nguồn đã xác minh:

1. **Chất lượng đỉnh thuộc về codec LM + flow matching quy mô lớn**: VALL-E 2 (human parity zero-shot), CosyVoice 2 (streaming, FSQ + flow matching nhân quả theo chunk), Fish Speech (Dual-AR), NaturalSpeech 3 (FACodec + factorized diffusion), F5-TTS/E2-TTS (flow matching NAR). Các khảo sát tổng quan gần đây (SpeechLM survey 2024/2025; khảo sát controllable TTS 2024) đều ghi nhận dịch chuyển sang mô hình nền + mô hình hóa prosody tường minh.
2. **Phân khúc on-device/CPU lại nghiêng về NAR feed-forward nhỏ**: Piper (VITS→ONNX), Kokoro-82M (StyleTTS2+iSTFTNet, Apache-2.0), Matcha-TTS (ONNX trong sherpa-onnx), VieNeu Nano (flow matching nhỏ), V-TTS (VITS tối giản 57–75M) — tất cả đều ưu tiên số forward nhỏ, deterministic khi cần và kích thước < ~100M.
3. **Mô hình hóa thuộc tính tách rời (factorization)** trở thành chuẩn mực: FastSpeech 2 (duration/pitch/energy), NaturalSpeech 3 (content/prosody/detail), FACodec. ViFastTts đã đi theo hướng này với 5 trường âm tiết + duration/F0/energy/phonation.
4. **Alignment tự học** dần thay aligner ngoài (Glow-TTS MAS → RAD-TTS forward-sum → Matcha prior loss → Super MAS 2024), nhưng aligner ngoài (MFA) vẫn là lựa chọn thực dụng khi cần ổn định và tái lập.

**Kết luận riêng cho ViFastTts (một giọng, deterministic, ONNX, CPU/edge):** các mô hình "foundation" giải bài toán *zero-shot, nhiều giọng, nhiều ngôn ngữ* — phần thách thức của chúng (mô hình hóa người nói, ngôn ngữ, style) với ViFastTts **không tồn tại** (đã bị loại bởi C1). Vì vậy lợi ích biên của mô hình lớn/AR/flow matching bằng 0, trong khi chi phí (RTF, tính ngẫu nhiên, kích thước, độ phức tạp export) rất lớn. Hướng đúng là **NAR feed-forward + variance tường minh + vocoder frequency-domain**, cập nhật các ý tưởng mới (factorization, alignment tự học, frequency-domain vocoder) vào khuôn khổ đó.

---

## 6. Vocoder (HiFi-GAN, iSTFTNet, Vocos, BigVGAN)

### 6.1 HiFi-GAN (NeurIPS 2020)

- Kiến trúc: generator chồng transposed conv + multi-receptive field fusion; discriminator đa chu kỳ (MPD) + đa tỉ lệ (MSD).
- Kết quả báo cáo: 22,05 kHz, **167,9× nhanh hơn thời gian thực trên 1 V100**; bản "small footprint" (V3, ~0,92M tham số) đạt **13,4× nhanh hơn thời gian thực trên CPU** với chất lượng tương đương một mô hình AR.
- HiFi-GAN là vocoder **được export ONNX nhiều nhất** (mọi voice Piper đều dùng decoder VITS kiểu HiFi-GAN), độ ổn định quantization tốt vì toàn bộ là `Conv`/`ConvTranspose`/`LeakyReLU`/`Tanh`/`SiLU`.
- Vai trò trong ViFastTts: **baseline bắt buộc** để so sánh với iSTFTNet (đúng như `DESIGN_V2.md` mục 23.1).

### 6.2 iSTFTNet (ICASSP 2022) và iSTFTNet2 (Interspeech 2023)

- **iSTFTNet**: thay một phần lớp output của vocoder HiFi-GAN bằng **iSTFT**, sau khi đã giảm chiều tần số bằng các lớp upsampling. Mục tiêu: tránh "ước lượng dư thừa" phổ biến đổi phức như HiFi-GAN phải làm (recovery magnitude + phase + frequency-to-time). Kết quả báo cáo: **nhanh hơn và nhẹ hơn** các biến thể HiFi-GAN với **chất lượng "hợp lý"** (tức là có thể thấp hơn một chút ở chi tiết tần số cao — cần nghe kiểm chứng trên 24 kHz tiếng Việt).
- **iSTFTNet2**: dùng **1D–2D CNN** (1D theo thời gian, 2D cho cấu trúc phổ), upsampling tần số trong không gian ít tần số ⇒ nhanh hơn + nhẹ hơn nữa, chất lượng tương đương iSTFTNet.
- Hệ quả ONNX (rất quan trọng): iSTFTNet đẩy phép **iSTFT vào trong graph**. ONNX có op `STFT` từ opset 17 nhưng **không có op ISTFT** (đề xuất ISTFT vẫn đang mở: onnx/onnx#4777 — "Unfortunately ONNX misses the inverse operation, called ISTFT"). PyTorch cũng có lịch sử không export được `aten::stft` lên opset 17 với exporter cũ (issue #81075; lỗi tương tự trong noisereduce#93), và có báo cáo sai lệch kết quả giữa PyTorch/ONNX cho mô hình có cả stft+istft (microsoft/onnxruntime#23219).

  ⇒ Với ViFastTts, iSTFT **không được** để phụ thuộc vào op `ISTFT` (không tồn tại) hay op `STFT` (khả năng hỗ trợ khác nhau giữa các execution provider). Cách chắc chắn: **biểu diễn iSTFT bằng các op cơ bản** (chi tiết ở mục 9.12).

### 6.3 Vocos (2023)

- Vocos **sinh trực tiếp hệ số Fourier (spectral coefficients)** cho từng frame thay vì sinh waveform trong miền thời gian, rồi mới đưa qua iSTFT; backbone là ConvNeXt.
- Kết quả báo cáo: ngang SOTA về chất lượng nhưng **nhanh hơn một bậc độ lớn (~10×)** so với vocoder time-domain; đây là lựa chọn "gần iSTFTNet" nhưng triệt để hơn (không còn upsampling miền thời gian nào).
- Về ONNX: cùng vấn đề iSTFT như trên nhưng **đã có giải pháp cộng đồng**: repo `mush42/istft-onnx` tạo một graph ONNX thực hiện ISTFT (thiết kế để nối sau Vocos); README ghi "chậm hơn ít nhất 4× so với torch istft, nhưng chi phí iSTFT là không đáng kể". Đây là phương án dự phòng đã được kiểm chứng bởi cộng đồng.
- Rủi ro quantization: đầu ra là số thực phức (mag/phase hoặc real/imag), nhạy cảm hơn với INT8 ở lớp cuối.

### 6.4 BigVGAN (ICLR 2023)

- Vocoder "universal" với **tối đa 112M tham số**, dùng activation tuần hoàn (snake) + anti-aliased representation; SOTA cho zero-shot/OOD (nhiều người nói, nhiều ngôn ngữ, cả nhạc).
- Với ViFastTts (một người nói, 24 kHz, edge): 112M tham số là quá lớn so với mục tiêu 40–60M cho **cả hệ thống**; lợi ích chính của BigVGAN (khái quát hóa người nói) không cần thiết. Tuy vậy, kỹ thuật **snake activation** có thể thử như cải tiến nhỏ cho iSTFTNet nếu chất lượng 24 kHz còn hạn chế.

### 6.5 Bảng tradeoff vocoder

| Vocoder | Kích thước | Tốc độ (báo cáo) | Chất lượng | ONNX | Khuyến nghị ViFastTts |
| --- | --- | --- | --- | --- | --- |
| HiFi-GAN (full) | cỡ chục M tham số (V1; không trích số chính xác ở đây) | 167,9× realtime (V100, 22 kHz) | Rất tốt | Dễ (chuẩn Piper) | **Baseline bắt buộc** |
| HiFi-GAN V3 (small) | ~0,92M (con số bài báo) | 13,4× realtime (CPU) | Tốt (bài báo) | Dễ | Phương án nhẹ nếu thiếu CPU |
| iSTFTNet | Nhẹ hơn HiFi-GAN cùng cấu hình | Nhanh hơn HiFi-GAN | "Reasonable" (bài báo); chi tiết cao tần có thể giảm | Cần xử lý iSTFT (mục 9.12) | **Mục tiêu chính** |
| iSTFTNet2 | Nhẹ hơn iSTFTNet | Nhanh hơn iSTFTNet | Tương đương iSTFTNet | Như trên (1D–2D conv nhiều hơn) | Tùy chọn nâng cấp |
| Vocos | ConvNeXt (nhỏ; không trích số tham số chính xác ở đây) | ~10× nhanh hơn time-domain | Ngang SOTA | Cần iSTFT ngoài/istft-onnx | Tùy chọn thay thế |
| BigVGAN | Tới 112M | Chậm hơn (tham số lớn) | SOTA universal | Được nhưng nặng | Không dùng cho edge |

---

## 7. Các dự án TTS tiếng Việt đã biết

| Dự án | Kiến trúc | Ghi chú đã xác minh | Bài học cho ViFastTts |
| --- | --- | --- | --- |
| **VietTTS** (`NTT123/vietTTS`, MIT, ngừng cập nhật) | Duration model riêng + acoustic model kiểu **Non-Attentive Tacotron** (encoder conv+BiLSTM; upsampling mềm bằng softmax qua vị trí; decoder LSTM có prenet/postnet, **sinh từng frame khi suy luận**) + **HiFi-GAN**; align bằng **MFA**; JAX/Haiku | README: "Duration model + Acoustic model + HiFiGAN vocoder"; mã `vietTTS/nat/model.py` xác nhận cấu trúc trên; có demo HF Spaces | Chứng minh pipeline duration + aligner ngoài + HiFi-GAN chạy tốt cho tiếng Việt một giọng; nhưng decoder AR theo frame là điểm ViFastTts cần tránh |
| **ViTTS** (`manhph2211/ViTTS`) | **PortaSpeech** (mixture alignment, VAE nhẹ, flow postnet) + **iSTFTNet**, multi-speaker, MFA, DeepSpeaker, đánh giá MOSNet | README xác nhận; có G2P tiếng Việt tự viết; không phát hành checkpoint | Tiền lệ trực tiếp nhất: **iSTFTNet được dùng cho tiếng Việt**; nhưng multi-speaker + VAE latent + flow postnet là phức tạp không cần thiết cho ViFastTts |
| **VieNeu-TTS** (`pnnbao97/VieNeu-TTS`, Apache-2.0; dòng mở mạnh nhất 10/2026) | **v1** (10/2025): **553M**, fine-tune từ `neuphonic/neutts-air` (LLM + NeuCodec). **v2 / 0.3B** (1/2026, deprecated): **244M**, NeuCodec, podcast + En–Vi. **v3 Turbo** (6/2026, dòng mở hiện hành): **131M**, codec LM **from-scratch** (12 lớp × 768, GQA), codec `MOSS-Audio-Tokenizer-Nano` (**48 kHz**, 16 VQ), phonemizer `sea-g2p`, 23–25 giọng preset + clone 3–8 s, emotion cues, streaming OpenAI-compatible. **v3 Nano** (9/2026, preview): flow-matching OT-CFM **48M**, ONNX-only, 24 kHz, chất lượng thấp hơn rõ. **v4** (10/2026): **đóng**, chỉ API/vieneu.io | README + model card xác minh: CPU v3 Turbo **RTF 0,55–0,62 fp32 / 0,35–0,37 int8** (i5-12th), GPU batched 0,011–0,02; Nano **RTF 0,22 (16 bước) / 0,11 (8 bước)**, load ~3 s; ONNX runtime CPU ≈ 0,5 GB (`vieneu_backbone_shared.data` 415 MB + acoustic 14 MB + heads 26 MB + speaker encoder 28 MB + denoiser 43 MB); checkpoint bf16: v1 553M · v2 244M · v3 Turbo 131M · Nano 48M | (1) **Xác nhận trần RTF CPU của codec-LM AR ≈ 0,35–0,62** — chậm hơn hẳn pipeline NAR + Vocos của ViFastTts (đo thật: Vocos ONNX **0,0031**); (2) ví dụ tốt nhất về đóng gói ONNX/streaming/API + LoRA cho tiếng Việt; (3) codec 48 kHz **không dùng lại được** cho pipeline mel-vocoder (R7-1); (4) dùng làm **baseline ngoài** (pip `vieneu`, CPU torch-free) để so CER/nghe trên cùng bộ câu eval — không đổi kiến trúc |
| **V-TTS** (`tronghieuit/v-tts`, CC BY-NC 4.0) | Họ **VITS** (tham số `noise_scale_w`, `sdp_ratio` 0/1), multi-speaker 5 giọng, zero-shot cloning, speaker/style encoder, prosody predictor F0/energy | README: 74,8M tham số tổng (synthesizer 56,45M), **RTF CPU ~0,24–0,48** trên i5-14500; ONNX deployment (~165 MB) cho edge/web/Android; `sdp_ratio=0` = chế độ deterministic | Xác nhận "VITS tối giản + ONNX" khả thi cho tiếng Việt; nhưng speaker/style encoder + cloning đi ngược C1; license hạn chế dùng thương mại |
| **F5-TTS tiếng Việt** (`nguyenthienhy/F5-TTS-Vietnamese`, `2tocom/F5-TTS-Vietnamese-Google-Colab`) | Fine-tune **F5-TTS** (flow matching DiT) cho tiếng Việt, zero-shot clone | Repo + HF Space công khai (theo kết quả tìm kiếm; chưa fetch trực tiếp) | Chất lượng zero-shot tốt nhưng cần GPU, cần audio tham chiếu, không deterministic ⇒ chỉ dùng tham chiếu |
| **PhoAudiobook / Zero-Shot TTS for Vietnamese** (Vu et al., ACL 2025) | Bộ dữ liệu **941 giờ** tiếng Việt; thử nghiệm **VALL-E, VoiceCraft, XTTS-v2** | Abstract xác nhận; kết luận: dữ liệu này cải thiện cả 3 mô hình; VALL-E/VoiceCraft tốt hơn với câu ngắn | Bằng chứng định lượng: con đường zero-shot cho tiếng Việt cần aligner/codec + hàng trăm–hàng nghìn giờ; không phù hợp dữ liệu vừa phải |

**Tổng hợp mục 7:** cộng đồng tiếng Việt đã đi cả 3 hướng: (a) NAR duration+acoustic+HiFi-GAN/iSTFTNet (VietTTS, ViTTS), (b) VITS tối giản ONNX (V-TTS), (c) codec LM/flow matching zero-shot (VieNeu, F5-TTS-Vi). **Chưa có dự án công khai nào** kết hợp: một giọng + cấu trúc âm tiết tường minh + NAR deterministic + ONNX + iSTFTNet như ViFastTts. Khoảng trống này vừa là rủi ro (chưa có gì để so sánh) vừa là cơ hội (thiết kế đã đi đúng chỗ mà các dự án trên phải đánh đổi: họ hy sinh tính deterministic/ONNX để lấy cloning/đa giọng).

**Cập nhật 10/2026 — dòng VieNeu (đối chiếu trực tiếp với ViFastTts):**

| Hạng mục | ViFastTts (kế hoạch hiện tại) | VieNeu v3 Turbo (mở) | VieNeu v3 Nano (preview) |
| --- | --- | --- | --- |
| Kiến trúc | NAR FS2 + variance tường minh → **Vocos 13,5M** (24 kHz, mel) | **AR codec-LM 131M** + codec MOSS 48 kHz (16 VQ) | **Flow-matching OT-CFM 48M** + codec, 24 kHz |
| Tham số / dấu chân | ~21M (acoustic 7,4M + vocoder 13,5M); ONNX vài chục MB | 131M; ONNX CPU ~0,5 GB | 48M; ONNX ~280 MB |
| Dữ liệu | **45 h, một giọng** (kênh của bạn) | ~10.000 h EN–Vi, đa giọng | ~1.000 h, đa giọng |
| Suy luận CPU | Vocos ONNX RTF **0,0031** + acoustic 1 forward (ước tính toàn pipeline ≪ 0,1) | RTF **0,55–0,62** (fp32) / 0,35 (int8) | RTF **0,22 / 0,11** |
| Tính năng | Một giọng, điều khiển duration/F0/energy, frontend âm tiết + thanh điệu, deterministic | 23–25 preset + clone 3–8 s, emotion cues, streaming, En–Vi | 6–11 preset + clone, emotion cues |
| License | repo MIT; checkpoint Vocos MIT | Apache-2.0 (v4 đóng, API-only) | Apache-2.0 |
| Chất lượng công bố | Chưa train xong (đang dựng lại v2) | Không công bố CER/MOS (chỉ demo + tốc độ) | Tác giả ghi "thấp hơn Turbo rõ, đặc biệt tiếng Anh" |
| Dùng gì từ họ | — | sea-g2p (đã ghi trong LICENSES), mẫu streaming/OpenAI API, recipe LoRA; **baseline ngoài** để so CER trên cùng bộ câu eval | Đối thủ 24 kHz gần tốc độ hơn để tham chiếu |

Kết luận: VieNeu là **baseline mạnh nhất để so sánh**, không phải kiến trúc để thay — họ đổi lấy cloning/đa giọng/48 kHz bằng kích thước ~6× và RTF CPU ~100× (so đường Vocos của ta); codec neural không thay được mel-vocoder. Việc nên làm: khi acoustic xong, chạy `pip install vieneu` (CPU, torch-free) trên ~100 câu eval cố định, đo CER + nghe song song — đây là mốc "SOTA mở của tiếng Việt" cho cổng nghiệm thu của dự án.

---

## 8. Bảng so sánh tổng hợp

Chú thích: "—" = không có số liệu được công bố trong nguồn đã xác minh. Các con số là **báo cáo của chính bài báo/repo**, không phải benchmark do tài liệu này tạo ra.

| Họ / mô hình | Chất lượng (báo cáo) | Tốc độ / RTF | Deterministic | ONNX | Dữ liệu cần | Rủi ro chính | Kết luận cho ViFastTts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Tacotron 2, Transformer TTS** (AR attention) | Rất cao (thời điểm 2017–2019); Transformer TTS MOS 4,39 vs GT 4,44 | Chậm: tuần tự theo frame | Không (độ dài/sinh phụ thuộc lịch sử; có thể greedy nhưng dễ lỗi) | Export được nhưng cần vòng lặp/cache ngoài graph | 5–20 h cho một giọng | Lặp/nuốt/dừng sớm; không ràng buộc duration | **Loại** |
| **Codec LM: VALL-E / VALL-E 2 / VoiceCraft / CosyVoice 2 / XTTS-v2 / Fish Speech** | SOTA zero-shot; VALL-E 2 đạt human parity (LibriSpeech/VCTK); NS3/F5 tương tự | AR từng token; RTF CPU của hệ tương đương (VieNeu v3 Turbo) ≈ 0,35–0,62 | Không (sampling rời rạc) | Có thể (ONNX Runtime CPU, ví dụ VieNeu) nhưng nặng | 10.000–60.000 h | Vi phạm C1/C4/C6; phụ thuộc audio prompt | **Loại** (chỉ tham chiếu chất lượng) |
| **NAR feed-forward: FastSpeech 1/2/2s, FastPitch, Parallel Tacotron, DelightfulTTS** | FS2: có thể vượt AR; FastPitch: điều khiển pitch; Parallel Tacotron: ngang AR | FS1 270×/38×; FS2 47×/51× so AR; 1 forward | **Có** (với duration đã tính) | **Rất tốt** (op cơ bản) | 5–50 h đủ dùng | Chất lượng phụ thuộc duration/F0/energy target; cần aligner | **Xương sống** |
| **JETS** (FS2 + HiFi-GAN joint) | Vượt pipeline hai tầng (bài báo) | 1 forward cả chuỗi; train joint | **Có** | Tốt (nếu discriminator tách khỏi graph inference) | 10–50 h | Joint training kém ổn định hơn huấn luyện tách | **Giai đoạn 5** (joint fine-tune) |
| **VITS / VITS2** | VITS: MOS ≈ GT trên LJSpeech; VITS2: tốt hơn, ít phụ thuộc phoneme hơn | Nhanh (1 forward), nhưng có flow + duration flow | **Không** mặc định (stochastic duration/prior); có thể cấu hình deterministic | Được (Piper) nhưng khó hơn: flow, coupling, sampling | 10–30 h | Stochastic; debug khó; quantization nhạy cảm | **Không chọn làm đường chính**; tham chiếu |
| **StyleTTS 2 (decoder) / Kokoro-82M** | StyleTTS2 vượt GT trên LJSpeech single-speaker, ngang VCTK; Kokoro "chất lượng tiệm cận mô hình lớn" (card) | Kokoro có bản ONNX phổ biến (kokoro-onnx, sherpa-onnx); đánh giá cộng đồng nói "gần thời gian thực" trên CPU (chưa xác minh con số) | Có nếu cố định style vector (bỏ style diffusion) | **Có, phổ biến** (kokoro-onnx, sherpa-onnx) | StyleTTS2 gốc: cần WavLM + SLM; Kokoro: vài trăm giờ | Kiến trúc gốc cần SLM discriminator (không có cho tiếng Việt); style diffusion stochastic | **Mượn decoder iSTFTNet**; bỏ style diffusion |
| **NaturalSpeech 2 / 3** | SOTA zero-shot cả hát; NS3 ngang người thật | Chậm (diffusion nhiều bước + codec) | Không (diffusion ngẫu nhiên) | Rất khó (RVQ + diffusion + codec) | Quy mô rất lớn (bài báo huấn luyện trên dữ liệu đa người nói quy mô lớn; không nêu con số ở đây) | Quy mô, dữ liệu, vi phạm C5/C6 | **Loại**, dùng ý tưởng factorization |
| **Diffusion: Grad-TTS, Diff-TTS, E3** | Cạnh tranh/ngang SOTA thời điểm | N× forward; Diff-TTS 28× realtime trên 2080Ti | Không (trừ khi cố định nhiễu + bước) | Có thể nhưng N lớn gây RTF cao; E3 bỏ mel | Cỡ 10–100 h (Diff-TTS/E3 dùng LJSpeech 22,05 kHz) | NFE; mất mel trung gian (E3) | **Loại** làm xương sống |
| **Flow matching: Matcha-TTS, E2-TTS, F5-TTS** | Matcha: MOS cao nhất trong so sánh của bài báo; E2: human-level + SOTA similarity | 8–32 bước ODE; Nano (48M, ONNX) RTF CPU 0,11–0,22 | Deterministic **nếu** cố định nhiễu + solver + bước | Có (Matcha trong sherpa-onnx; Nano ONNX) | Matcha: cỡ 5–44 h (thí nghiệm bài báo); E2/F5: zero-shot, dữ liệu lớn | NFE làm RTF CPU cao hơn NAR 1 bước; nhiễu "nhớ" trong graph | **Loại** làm đường chính; theo dõi |
| **Vocoder HiFi-GAN** | Tương đương người (bài báo) | 167,9× (V100); 13,4× CPU (bản nhỏ) | Có (với mel đầu vào đã fix) | **Dễ nhất** | — | Không | **Baseline** |
| **Vocoder iSTFTNet / iSTFTNet2** | "Reasonable", nhẹ hơn | Nhanh hơn HiFi-GAN | Có | Cần iSTFT trong graph (mục 9.12) | — | iSTFT/ONNX là điểm cần tự cài | **Mục tiêu chính** |
| **Vocoder Vocos** | Ngang SOTA | ~10× nhanh hơn time-domain | Có | iSTFT ngoài graph hoặc istft-onnx | — | Phase prediction nhạy quantization | Tùy chọn thay thế |
| **Vocoder BigVGAN** | SOTA universal | — | Có | Được, nhưng nặng (tới 112M) | — | Quá lớn cho edge 24 kHz | Không dùng |

**Kết luận bảng:** sự kết hợp hợp lý cho ViFastTts là **họ NAR feed-forward (FastSpeech 2 làm xương sống) + variance tường minh (duration/F0/energy/[phonation]) + decoder mel nhẹ + vocoder iSTFTNet (baseline HiFi-GAN) + MFA/MAS cho alignment + ONNX với iSTFT tự cài**. Toàn bộ phần "2023–2026" (codec LM, flow matching, diffusion, style diffusion) bị loại khỏi đường triển khai do vi phạm các ràng buộc C1/C4/C6/C7, nhưng các *thành phần* của chúng (MAS, prior loss, factorized attributes, frequency-domain vocoder, snake activation) được chọn lọc dùng lại.

---

## 9. Đề xuất kiến trúc v2 cho ViFastTts

Đề xuất dưới đây **giữ nguyên triết lý** của `docs/DESIGN_V2.md` và **cụ thể hóa** những điểm còn mơ hồ hoặc chưa được triển khai trong code hiện tại (`src/vifasttts/models/vifasttts.py` hiện chỉ expand ở cấp âm tiết, variance ở cấp âm tiết). Nội dung là "v2.1": cùng sơ đồ, thêm cấp segment và các quyết định kỹ thuật để ONNX + deterministic không còn rủi ro.

### 9.1 Sơ đồ tổng thể (inference)

```text
Raw text
  → chuẩn hóa Unicode/ngữ cảnh (đã có)
  → tách câu, phân tích âm tiết (đã có)
  → structured syllables [B, T_syl, 7]
  → Component Embedding (cộng hoặc concat-linear)
  → Syllable Conformer Encoder                      # ngữ cảnh cấp âm tiết
  → Segment Tripler/Expander + mask                  # onset/medial/nucleus/coda
  → Duration Predictor (deterministic)               # duration từng segment + pause
  → Length Regulator cấp segment                     # ra frame, đúng tổng length
  → F0 / Voicing / Energy / Phonation Predictors     # đều deterministic
  → Variance Projection + cộng vào frame features
  → Acoustic Mel Decoder (Conformer/Conv nhẹ)
  → Mel 100 bins @ 24 kHz (hop 256)
  → iSTFTNet (24 kHz)
  → PCM 24 kHz
```

Ràng buộc: không có bất kỳ phép lấy mẫu nào trong graph; mọi dropout tắt ở eval; mọi nhánh phụ (tone classifier, alignment head) chỉ tồn tại khi train.

### 9.2 Đầu vào và embedding

- Giữ đúng 7 trường như `DESIGN_V2.md` mục 15: `onset, medial, nucleus, coda, tone, boundary, punctuation`.
- Embedding: bắt đầu bằng **concat + Linear** (giữ thông tin riêng của từng trường), `component_dim=96`, `hidden=256`. Không dropout tone/nucleus/coda (thông tin bắt buộc phát âm); có thể dropout nhẹ boundary (10%) sau khi có baseline.
- Lý do không dùng chuỗi ký tự: vocab nhỏ, không có unknown ngoại trừ từ ngoại lai đã có fallback ở frontend, và tone là biến rời rạc 6 giá trị — có thể dùng trực tiếp làm điều kiện.

### 9.3 Text encoder (cấp âm tiết)

- Conformer nhẹ: `6 lớp, hidden 256, 4 head, FFN 1024, conv kernel 15, dropout 0.1` (đúng `base_24k.yaml`).
- Vai trò: ngữ cảnh câu, ảnh hưởng dấu câu, phối hợp âm tiết kề (nối âm), cung cấp state cho mọi predictor. Đây là điểm tương đồng với "variance adaptor input" của FastSpeech 2, nhưng ở cấp âm tiết thay vì phoneme.
- Không dùng BERT/SLM pretraining: một ngôn ngữ, một giọng, vocab nhỏ; chi phí vô ích và làm phình graph ONNX.

### 9.4 Segment Expander + alignment (quyết định quan trọng nhất của v2.1)

**Vấn đề hiện tại:** code baseline mở rộng theo từng *âm tiết* (một duration cho cả âm tiết), trong khi `DESIGN_V2.md` mục 18 yêu cầu duration ở cấp *segment* (onset/medial/nucleus/coda). Duration cấp segment là điều kiện để phonation/F0/energy bám đúng cấu trúc phát âm (ví dụ: vang của nguyên âm khác phụ âm đầu; tone khu trú ở nucleus).

**Thiết kế đề xuất:**

1. Từ state âm tiết `h_t`, tạo 4 vector vai trò bằng **role embedding + Linear** (onset/medial/nucleus/coda), cộng thêm mask hoạt động (ví dụ `ăn`: onset/medial inactive; `hoa`: coda inactive). Không tạo segment cho tone.
2. **Alignment (train):**
   - Giai đoạn đầu (v2.0): **aligner ngoài (MFA)** trên phoneme, rồi **gộp phoneme → segment bằng luật + ánh xạ từ điển** (parser biết grapheme ↔ abstract, nên gộp được: `ngh`→ONSET, `uô`→NUCLEUS, `ng`→CODA...). Cần lọc các trường hợp aligner không khớp ranh giới; lưu `alignment_confidence`.
   - Giai đoạn sau (v2.2): thêm **aligner tự học** để giảm phụ thuộc MFA. Hai lựa chọn đã được chứng minh: **MAS** (Glow-TTS: hard monotonic + quy hoạch động) và **forward-sum + Viterbi + static prior** (RAD-TTS / "One TTS Alignment To Rule Them All" — framework này cải thiện cả Tacotron 2, FastSpeech 2, FastPitch, RAD-TTS). Nếu cần nhanh hơn, tham khảo **Super MAS** (2024). Khuyến nghị: bắt đầu bằng forward-sum + static prior vì nó không yêu cầu kiến trúc flow và dễ chèn vào mô hình NAR hiện có.
3. **Ràng buộc duration cấp segment:**
   - `duration(segment active) ≥ 1 frame`; `duration(inactive) = 0`.
   - Tổng duration khớp số frame của mel target (sai lệch do làm tròn được phân bổ lại bằng "largest remainder" thay vì cộng dồn sai số — tránh lệch pha toàn câu).
   - Với câu tiếng Việt, tỉ lệ duration nucleuscoda ~ ổn định; có thể thêm **monotonic bias** (ví dụ: trên mỗi âm tiết, duration nằm trong khoảng percentile của tập train) để chặn các dự đoán cực đoan.
4. **Khi suy luận**, expander chỉ còn: role embedding + length regulator (repeat theo duration đã làm tròn). Toàn bộ đều là op tĩnh, export ONNX tốt.

### 9.5 Duration predictor (deterministic)

- Cấu trúc: 3 lớp conv/Transformer nhỏ trên state segment, đầu ra `log(1 + d)` cho từng vai trò; loss `MSE(log(1+d_pred), log(1+d_true))` (đúng `DESIGN_V2.md` mục 27.1). Không dùng flow/stochastic duration như VITS.
- Đầu vào gồm: state âm tiết, role, tone, boundary/punctuation. **Tone phải ảnh hưởng duration** (đã ghi trong `DESIGN_V2.md`; có cơ sở ngôn ngữ học: thanh sắc/nặng/nã ngắn hơn thanh ngang/huyền ở nhiều nghiên cứu tiếng Việt — cần xác nhận trên chính dữ liệu train bằng phân tích thống kê duration theo tone).
- Tốc độ: `d' = d / speed`, kẹp `speed ∈ [0.75, 1.35]` (như thiết kế). Có thể đưa `speed` vào graph như một scalar input để runtime khỏi phải tính lại.
- Tùy chọn tăng cường (VITS2 gợi ý): thêm một **discriminator trên phân bố duration ở cấp segment** khi train để tránh duration "trung bình hóa" quá mức. Không ảnh hưởng inference.

### 9.6 F0 và hữu thanh (voicing)

- Giữ triết lý "tone-conditioned contour + sentence intonation residual" (`DESIGN_V2.md` mục 20).
- Biểu diễn: `log-F0` đã chuẩn hóa theo phân bố của người nói (trừ mean, chia std của voiced frames) — giúp giảm phương sai và ổn định khi quantize; khi suy luận cộng lại mean/std của checkpoint (hằng số lưu trong metadata model).
- Đầu ra: (i) log-F0 trung bình cấp segment (ổn định, dễ học), (ii) **contour residual cấp frame trong segment** (tùy chọn, bật khi đủ dữ liệu), (iii) xác suất hữu thanh (voiced/unvoiced) per frame.
- Loss: L1 trên log-F0 (chỉ trên frame hữu thanh), L1 trên delta-F0 (bắt độ dốc contour — quan trọng cho thanh hỏi/ngã), BCE cho voicing, và **loss phụ phân loại 6 thanh** từ hidden state acoustic (đã có `tone_classifier` trong code; giữ như auxiliary, tắt ở inference).
- Vì sao không dùng F0 của FastSpeech 2 dạng CWT: CWT thêm một nhánh tần số phức tạp, lợi ích chính là khi pitch biến động mạnh nhiều người nói; với một người nói, log-F0 + delta loss đơn giản hơn và ONNX thân thiện hơn.

### 9.7 Energy và phonation

- **Energy**: log-RMS per frame (đã chuẩn hóa), predictor 2 lớp conv, điều kiện hóa bởi tone + boundary + role. Vai trò: giữ phụ âm rõ, nhấn câu vừa phải, tránh giọng phẳng (`DESIGN_V2.md` mục 21).
- **Phonation (mới, đánh dấu thử nghiệm)**: thay vì để "phonation" là khái niệm mơ hồ, đề xuất 2–4 kênh **đo được** và dễ trích xuất:
  1. **Độ nghiêng phổ / spectral tilt** (xấp xỉ H1–H2: hiệu năng lượng hài bậc 1 và bậc 2) — tương quan với "giọng thở/giọng chặt";
  2. **HNR xấp xỉ** (harmonic-to-noise ratio) hoặc tỉ lệ năng lượng hài/tổng ở dải voiced;
  3. (tùy chọn) **độ biến thiên F0 cục bộ** (proxy cho rung/creak — liên quan trực tiếp đến thanh ngã trong tiếng Việt, nơi hiện tượng "vỡ giọng/glottalization" thường gặp);
  4. (tùy chọn) xác suất "tắt thanh quản" (glottalization) như một phân loại nhị phân per frame.
- Predictor: 2–3 lớp conv, đầu ra là vector phonation (thay vì scalar), điều kiện hóa bởi tone (vì thanh ngã/hỏi có đặc trưng phonation khác biệt). Cơ chế ghép vào decoder giống hệt energy/F0 (variance adaptor của FastSpeech 2).
- **Lộ trình an toàn**: bật phonation sau khi có baseline duration/F0/energy ổn định và **chỉ khi** dữ liệu đủ sạch (≥ 20 giờ cùng điều kiện thu). Kiểm tra lợi ích bằng listening test A/B + đo lỗi thanh; nếu không cải thiện, tắt kênh này (giữ code có cờ config). Cơ sở học thuật: mô hình hóa phonation/creak tường minh đã được chứng minh có thể điều khiển (ICPhS 2023: "Neural speech synthesis with controllable creaky voice style"); factorization kiểu NS3 cũng cho thấy lợi ích khi tách thuộc tính.

### 9.8 Acoustic mel decoder

- Conformer/Conv nhẹ: `4 lớp, hidden 256, 4 head, FFN 1024` (đúng `DESIGN_V2.md` mục 22.2), đầu ra `Linear → 100 mel bins`.
- Đầu vào frame: `expanded segment features + F0 embedding + voicing + energy + phonation + frame position in segment + duration/role`. 
- "Frame position in segment" (0→1) là đặc trưng quan trọng: giúp decoder biết đang ở đầu/giữa/cuối nguyên âm — thay cho positional encoding tương đối trong các mô hình phoneme-level.
- Cửa sổ attention giới hạn (ví dụ ±3.2 s hoặc theo câu) để tránh quadratic trên câu dài; hoặc dùng conv + local attention. Khi export ONNX cần độ dài động (dynamic axes) — đã được hỗ trợ.

### 9.9 Vocoder iSTFTNet 24 kHz

Cấu hình đề xuất (thêm vào `configs/base_24k.yaml` như nhánh vocoder):

```yaml
vocoder:
  type: istftnet
  sample_rate: 24000
  n_fft: 1024
  win_length: 1024
  hop_length: 256              # khớp mel hop; 4 frame/window
  iSTFT_after: last_two_blocks # thay 2 tầng upsample cuối
  upsample_initial_channel: 256
  upsample_rates: [8, 8, 2]    # 128, iSTFT "vượt" phần còn lại theo thiết kế gốc
  resblock_kernel_sizes: [3, 7, 11]
  resblock_dilation_sizes: [[1,3,5],[1,3,5],[1,3,5]]
  # baseline đối chứng:
  baseline_hifigan:
    upsample_rates: [8, 8, 2, 2]
```

Lưu ý triển khai: giữ cấu hình generator HiFi-GAN **có thể chạy như baseline** bằng cờ `type: hifigan` — điều này cho phép so sánh A/B trên cùng mel và cùng đường ống, đúng yêu cầu "có baseline iSTFTNet" (ở đây iSTFTNet là mục tiêu, HiFi-GAN là baseline).

### 9.10 Hệ loss

```text
L_total =
    λ_mel      × L1(mel)                      # chính
  + λ_duration × MSE(log(1+d))                # deterministic
  + λ_f0       × L1(log-F0) + λ_f0delta × L1(delta log-F0)
  + λ_vuv      × BCE(voiced)
  + λ_energy   × L1(log-energy)
  + λ_phonation× L1(phonation vector)         # tắt được
  + λ_tone     × CE(6 thanh)                  # auxiliary, tắt ở inference
  + λ_align    × L_forward-sum/MAS            # khi dùng aligner tự học
  + λ_stft     × Multi-resolution STFT loss   # vocoder
  + λ_adv      × Adversarial (MPD/MSD)        # vocoder
  + λ_fm       × Feature matching             # vocoder
```

Giai đoạn joint (JETS-style) thêm `λ_joint × L1/L_stft(mel→waveform)` với learning rate thấp, chỉ khi acoustic + vocoder đã hội tụ riêng.

### 9.11 Lộ trình huấn luyện

| Giai đoạn | Nội dung | Tiêu chí chuyển tiếp |
| --- | --- | --- |
| 0. Parser + dataset | Unit test parser 100%; manifest; coverage theo onset/nucleus/coda/tone; alignment MFA | Không có âm tiết `UNKNOWN` trong train; coverage tone-coda đủ |
| 1. Vocoder | Train iSTFTNet trên **mel thật**; train HiFi-GAN baseline cùng cấu hình; nghe A/B | So sánh MOS nội bộ/MCD; chọn iSTFTNet nếu tương đương |
| 2. Acoustic với GT variance | Train decoder mel với duration/F0/energy **thật** (từ aligner + trích xuất) | Mel L1 ổn định; nghe rõ phụ âm |
| 3. Predictors | Đóng băng (hoặc giảm LR) decoder, train duration + F0 + voicing + energy | Sai số duration/F0 tiệm cận mức GT-noise (xác định bằng cách đo trên tập hold-out) |
| 4. Full predicted | Chạy toàn bộ với giá trị dự đoán | Không active-segment duration 0; không lặp/nuốt trên bộ câu chuẩn |
| 5. Joint fine-tune | JETS-style acoustic + vocoder, LR thấp; thêm adversarial | Không giảm chất lượng trên bộ test cũ; MOS tăng |
| 6. ONNX + tối ưu | Export FP32 → FP16 → INT8 từng module; benchmark; kiểm tra tương đương | Sai số output < ngưỡng đặt trước; RTF đạt mục tiêu |

### 9.12 Chiến lược ONNX (phần kỹ thuật quan trọng nhất)

**a) Chia graph hay gộp?**

Đề xuất **2 graph** (khuyến nghị): 
1. `acoustic.onnx`: text features → mel (+ có thể xuất luôn duration/F0/energy để debug). 
2. `vocoder.onnx`: mel → waveform.

Lý do: (i) có thể cập nhật/quantize riêng vocoder (khối nặng nhất); (ii) mel là điểm kiểm tra trung gian hữu ích (so sánh PyTorch vs ONNX tại cùng một tensor); (iii) tránh một graph quá lớn khó chẩn đoán. Sau khi ổn định, có thể gộp thành 1 graph cho runtime gọn (tùy nhu cầu).

**b) iSTFT trong ONNX — 3 phương án, xếp theo độ khuyến nghị**

1. **Tự cài iSTFT bằng op cơ bản (khuyến nghị).** iSTFT là một phép biến đổi tuyến tính cố định: mỗi frame phổ (real/imag, `n_fft/2+1` hệ số) được biến thành một đoạn dài `n_fft` (nhân với ma trận cơ sở cos/sin — hoàn toàn tính trước), nhân cửa sổ tổng hợp, rồi **overlap-add**. Overlap-add của các đoạn cách nhau `hop` chính là một phép **`ConvTranspose1d`** với `in_channels = 2*(n_fft/2+1)`, `out_channels = 1`, `kernel_size = n_fft`, `stride = hop`, trọng số = ma trận cơ sở × cửa sổ. Do đó toàn bộ iSTFT export được thành các op chuẩn (`ConvTranspose1d`/`MatMul` + phép chuẩn hóa envelope hằng số với Hann/hop = win/4). Đây là cách duy nhất **chắc chắn chạy** trên mọi execution provider (CPU/ARM/NNAPI…) và **quantize INT8 được** (vì chỉ là conv).
2. **Xuất phổ ra khỏi graph, làm iSTFT ở runtime (Rust/C++).** Phù hợp với `RUST_ROADMAP.md`: graph ONNX chỉ trả mag/phase (hoặc real/imag) per frame; runtime thực hiện overlap-add/FFT ngược bằng code tối ưu. Ưu điểm: không phụ thuộc bất kỳ hành vi nào của ONNX Runtime; nhược điểm: phải viết + test phần này ở mỗi ngôn ngữ runtime.
3. **Dùng graph ISTFT sẵn có (fallback).** `mush42/istft-onnx` sinh một graph ONNX thực hiện ISTFT từ mag/phase (thiết kế cho Vocos; README nói chậm hơn torch ~4× nhưng chi phí iSTFT nhỏ so với toàn hệ). Có thể nối sau graph vocoder. Không nên dựa vào op `STFT`/`ISTFT` của ONNX spec cho đường triển khai chính vì ONNX **không có** op ISTFT (đề xuất còn mở) và hỗ trợ STFT khác nhau giữa các provider/runtime.

**c) Danh sách op an toàn / cần tránh**

- An toàn: `Conv`, `ConvTranspose`, `MatMul`, `Gemm`, `LayerNormalization`, `BatchNormalization` (fuse được), `GELU/SiLU/Tanh/LeakyReLU/Sigmoid`, `Softmax`, `Gather`, `RepeatInterleave` (export thành `Expand`/`Tile`/`Reshape`), `Concat`, `Slice`, `Pad`, `Erf` (dùng bởi GELU — hỗ trợ rộng nhưng có thể thay bằng biến thể GELU tanh), `Pow`, `Sqrt`, `ReduceMean`.
- Cần tránh hoặc phải kiểm thử kỹ: `STFT`/`ISTFT`, `Loop`/`Scan` (không cần), bất kỳ op custom (`com.microsoft.*`), các phép phụ thuộc length runtime nếu không khai báo dynamic axes.
- Với F0 path (nhạy cảm số học): giữ FP32 khi lượng tử hóa INT8; các lớp `Norm` và `Mel projection` cũng nên giữ FP32/FP16. Điều này khớp cảnh báo trong `DESIGN_V2.md` mục 30.6.

**d) Dynamic axes và các biến thể kích thước**

- Khai báo dynamic cho: `T_syl` (số âm tiết), `T_frames` (số frame), `n_samples`.
- `duration → repeat` cần biểu diễn không phụ thuộc vòng lặp Python: dùng `RepeatInterleave` hoặc `MatMul` với ma trận one-hot (T_frames × T_segments) — cả hai export được; ma trận one-hot này xây bằng `cumsum`/`arange` tĩnh (không có vòng lặp động).
- Kiểm thử bắt buộc: chạy ONNX so với PyTorch trên 3 nhóm độ dài (ngắn/trung/dài) + 1 câu có speed 0.75/1.0/1.35; ngưỡng sai số mel ~1e-4 và waveform tương quan cao (> 0.999) với FP32.

**e) Quantization**

- INT8: ưu tiên embeddings, encoder Linear, duration/energy predictor (khớp `DESIGN_V2.md`).
- FP16: vocoder và mel projection (giảm rủi ro biên).
- Giữ FP32: F0 path, các lớp Norm.
- Calibration set phải phủ đủ 6 thanh × các coda đặc biệt (`-p/-t/-c/-ch/-nh/-ng`) — nếu không, INT8 dễ méo ở các âm tiết tắc (stop coda).

### 9.13 Bảng truy vết: yêu cầu → thành phần thiết kế

| Yêu cầu | Thành phần đáp ứng |
| --- | --- |
| Một giọng, không id điều kiện | Không embedding speaker/style/dialect; chất giọng nằm trong trọng số (mục 9.2) |
| Cấu trúc âm tiết tường minh | 7 trường input + segment expander theo role + tone conditioning (9.2, 9.4) |
| Deterministic | Duration/F0/energy/phonation regression thuần; mọi stochastic/VAE/sampling bị loại; dropout off ở eval (9.5–9.7) |
| Non-autoregressive | 1 forward acoustic + 1 forward vocoder (9.1, 9.11) |
| Ràng buộc đọc đúng, không lặp/nuốt | Duration per segment + ràng buộc min/0 + largest-remainder rounding; không có attention tuần tự (9.4, 9.5) |
| Thanh điệu | F0 tone-conditioned + voicing + delta loss + tone classifier phụ (9.6) |
| ONNX | iSTFT tự cài bằng ConvTranspose; op an toàn; dynamic axes; quantize từng phần (9.12) |
| 24 kHz | `n_fft=1024/hop=256/mel=100` xuyên suốt (9.8, 9.9) |
| Baseline iSTFTNet | Nhánh `hifigan` để A/B (9.9) |
| Tốc độ | 1 forward/mô hình; vocoder frequency-domain nhẹ; không NFE (mục 6, 9.12) |
| Dữ liệu vừa phải | Không cần pretrain LLM/SLM; kiến trúc ~30–80M (mục 3, 4) |

---

## 10. Những gì KHÔNG nên chọn và vì sao

1. **Tacotron 2 / Transformer TTS / mọi decoder AR**: vi phạm C3, dễ lặp/nuốt, cần vòng lặp khi export, không ràng buộc được duration. Đây chính là động cơ lịch sử khiến cả ngành chuyển sang FastSpeech.
2. **Codec language models (VALL-E, VALL-E 2, VoiceCraft, CosyVoice 2, XTTS-v2, Fish Speech, VieNeu)**: điều kiện tiên quyết của chúng là zero-shot/cloning (ngược C1), sampling rời rạc (ngược C4), dữ liệu 10k+ giờ (ngược C7), RTF CPU cao (ngược C6) — dù export ONNX được. Chỉ dùng làm tham chiếu chất lượng/công cụ tạo dữ liệu.
3. **Diffusion (Grad-TTS, Diff-TTS, E3) và flow matching (Matcha, E2, F5) làm xương sống**: NFE lớn, RTF CPU cao; E3/E2/F5 còn bỏ duration/alignment tường minh (mất khả năng kiểm soát phát âm — đúng thứ ViFastTts ưu tiên nhất). Có thể dùng flow như module prosody phụ sau này, nhưng không đặt ở đường chính.
4. **VITS/VITS2 nguyên bản end-to-end làm đường chính**: stochastic duration + prior sampling (ngược C4), flow/coupling + posterior encoder làm graph ONNX và quantization phức tạp hơn; phần "end-to-end" ít giá trị khi ta đã có mel trung gian để kiểm soát. (Vẫn có thể dùng làm mốc so sánh, giống Piper.)
5. **StyleTTS 2 bản gốc**: cần SLM (WavLM) làm discriminator và style diffusion — với một giọng tiếng Việt, ta không có pretraining SLM phù hợp và không cần style. Chỉ **mượn decoder iSTFTNet** (điều Kokoro đã chứng minh là đủ).
6. **NaturalSpeech 2/3 và các hệ codec + factorized diffusion**: kích thước, dữ liệu, số bước đều vượt xa ngân sách edge; FACodec lại ở 16 kHz (ngược C8).
7. **BigVGAN (nhất là bản 112M)**: lợi thế "universal" vô nghĩa với một người nói, nhưng vẫn trả giá tham số/tốc độ. (Có thể mượn `snake activation`.)
8. **Bất kỳ thành phần điều kiện hóa theo `speaker_id`/`dialect_id`/`style_id`/reference audio**: bị cấm bởi C1; kể cả khi thêm để "tiện mở rộng", nó làm tăng bề mặt lỗi (mô hình có thể học sai tương quan id) và làm phình xuất ONNX.
9. **Alignment ngầm bằng attention không ràng buộc**: nguồn gốc của lặp/nuốt; ViFastTts cần alignment đơn điệu tường minh (MFA/forward-sum/MAS).
10. **Vocoder time-domain cỡ lớn (BigVGAN, WaveNet/WaveGlow, diffusion vocoder)**: tất cả đều chậm hơn frequency-domain (iSTFTNet/Vocos) ở cùng ngân sách chất lượng, theo chính các bài báo đã dẫn.
11. **Grapheme-only, bỏ phoneme hoàn toàn** (kiểu Fish Speech "loại bỏ G2P"): hấp dẫn cho vạn ngôn ngữ, nhưng với một ngôn ngữ và mục tiêu phát âm chính xác, cấu trúc âm tiết + luật đọc đã cho ta chính xác cao hơn với dữ liệu nhỏ hơn nhiều.
12. **Streaming từng frame AR (kiểu codec LM streaming)**: vi phạm C3/C4 và không cần thiết cho mục tiêu "đọc đúng, nhanh" — có thể đạt độ trễ thấp bằng cách chia câu thành đoạn và synthesize song song, đơn giản hơn nhiều.

---

## 11. Tài liệu tham khảo

Trạng thái: ✅ = đã tải trực tiếp bằng webfetch; 🔎 = xác minh qua kết quả tìm kiếm (tiêu đề/abstract khớp), chưa fetch trực tiếp; ⚠️ = chưa xác minh đầy đủ.

### AR / attention-based

1. 🔎 Tacotron 2 — *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*, arXiv:1712.05884 — https://arxiv.org/abs/1712.05884
2. ✅ Transformer TTS — Li et al., *Neural Speech Synthesis with Transformer Network*, arXiv:1809.08895 — https://arxiv.org/abs/1809.08895
3. ✅ VALL-E — Wang et al., *Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers*, arXiv:2301.02111 — https://arxiv.org/abs/2301.02111
4. ✅ VALL-E 2 — Chen et al., arXiv:2406.05370 — https://arxiv.org/abs/2406.05370
5. ✅ VoiceCraft — Peng et al., arXiv:2403.16973 — https://arxiv.org/abs/2403.16973
6. ✅ CosyVoice 2 — Du et al., arXiv:2412.10117 — https://arxiv.org/abs/2412.10117
7. 🔎 CosyVoice 1 — arXiv:2407.05407 (được nhắc trong CosyVoice 2; chưa fetch trực tiếp)
8. ✅ XTTS — Casanova et al., *XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model*, arXiv:2406.04904 — https://arxiv.org/abs/2406.04904; card mô hình XTTS-v2: https://huggingface.co/coqui/XTTS-v2 (🔎)
9. ✅ Fish-Speech — Liao et al., arXiv:2411.01156 — https://arxiv.org/abs/2411.01156; repo: https://github.com/fishaudio/fish-speech (🔎)

### NAR / FastSpeech

10. ✅ FastSpeech — Ren et al., arXiv:1905.09263 — https://arxiv.org/abs/1905.09263
11. 🔎 FastSpeech 2 / 2s — Ren et al., arXiv:2006.04558 — https://arxiv.org/abs/2006.04558 (abstract đầy đủ qua tìm kiếm; trang Microsoft Research xác nhận các con số 3,12× / 47× / 51×)
12. 🔎 FastPitch — Łańcucki, arXiv:2006.06873 — https://arxiv.org/abs/2006.06873
13. ✅ Parallel Tacotron — Elias et al., ICASSP 2021 — https://research.google/pubs/parallel-tacotron-non-autoregressive-and-controllable-tts/ (bản ghi Google Research + abstract)
14. 🔎 DelightfulTTS — Liu et al., arXiv:2110.12612 — https://arxiv.org/abs/2110.12612

### VAE / GAN / end-to-end

15. 🔎 VITS — Kim et al., arXiv:2106.06103 — https://arxiv.org/abs/2106.06103; repo https://github.com/jaywalnut310/vits (🔎)
16. ✅ VITS2 — Kong et al., arXiv:2307.16430 — https://arxiv.org/abs/2307.16430
17. 🔎 JETS — Lim et al., arXiv:2203.16852 — https://arxiv.org/abs/2203.16852
18. ✅ StyleTTS 2 — Li et al., arXiv:2306.07691 — https://arxiv.org/abs/2306.07691
19. ✅ Kokoro-82M — model card: https://huggingface.co/hexgrad/Kokoro-82M (xác nhận StyleTTS 2 + ISTFTNet, 82M, 24 kHz, Apache-2.0)
20. 🔎 NaturalSpeech 2 — Shen et al., arXiv:2304.09116 — https://arxiv.org/abs/2304.09116
21. 🔎 NaturalSpeech 3 — Ju et al., arXiv:2403.03100 — https://arxiv.org/abs/2403.03100; giới hạn 16 kHz của FACodec qua Q&A repo https://github.com/lifeiteng/naturalspeech3_facodec (🔎)

### Diffusion / flow matching / foundation

22. ✅ Grad-TTS — Popov et al., arXiv:2105.06337 — https://arxiv.org/abs/2105.06337
23. ✅ Diff-TTS — Jeong et al., arXiv:2104.01409 — https://arxiv.org/abs/2104.01409
24. ✅ E3 TTS — Gao et al., arXiv:2311.00945 — https://arxiv.org/abs/2311.00945
25. 🔎 Matcha-TTS — Mehta et al., arXiv:2309.03199 — https://arxiv.org/abs/2309.03199; repo https://github.com/shivammehta25/Matcha-TTS (🔎)
26. 🔎 E2 TTS — Eskimez et al., arXiv:2406.18009 — https://arxiv.org/abs/2406.18009
27. 🔎 F5-TTS — Chen et al., arXiv:2410.06885 — https://arxiv.org/abs/2410.06885 và repo https://github.com/SWivid/F5-TTS
28. 🔎 Survey on Neural Speech Synthesis — Tan et al., arXiv:2106.15561 — https://arxiv.org/abs/2106.15561
29. 🔎 Recent Advances in Speech Language Models: A Survey — ACL 2025, https://aclanthology.org/2025.acl-long.682.pdf và arXiv:2410.03751
30. 🔎 Towards Controllable Speech Synthesis in the Era of Large Language Models — arXiv:2412.06602 — https://arxiv.org/abs/2412.06602

### Vocoder

31. ✅ HiFi-GAN — Kong et al., arXiv:2010.05646 — https://arxiv.org/abs/2010.05646
32. ✅ iSTFTNet — Kaneko et al., arXiv:2203.02395 — https://arxiv.org/abs/2203.02395
33. ✅ iSTFTNet2 — Kaneko et al., Interspeech 2023 — https://www.isca-archive.org/interspeech_2023/kaneko23_interspeech.html
34. ✅ Vocos — Siuzdak, arXiv:2306.00814 — https://arxiv.org/abs/2306.00814; repo https://github.com/gemelo-ai/vocos (🔎)
35. ✅ BigVGAN — Lee et al., arXiv:2206.04658 — https://arxiv.org/abs/2206.04658

### Alignment

36. ✅ Glow-TTS / MAS — Kim et al., arXiv:2005.11129 — https://arxiv.org/abs/2005.11129
37. ✅ One TTS Alignment To Rule Them All (forward-sum + Viterbi + static prior, RAD-TTS) — Badlani et al., arXiv:2108.10447 — https://arxiv.org/abs/2108.10447
38. 🔎 RAD-TTS — Shih et al. — https://openreview.net/pdf?id=0NQwnnwAORi và https://github.com/NVIDIA/radtts
39. 🔎 Super Monotonic Alignment Search — Lee et al., arXiv:2409.07704 — https://arxiv.org/abs/2409.07704

### ONNX / triển khai

40. 🔎 ONNX STFT operator (opset 17) — https://onnx.ai/onnx/operators/onnx__STFT.html (trang đã tải được nhưng nội dung chính xác nhận qua kết quả tìm kiếm)
41. 🔎 Đề xuất op ISTFT (chưa có) — https://github.com/onnx/onnx/issues/4777
42. 🔎 PyTorch ONNX export & STFT — https://github.com/pytorch/pytorch/issues/81075 ; lỗi tương tự https://github.com/timsainb/noisereduce/issues/93
43. 🔎 Sai lệch PyTorch/ONNX với stft+istft trong model — https://github.com/microsoft/onnxruntime/issues/23219
44. ✅ Graph ISTFT dựng sẵn cho ONNX — https://github.com/mush42/istft-onnx
45. ✅ Piper (VITS → ONNX, Raspberry Pi) — https://github.com/rhasspy/piper (repo đã archive 06/10/2025; phát triển chuyển sang https://github.com/OHF-Voice/piper1-gpl)
46. ✅ sherpa-onnx — danh sách model TTS hỗ trợ ONNX — https://k2-fsa.github.io/sherpa/onnx/tts/all/ ; repo https://github.com/k2-fsa/sherpa-onnx (🔎)
47. 🔎 Hướng dẫn huấn luyện Piper — https://tderflinger.github.io/piper-docs/guides/training/

### Tiếng Việt

48. ✅ VietTTS — https://github.com/NTT123/vietTTS (README + mã nguồn `vietTTS/nat/model.py`)
49. ✅ ViTTS (PortaSpeech + iSTFTNet) — https://github.com/manhph2211/ViTTS
50. ✅ VieNeu-TTS — https://github.com/pnnbao97/VieNeu-TTS
51. ✅ V-TTS — https://github.com/tronghieuit/v-tts
52. ✅ Zero-Shot TTS for Vietnamese (PhoAudiobook, 941 giờ; VALL-E/VoiceCraft/XTTS-v2) — Vu et al., arXiv:2506.01322 — https://arxiv.org/abs/2506.01322
53. 🔎 F5-TTS Vietnamese — https://github.com/nguyenthienhy/F5-TTS-Vietnamese ; https://github.com/2tocom/F5-TTS-Vietnamese-Google-Colab (chưa fetch trực tiếp)
54. 🔎 Nhóm chủ đề GitHub "vietnamese-tts" (danh sách dự án liên quan) — https://github.com/topics/vietnamese-tts (đã fetch trang chủ đề; các repo con được liệt kê trong kết quả)

### Mô hình hóa phonation / creak

55. 🔎 Neural speech synthesis with controllable creaky voice style — Lameris et al., ICPhS 2023 — https://www.internationalphoneticassociation.org/icphs-proceedings/ICPhS2023/full_papers/717.pdf
56. 🔎 The Role of Creaky Voice in Turn Taking — Lameris et al., LREC-COLING 2024 — https://aclanthology.org/2024.lrec-main.1396.pdf

---

*Tài liệu này chỉ khảo sát và đề xuất kiến trúc; không thay đổi mã nguồn hoặc cấu hình huấn luyện hiện có.*

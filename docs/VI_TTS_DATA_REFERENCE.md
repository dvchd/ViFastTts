# Tài liệu tham khảo: Model TTS mã nguồn mở có tiếng Việt + Bộ dữ liệu giọng nói tiếng Việt công khai
**Phục vụ dự án ViFastTts** (TTS tiếng Việt 1 giọng nam Bắc, tự train acoustic text→mel, chạy ONNX CPU trên Mac).
**Ngày truy cập/tổng hợp: 15/10/2026.** Quy ước: `(chưa xác minh)` = chưa kiểm chứng trực tiếp, cần kiểm lại trước khi dùng để quyết định.

---

## Phần 1 — Model TTS mã nguồn mở có tiếng Việt

> Cột "Rõ chữ (VI)": ưu tiên bằng chứng WER/CER. Hầu hết model đa ngữ KHÔNG công bố WER tiếng Việt → ghi rõ.

| # | Model | Tổ chức/tác giả | Năm | Giấy phép (thương mại?) | Tiếng Việt (chất lượng / rõ chữ) | Clone giọng? | Cỡ model | Mac M-series / CPU? | Lấy ở đâu | Ghi chú thực dụng |
|---|-------|-----------------|-----|--------------------------|----------------------------------|---------------|----------|---------------------|-----------|-------------------|
| 1 | **VieNeu-TTS v3 Turbo** | Phạm Nguyễn Ngọc Bảo (pnnbao97) | 2025–2026 | **Apache-2.0 — thương mại OK** (gồm cả preset voice + audio sinh ra) | **Có, tốt** — 48 kHz, thiết kế riêng cho Việt + En-Vi code-switch (sea-g2p). Số WER/CER công bố: không thấy → (chưa xác minh) | **Có, zero-shot 3–5 s** | Nhẹ (bản ONNX CPU; Nano ~3× nhanh hơn Turbo) | https://github.com/pnnbao97/VieNeu-TTS · https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo | Đang dùng để sinh dữ liệu bù (RTF 0,12 trên Mac). v3 Nano (preview, 24 kHz, không clone) cho CPU yếu. **v4 đóng, không open-source** |
| 2 | VieNeu-TTS v1 / v2 | cùng trên | 2025–2026 | Apache-2.0 — thương mại OK | Có, khá–tốt; v2 train 10.000+ h song ngữ, podcast/dialogue mode | Có, 3–5 s | v1: 0.3B/0.5B + bản GGUF Q4/Q8 | Có (GGUF/ONNX CPU; GPU qua LMDeploy) | cùng repo trên · HF `pnnbao-ump/VieNeu-TTS-v2`, `VieNeu-TTS` | v2 nặng hơn, cần GPU để chạy mượt; v3 Turbo là bản nên dùng trên Mac |
| 3 | **VietTTS** | Đặng Văn Sâm (dangvansam) | 2025 | Code **Apache-2.0** nhưng **weights CC BY-NC — KHÔNG thương mại** (do train dữ liệu in-the-wild) | Có, khá (mượn code CosyVoice) | Có, prompt-audio clone | (chưa xác minh) | Linux/GPU chính; macOS "coming soon" | https://github.com/dangvansam/viet-tts · https://huggingface.co/dangvansam/viet-tts | API tương thích OpenAI. Chỉ dùng nghiên cứu/sinh dữ liệu nội bộ |
| 4 | **viXTTS** | thinhlpg (+ nhóm FPT) | 2024 | Kế thừa **CPML non-commercial** của XTTS-v2 — KHÔNG thương mại | Có, khá; fine-tune XTTS-v2 trên viVoice. Câu <10 từ kém (tác giả tự ghi) | Có, ~6 s (giống XTTS-v2) | ~1,9 GB checkpoint | CPU chạy được nhưng rất chậm; cần GPU ≥4 GB VRAM; **không tối ưu macOS** | https://huggingface.co/spaces/thinhlpg/vixtts-demo · https://github.com/hoangv97/vixtts | MOS naturalness/similarity hơn bản train trên BUD500 khoảng 0,3 điểm (paper viVoice). Demo Space tiện sinh mẫu nhanh |
| 5 | XTTS-v2 (gốc) | Coqui (đã đóng cửa 1/2024) | 2023–24 | **CPML — chỉ phi thương mại**, không còn nơi mua license | **Không chính thức** (17 ngôn ngữ, không có vi); ép đọc vi = tiếng Việt lơ lớ | Có, ~6 s, cross-lingual | ~1,8–1,9 GB | CPU chậm; GPU 4–6 GB VRAM | https://huggingface.co/coqui/XTTS-v2 | Chỉ làm nền cho viXTTS. Không dùng trực tiếp cho tiếng Việt |
| 6 | **F5-TTS** (gốc) | SWivid và cộng sự | 2024 | Code MIT; **weights CC-BY-NC — KHÔNG thương mại** | Không chính thức (chủ yếu en/zh) | Có, ~3 s ref | **0,3B** (Base) | PyTorch + GPU; CPU chậm | https://github.com/SWivid/F5-TTS | Seed-TTS test-hard CER 8,67% (bảng IndexTTS). Chỉ làm backbone cho các bản Việt hoá |
| 7 | **F5-TTS-Vietnamese** | nguyenthienhy | 2025 | Repo MIT; weights (chưa xác minh — nhiều khả năng NC theo gốc) | Có, khá–tốt (fine-tune ~1000 h) | Có (zero-shot theo kiến trúc F5) | 0,3B + vocoder | Cần GPU để train/infer mượt | https://github.com/nguyenthienhy/F5-TTS-Vietnamese · demo https://huggingface.co/spaces/hynt/F5-TTS-Vietnamese-100h | Có pipeline fine-tune công khai → **mẫu tốt để học cách pretrain→fine-tune giọng mình** |
| 8 | **viF5TTS / EraX-Smile-Female-F5** | EraX-AI | 2025 | (chưa xác minh) | Có (train 800k mẫu, ~500 h có private) | Có, zero-shot | 0,3B | GPU | https://github.com/EraX-AI/viF5TTS | Kèm FastAPI server. Cảnh báo: một phần dữ liệu private → không tái tạo được |
| 9 | nam194/F5-TTS-Vietnamese | nam194 (cộng đồng) | 2024–25 | Apache-2.0 (tag trên HF; gated, cần share contact) | Có (chưa xác minh chất lượng) | Theo F5 (~3 s) | 1,35 GB/checkpoint | GPU | https://huggingface.co/nam194/F5-TTS-Vietnamese | Checkpoint cộng đồng, ít tài liệu |
| 10 | **MMS-TTS-vie** | Meta AI (Pratap et al.) | 2023 | **CC-BY-NC-4.0 — KHÔNG thương mại** | Có nhưng **yếu** (1 speaker/checkpoint, 16 kHz, robotic, trọng âm/thanh điệu Việt kém) | Không | **36,3M** (~38 MB bản ONNX q8) | **Rất tốt trên Mac/CPU**, chạy cả trên trình duyệt (WASM) | https://huggingface.co/facebook/mms-tts-vie | Giá trị chính là **điểm khởi đầu pretrain/fine-tune** (recipe HF `suhaibrashid17/MMS-TTS`, pipeline `Scorpionzzzzz/vits-vietnamese` chạy được Win/Linux/Mac). Finetune từ checkpoint này chỉ cần 80–150 mẫu (~20 phút theo recipe) |
| 11 | **Piper giọng vi (vi_VN-vais1000-medium)** | rhasspy/M. Hansen (voice); engine Piper | 2022–23 | Voice files xem MODEL_CARD từng voice (thường permissive); **engine piper-tts là GPL-3.0** | Có, trung bình — **rõ chữ ở mức đủ nghe** nhưng máy móc (VITS đơn giọng, 22,05 kHz (chưa xác minh)) | Không | ~63 MB ONNX | **Rất tốt trên Mac** (có wheel arm64, realtime CPU) | https://huggingface.co/rhasspy/piper-voices/tree/main/vi · `pip install piper-tts` | Baseline "rõ chữ" rẻ nhất để so CER. **viPiper** (EraX-AI, MIT) là bản cải tiến pipeline train/prune Piper cho tiếng Việt: https://github.com/EraX-AI/viPiper |
| 12 | Coqui TTS (engine + YourTTS/VITS đa ngữ) | Coqui/厍cộng đồng | 2021–24 | Engine **MPL-2.0 — thương mại OK**; weights XTTS thì CPML | Giọng vi cộng đồng (chưa xác minh) | Có (YourTTS/XTTS) | Tùy checkpoint | CPU được, chậm với model lớn | https://github.com/coqui-ai/TTS | Công ty đã đóng cửa; chỉ dùng engine + model cộng đồng license rõ |
| 13 | **Kokoro-82M** | hexgrad | 2024–25 | **Apache-2.0 — thương mại OK** | **Gốc KHÔNG có vi** (en/es/fr/ja/zh/hi/it/pt) | Không (giọng cố định, ~50 voice) | **82M** (~320 MB fp32, ~90 MB q8) | **Rất tốt trên Mac** (kokoro-onnx, realtime M1) | https://huggingface.co/hexgrad/Kokoro-82M · https://github.com/thewh1teagle/kokoro-onnx | Không đọc Việt. Đáng chú ý vì **Kokoro-Vietnamese** (iamdinhthuan, 2026): fine-tune + vig2p G2P, có export ONNX → https://github.com/iamdinhthuan/Kokoro-Vietnamese (license/chất lượng: chưa xác minh) |
| 14 | **Zonos v0.1 / Zonos2** | Zyphra | 2025–26 | v1 Apache-2.0; **Zonos2 MIT** | Zonos2 Tier-2 có **Vietnamese** (chất lượng vi: chưa xác minh) | Có, high-fidelity clone | Zonos2 **8B MoE** | **Linux + NVIDIA GPU only — KHÔNG Mac** | https://github.com/Zyphra/Zonos2 | Chỉ tham khảo/sinh dữ liệu trên máy GPU thuê; không deploy Mac |
| 15 | **OpenVoice V1/V2** | MyShell + MIT | 2023–24 | **MIT — thương mại OK** | V2 native 6 ngôn ngữ **không gồm vi**, nhưng là **tone-color converter**: gắn lên bất kỳ base TTS nào (kể cả base Việt) để đổi màu giọng | Có, cross-lingual, chỉ cần clip ngắn | Nhỏ (converter) | CPU/Mac chạy được | https://github.com/myshell-ai/OpenVoice | Hữu ích theo hướng: base Việt rõ chữ (Piper/MMS) + OpenVoice đổi màu giọng. Chất lượng tổ hợp với tiếng Việt: chưa xác minh |
| 16 | **Chatterbox / Multilingual / Turbo** | Resemble AI | 2025 | **MIT — thương mại OK** | Multilingual 23 ngôn ngữ, **có vi hay không: chưa xác minh** | Có, ~5 s + emotion control + watermark | ~0,5B | Chủ yếu GPU; CPU chậm | https://github.com/resemble-ai/chatterbox · https://www.resemble.ai/learn/models/chatterbox | Nếu xác minh có vi → ứng viên sinh dữ liệu bù license sạch. Cần test CER trước |
| 17 | **GPT-SoVITS** | RVC-Boss | 2024 | **MIT (code + weights)** | Gốc en/zh/ja; bản Việt cộng đồng (chưa xác minh) | **Có, few-shot 1 phút** data là train được giọng mới | **167M**, 32 kHz | Cần GPU | https://github.com/RVC-Boss/GPT-SoVITS · mirror https://huggingface.co/ttsds/gpt-sovits | Điểm mạnh: **train giọng mới với rất ít dữ liệu** → phù hợp bài toán "thêm giờ cho chính giọng này" nếu có GPU |
| 18 | **Fish-Speech** | Fish Audio | 2024–25 | Code Apache-2.0; **weights CC-BY-NC-SA-4.0 — KHÔNG thương mại** | Chính thức 8 ngôn ngữ **không gồm vi**; CER/WER ~2% là đo trên tiếng Anh 5 phút | Có, 10–30 s | 0,5B (S1-mini) / 4B (S1) | Có GUI Mac nhưng inference nặng, GPU khuyến nghị | https://github.com/fishaudio/fish-speech | Chỉ tham khảo kiến trúc/kỹ thuật; không dùng weights cho sản phẩm |
| 19 | **IndexTTS / 1.5 / 2.x** | Bilibili | 2025–26 | **Bilibili Model Use License — KHÔNG phải OSI; có ngưỡng doanh thu** (bản EN ghi 1 tỉ RMB/năm, bản ZH ghi 100 triệu RMB — mâu thuẫn, đọc kỹ) | zh/en chính; **không vi** | Có | ~1,5B (v2) | GPU | https://github.com/index-tts/index-tts | WER tốt nhất nhóm GPT-style trên seed-test (test-en 1,6%). Không liên quan trực tiếp tiếng Việt |
| 20 | **MaskGCT** | Seed team (ByteDance) | 2024 | Research code (chưa xác minh thương mại) | Không vi chính thức | Có (3 s prompt) | Lớn | GPU | https://arxiv.org/abs/2409.00750 | Quan trọng về **kiến trúc**: fully **non-autoregressive**, không cần alignment text-speech — cùng họ bài toán với ViFastTts. Seed test-en WER 2,62% |
| 21 | **CosyVoice 2 / 3** | FunAudioLLM (Alibaba) | 2024–25 | **Apache-2.0 — thương mại OK** | 9+ ngôn ngữ, **vi chính thức: chưa xác minh (nhiều khả năng không)** | Có, ~3 s + cross-lingual + emotion | **0,5B** | GPU (có streaming/fastapi; CPU chậm) | https://github.com/FunAudioLLM/CosyVoice | Seed test-en WER 2,57%, test-hard CER 6,83%. VietTTS mượn code từ đây |
| 22 | **Spark-TTS** | (cộng đồng/SparkAudio) | 2025 | (chưa xác minh — nhiều nguồn ghi Apache-2.0, cần kiểm) | zh/en chính; không vi | Có | **0,5B** | GPU | tìm "Spark-TTS" trên GitHub/HF | Seed test-en WER 1,98%. Tham khảo kiến trúc |
| 23 | **Higgs Audio v2** | Boson AI | 2025 | (chưa xác minh) | Đa ngữ, vi: chưa xác minh | Có | **3B** | GPU | HF Space `multimodalart/higgs-audio-v3-tts` (demo) | Long-form tốt (Qwen3-TTS paper đo long-en WER chunk 6,9%) |
| 24 | **Qwen3-TTS** (0.6B/1.7B) | Alibaba Qwen | 2026 | **Apache-2.0 — thương mại OK** | 10 ngôn ngữ **KHÔNG gồm vi** | Có, 3 s clone + voice design | 0,6B / 1,7B | Cần CUDA + flash-attn — **không hợp Mac** | https://github.com/QwenLM/Qwen3-TTS | **WER tốt nhất hiện nay**: Seed test-zh 0,77 / test-en 1,24 (bản 1.7B-Base). Không đọc Việt → chỉ tham khảo kỹ thuật |
| 25 | **Bark (+ VietBark cộng đồng)** | Suno | 2023 | **MIT — thương mại OK** | 13 ngôn ngữ **không gồm vi**; bản fine-tune Việt cộng đồng (chưa xác minh) | Voice preset + history prompt (không clone đúng nghĩa) | ~0,9B (3 transformer) | CPU chạy được nhưng chậm (4–8 GB VRAM nếu GPU) | https://github.com/suno-ai/bark | Generative nên **hay bịa thêm tiếng cười/nhạc, WER không ổn định** → không hợp mục tiêu "rõ chữ" |
| 26 | Parler-TTS | Lyth & King (HF) | 2024 | Apache-2.0 (bản mini/large) (chưa xác minh lại từng checkpoint) | Chủ yếu en | Không (prompt mô tả giọng) | 0,9B (large) | GPU/CPU chậm | https://huggingface.co/parler-tts | Chỉ tham khảo điều khiển giọng bằng mô tả |
| 27 | VALL-E / VoiceCraft | Microsoft / Meta-FAIRakov | 2023–24 | Nghiên cứu, không license thương mại rõ | Dùng trong paper PhoAudiobook cho tiếng Việt (kết quả: tốt với câu ngắn) | Có | Lớn | GPU | paper arXiv:2506.01322 | Bằng chứng duy nhất dạng benchmark cho TTS tiếng Việt: **XTTS-v2 train trên PhoAudiobook thắng bản train trên viVoice mọi metric** |
| 28 | ElevenLabs (đóng) | ElevenLabs | — | **Đóng, trả phí theo ký tự** | Có vi, chất lượng cao (làm mốc so sánh) | Có (IVC/PVC) | — | Cloud API → gọi được từ Mac | https://elevenlabs.io | Chỉ để so sánh chất lượng và sinh mẫu đối chứngмата khi cần |

> Model Việt khác tìm thêm được: **EraX-Smile-UniSex-F5** (EraX-AI, unisex F5, chưa xác minh), **quangvu3/coqui-xtts** (thêm vi vào XTTS, cộng đồng, chưa xác minh), các **HF Space**: `thinhlpg/vixtts-demo`, `hynt/F5-TTS-Vietnamese-100h`, `multimodalart/higgs-audio-v3-tts`.

---

## Phần 2 — Bộ dữ liệu giọng nói tiếng Việt công khai

| # | Dataset | Tổ chức | Năm | Giờ | Người nói | Kiểu | Chất lượng | Giấy phép (thương mại?) | Lấy ở đâu | Ghi chú |
|---|---------|---------|-----|-----|-----------|------|------------|--------------------------|-----------|---------|
| 1 | **VLSP 2020 TTS** | VLSP + Vbee/HUST | 2020 | **~9,5 h** (7.770 câu) | **1 nữ, giọng Tây Nam Bộ** | Đọc | Studio sạch (chưa xác minh sample rate) | **Research-only** (ký thỏa thuận, phải tham gia chấm dữ liệu mới được tải) | https://vlsp.org.vn/vlsp2020/eval/tts | Đơn giọng nữ miền Nam — **không hợp giọng nam Bắc**, chỉ tham khảo pipeline |
| 2 | **VLSP 2021 TTS** | VLSP | 2021 | **~7,2 h** (5.341 câu; bản mở rộng ~9,7 h/7.743 câu), **44,1 kHz mono** | **1 nữ Hà Nội** (kênh Giang Ơi YouTube) | **Nói tự nhiên** (spontaneous) | In-the-wild, có nhiễu nền | Research-only | https://vlsp.org.vn/vlsp2021/eval/tts (paper JCSCE 2022) | Cùng miền "YouTube nói tự nhiên" với dự án → tham khảo xử lý spontaneous (FastSpeech2 + HiFi-GAN của team Viettel) |
| 3 | **VLSP 2020 ASR (VinBigData)** | VinBigData/VLSP | 2020 | **~250 h tổng, ~100 h public** (20 h đọc + 80 h tự phát) | Nhiều | Đọc + tự phát (news) | 16 kHz; transcript thủ công ~96% | **Research-only** (đăng ký participant) | https://vlsp.org.vn/vlsp2020/eval/asr · HF `doof-ferb/vlsp2020_vinai_100h` | 1 trong 2 benchmark ASR chuẩn (Test-T1). Dùng đo CER thì hợp, train TTS thì license vướng |
| 4 | **VLSP 2021 ASR** | VLSP | 2021 | **280 h + 400 h unlabeled** | Nhiều | Hỗn hợp | (chưa xác minh) | Research-only | https://vlsp.org.vn/vlsp2021 | — |
| 5 | **VLSP 2023 ASR** | VLSP | 2023 | (chưa xác minh) | Nhiều | Hỗn hợp + pseudo-label voting | Mixed | Research-only | có track `VLSP2023-voting-pseudo-labeled` (HF) | Tiền lệ kiểm soát chất lượng pseudo-label bằng voting đa ASR |
| 6 | **VLSP 2025 ASR/SER** | VLSP | 2025 | (chưa xác minh) | Nhiều | + emotion | (chưa xác minh) | Research-only | ZipFormer-30M-6000h đạt SOTA WER 7,97–8,1% public/private 2025 | Tham khảo chiến lược data của quán quân (6.000 h, OOV augmentation, Whisper refine) |
| 7 | **viVoice** | FPT Univ HCMC (Thinh Le et al.) | 2024 | **1.017 h, 887.772 mẫu** | Rất nhiều (186 kênh YouTube) | **Nói tự nhiên đa miền** | **24 kHz**, đã denoise/tách nhạc, lỗi transcript ước ~1,8% | **CC-BY-NC-SA-4.0 — KHÔNG thương mại**; gated (cần email trường/cơ quan) | https://huggingface.co/datasets/capleaf/viVoice · https://github.com/thinhlpg/viVoice | **ƯCV pretrain số 1**: nam ~61%, đa điều kiện thu âm (trị lệch miền). Nhược: pseudo-label, transcript ngắn (trung bình 4,1 s) |
| 8 | **PhoAudiobook** | VinAI/UK (Thị Vũ et al., ACL 2025) | 2025 | **941 h** (train 940 h/710 spk; +554 h mẫu ngắn augmented; test seen/unseen) | **735 người đọc sách nói** | **Audiobook (đọc diễn cảm)** | **16 kHz**, SI-SNR cao nhất trong các bộ vi, câu dài 10–20 s, test set soát tay | **Phi thương mại** (non-commercial); gated | https://huggingface.co/datasets/thivux/phoaudiobook (paper https://aclanthology.org/2025.acl-short.81/) | **ƯCV pretrain số 2**: SI-SNR cao nhất, câu dài hợp TTS; XTTS-v2+PhoAudiobook thắng XTTS-v2+viVoice mọi metric. Nguồn gốc sachnoiviet.net (23k h thô) |
| 9 | **VIVOS** | AILAB-VNUHCM | 2016 | **15 h** (train 46 spk/11.660 câu; test 19 spk) | 65 | Đọc studio | 16 kHz, sạch | **CC-BY-NC-SA-4.0 — KHÔNG thương mại** | https://huggingface.co/datasets/AILAB-VNUHCM/vivos | Benchmark smoke-test 10 phút; quá nhỏ để train |
| 10 | **Common Voice vi** (v27) | Mozilla | 2019–nay | **~23 h tổng / ~7,8 h validated / 419 spk** (bản v27) | 419 | Đọc crowdsourced (đa mic/phòng → **nhiều điều kiện thu**) | 32–48 kHz gốc; chất lượng lẫn lộn | **CC0 — THƯƠNG MẠI OK** (duy nhất license sạch trong nhóm) | https://mozilladatacollective.com/datasets/cmu5vci3w00agnq07hy4nbauv · HF `mozilla-foundation/common_voice_17_0` | Nhỏ nhưng license sạch + đa điều kiện thu → hợp trộn vào train chống lệch miền |
| 11 | **FLEURS vi** | Google | 2022 | **~12–13 h** (song song 102 ngôn ngữ, câu FLoRes) | Nhiều | Đọc | 16 kHz | **CC-BY-4.0 — THƯƠNG MẠI OK** | https://huggingface.co/datasets/google/fleurs (split vi) | Benchmark song song đa ngữ; nhỏ |
| 12 | **MLS** | Facebook/Meta | 2019 | **0 h tiếng Việt — KHÔNG có vi** (8 thứ tiếng Âu) | — | — | — | CC-BY | https://huggingface.co/datasets/facebook/multilingual_librispeech | Ghi rõ để khỏi mất công tìm |
| 13 | **BUD500 (VIET_BUD500)** | VietAI (non-profit) | 2024 | **~500 h** (634k train + 7,5k val + 7,5k test) | Nhiều, 3 miền Bắc/Trung/Nam | Tự phát: podcast/du lịch/sách/ẩm thực | 16 kHz, ~98 GB | **MÂU THUẪN: GitHub ghi Apache-2.0 nhưng HF ghi CC-BY-NC-SA-4.0** → coi như KHÔNG thương mại đến khi làm rõ | https://huggingface.co/datasets/linhtran92/viet_bud500 · https://github.com/apluka34/Bud500 | Đa miền Bắc-Trung-Nam → tốt cho chống lệch miền. viXTTS train trên BUD500 thua bản train viVoice 0,3 MOS |
| 14 | **VietMed** | Le-Duc et al. (LREC-COLING 2024) | 2024 | **16 h labeled + ~966 h unlabeled medical + ~1.200 h unlabeled general** | 61 (labeled) / 2.352 / 202 | Hội thoại y tế + general | **8–9 điều kiện thu âm**, 6 giọng vùng, metadata bệnh ICD-10/vai trò | (chưa xác minh — kiểm HF trước khi dùng) | https://huggingface.co/datasets/leduckhai/VietMed | **Nhiều điều kiện thu âm nhất** → quý cho bài toán lệch miền; unlabeled cần pseudo-label |
| 15 | **VietSpeech (VSV-1100)** | NhutP (cộng đồng) | 2024 | **~1.100 h, 1.026.047 mẫu** | Nhiều | Tự phát mạng xã hội, đa accent/style | 16 kHz, 131 GB | (chưa xác minh) | https://github.com/NhutP/VietSpeech (HF cùng tên) | Lớn thứ 2 sau GigaSpeech2-vi; chất lượng transcript (chưa xác minh) |
| 16 | **GigaSpeech 2 (vi)** | SpeechColab/DataoceanAI | 2024–25 | **refined 6.039 h / raw 7.324 h** (lớn nhất) | Rất nhiều (YouTube) | Đa miền general | 16 kHz, **pseudo-label** (NST lọc nhiều vòng); baseline ASR WER test 12,83% | (chưa xác minh — kiểm repo/HF trước khi dùng) | https://github.com/SpeechColab/GigaSpeech2 · HF/ModelScope | Chỉ nên pretrain/ASR; pseudo-label ồn → không fine-tune giọng trực tiếp |
| 17 | **LSVSC** | (VN) | (chưa xác minh) | **~100 h, ~57k câu** (theo VietPrism) | (chưa xác minh) | Hỗn hợp, human label | 16 kHz | (chưa xác minh) | tìm "LSVSC Vietnamese" trên HF (`LSVSC` trong VietAudio leaderboard) | Nằm trong top-9 bộ của Vietnamese ASR leaderboard |
| 18 | **ViMD** | (VN) | (chưa xác minh) | **~100 h, ~19k câu** | **~13k spk** (theo VietPrism — (chưa xác minh)) | Broadcast, đa phương ngữ | 32–48 kHz (theo leaderboard) | (chưa xác minh) | HF (trong `VietAudio-team/Vietnamese-asr-leaderboard`) | Bộ test đa phương ngữ chuẩn |
| 19 | **FOSD (FPT Open Speech)** | FPT | (chưa xác minh) | (chưa xác minh, ~7 h trong leaderboard sample) | Nhiều | Đọc | 16/48 kHz | (chưa xác minh) | HF (trong leaderboard) | — |
| 20 | **InfoRe 1 & 2** | InfoRe | (chưa xác minh) | (chưa xác minh) | — | Đọc | — | (chưa xác minh) | liệt kê trong awesome-vietnamese-speech-datasets | Cần tìm link gốc |
| 21 | **VAIS-1000 / VLSP-VAIS** | VAIS | ~2020 | (chưa xác minh; tên gợi ý ~1.000 câu) | 1+ | Đọc | (chưa xác minh; bản Piper 22,05 kHz) | (chưa xác minh) | giọng Piper `vi_VN-vais1000` là dẫn chứng tồn tại | Truy tìm bản gốc vì đã có voice Piper từ nó |
| 22 | **VinBigData 101 h** | VinGroup/VinAI | 2023 | **101 h** | Nhiều | General | 16 kHz, 229 wpm | (chưa xác minh) | trích trong paper PhoAudiobook Tab.1 | — |
| 23 | **VietnamCeleb** | Pham et al. | 2023 | **187 h, KHÔNG transcript** | Nhiều | Giải trí tự phát | 16 kHz | (chưa xác minh) | paper trích Tab.1 PhoAudiobook | Chỉ dùng speaker-pretrain, không train TTS |
| 24 | **VDSPEC** | (VN) | (chưa xác minh) | **~45 h, 150 spk, 3 miền** | 150 | (chưa xác minh) | (chưa xác minh) | (chưa xác minh) | trích trong VietPrism Tab.1 | Cân bằng 3 miền, cỡ vừa |
| 25 | **YODAS/YODAS2 (vi)** | (đa quốc gia) | 2024 | Số giờ vi (chưa xác minh) | Nhiều | YouTube đa ngữ | Mixed | **CC-BY-3.0** (theo catalogue — kiểm lại) | HF `espnet/yodas` | License tương đối thoáng nếu đúng CC-BY |
| 26 | **PhoST** | (EN-VI speech translation) | (chưa xác minh) | **508 h** | — | TED talks | Human label | (chưa xác minh) | catalogue awesome-vi-speech | Kênh EN-VI, ít liên quan TTS đơn ngữ |
| 27 | **VietPrism** | M. Hoang / Indiana Univ. | 2026 | **993 h bona fide (403.941 câu, 1.262 spk, 8.388 video) + 2.765 h spoof** | 1.262 | Đa miền + code-switch + deepfake pair | Transcript + metadata dialect/speaker | (chưa xác minh — paper mới 9/2026) | https://arxiv.org/html/2609.30005v1 | Mới, giàu metadata miền/phương ngữ (cả giọng hải ngoại); spoof từ 4 hệ TTS (kể cả thương mại) |
| 28 | **VietSuperSpeech / Nexdata 500 h conversational** | Viet/Nexdata.ai | 2022 | **500 h, 750 spk**, mobile phone | 750 | **Hội thoại mặt đối mặt tự nhiên** | 16 kHz wav, transcript thủ công ~95–96% | **Commercial license (trả phí)** | https://github.com/Nexdata-AI/500-Hours-Vietnamese-Conversational-Speech-Data-by-Mobile-Phone | Hội thoại tự nhiên đa mic điện thoại → rất hợp chống lệch miền nếu ngân sách cho phép |
| 29 | **ntt123/viet-tts-dataset** | ntt123 (cộng đồng) | (chưa xác minh) | **35,9 h** | Ít | Văn học (tổng hợp?) | Human label | Research-only | catalogue awesome-vi-speech | Cỡ tương đương dự án (17 h) → so sánh được |
| 30 | **CodeLinkIO vietnamese-voice-dataset** | CodeLinkIO | (chưa xác minh) | **~10 h (619 phút)** | **1 nữ miền Nam** | Đọc, studio | Thu chuyên nghiệp, môi trường kiểm soát | Research-only | https://github.com/CodeLinkIO/vietnamese-voice-dataset | Mẫu quy trình thu sạch đơn giọng để đối chiếu |
| 31 | **ViSpeech (gender/dialect)** | TranNguyenNB | 2024 | **~14 h, 10.686 file, 449 spk**, Bắc/Trung/Nam, nam+nữ | 449 | Không theo kịch bản | **44,1 kHz** + bộ test nhiễu riêng | (chưa xác minh — Drive công khai) | https://github.com/TranNguyenNB/ViSpeech | Có sẵn split **test nhiễu** → đo độ bền miền |
| 32 | **OpenBible / Kinh Thánh nói tiếng Việt** | (cộng đồng Bible) | — | Lớn (toàn bộ Tân Ước, (chưa xác minh) giờ) | 1–vài (đọc) | Đọc chậm, rõ | Sạch, câu ngắn | Public domain/thoáng (kiểm từng bản dịch) | tìm "Vietnamese OpenBible audio / MMS-lab vie" | Cùng họ dữ liệu MMS đã dùng; đơn giọng đọc rõ → hợp pretrain độ rõ chữ |
| 33 | **Voiz FM / Fonos (sách nói thương mại)** | DN Việt Nam | — | Rất lớn | Hàng trăm (giọng Bắc đủ) | Audiobook diễn cảm | Studio 44,1 kHz | **Bản quyền — KHÔNG dùng khi chưa mua** | voiz.vn, fonos.vn | Chỉ liệt kê để khỏi nhầm với sachnoiviet.net (nguồn PhoAudiobook là web công khai) |

---

## Phần 3 — Hướng dẫn chọn

### Q1. Muốn THÊM GIỜ cho CHÍNH giọng này (1 người, giọng Bắc) — cách nào khả thi?

| # | Lựa chọn | Vì sao | Cảnh báo |
|---|----------|--------|----------|
| A | **Thu thêm có kiểm soát**: cùng phòng + cùng mic + kịch bản phủ ô hiếm (câu chứa NGÃ+HỎI và tổ hợp NGÃ+coda m/n/ng...; đọc cả câu ngắn ≤10 từ) | Trị đúng gốc bệnh (ô hiếm + lệch miền); mỗi giờ thu sạch giá trị hơn nhiều giờ clone | Tốn công nhất; phải giữ nguyên setup thu (mic/vị trí/phòng) kẻo tự tạo thêm miền mới |
| B | **Clone chính giọng bằng VieNeu-TTS v3 Turbo từ clip 8 s** (đang dùng, RTF 0,12, Apache-2.0) để sinh câu phủ ô hiếm | Rẻ, nhanh, license sạch; đã chứng minh oversample giảm rè 13% | Giọng clone "con" không thêm thông tin giọng thật; mel mờ 63% của model sẽ di truyền → chỉ dùng cho **text-side augmentation** (câu hiếm), không thay giờ thật; kiểm CER từng batch sinh |
| C | **Few-shot train giọng mới bằng GPT-SoVITS (MIT, 1 phút data)** trên GPU thuê rồi sinh hàng loạt | Sinh được nhiều giờ "giọng mình" với ngữ điệu đa dạng hơn clone zero-shot | Cần GPU; bản Việt cộng đồng chưa xác minh; giọng sinh vẫn là xấp xỉ → trộn tỷ lệ thấp (10–20%) và đo CER holdout |
| D | **Tái khai thác 17,1 h hiện có**: cắt lại theo câu ngắn, oversample ô hiếm đã làm, thêm speed/room-augmentation (mục Q2) | Không tốn thu/sinh, hiệu quả đã đo được (giảm rè 13%) | Trần cải thiện thấp; không thêm phoneme mới thật sự |

### Q2. Muốn GIẢM LỆCH MIỀN (phòng/mic) — bộ nào / kỹ thuật nào giúp?

| # | Lựa chọn | Vì sao | Cảnh báo |
|---|----------|--------|----------|
| A | **Trộn dữ liệu đa điều kiện thu**: Common Voice vi (CC0, đa mic/phòng) + BUD500 (3 miền) + VietMed-U (8–9 điều kiện thu) vào pretrain/multitask | Buộc acoustic học đặc trưng bất biến miền; license CC0 của Common Voice sạch | Chỉ pretrain/phụ trợ — KHÔNG trộn giọng khác vào fine-tune giọng chính (sẽ pha giọng) |
| B | **Augmentation phòng/mic tổng hợp**: RIR (reverb phòng), noise phòng, EQ mic, codec (mp3/opus), gain — áp lên chính 17,1 h | Mô phỏng đúng "video khác ⇒ phòng/mic khác" mà không cần dữ liệu ngoài; rẻ, chạy được trên Mac | Đo chênh val−train sau mỗi biến (mục tiêu kéo từ +26% xuống); augmentation quá tay làm mất độ nét mel |
| C | **Chuẩn hoá miền ở feature**: CMVN theo clip, chuẩn hoá loudness (sox gain như PhoAudiobook), enhancement nhẹ trước khi trích mel | Rẻ nhất, làm ngay trong pipeline cache hiện có | Enhancement mạnh (như paper VLSP2021 đã ghi) có thể **giảm** chất lượng tổng hợp — test A/B |
| D | **Chia val theo miền** (val cùng video vs khác video tách riêng) + chọn checkpoint bằng val-khác-miền | Tránh tự lừa: checkpoint thắng val-cùng-miền thường thua khi ra video mới | Tốn công tổ chức lại split; nhưng là việc nên làm trước mọi thay đổi khác |

### Q3. Muốn PRETRAIN rồi FINE-TUNE giọng mình (thay vì train từ đầu) — nên dùng bộ nào + model nào?

| # | Lựa chọn | Vì sao | Cảnh báo |
|---|----------|--------|----------|
| A | **Pretrain acoustic ViFastTts trên PhoAudiobook-940 h (710 spk) → fine-tune 17,1 h giọng mình** | Cùng ngôn ngữ + câu dài 10–20 s + SI-SNR cao nhất; XTTS-v2 đã chứng minh PhoAudiobook > viVoice | Phi thương mại (gated); 16 kHz vs pipeline 24 kHz hiện tại → cần resample/thống nhất feature; pretrain đa giọng rồi fine-tune đơn giọng phải đóng băng/khởi tạo lại phần speaker |
| B | **Pretrain trên viVoice-1.017 h (24 kHz, khớp sample rate) → fine-tune** | Khớp 24 kHz pipeline, đa miền YouTube (gần miền dự án), nam 61% | Pseudo-label ồn (~1,8%), câu ngắn (4,1 s); NC + gated email trường; transcript ngắn kém dạy prosody câu dài |
| C | **Fine-tune từ MMS-TTS-vie (36M, recipe 20 phút/80–150 mẫu)** | Nhanh nhất để có baseline giọng-mình-chạy-được; chạy Mac/CPU | Chất lượng trần thấp (robotic), 16 kHz, NC; chỉ làm baseline so CER, không phải đích sản phẩm |
| D | **Bỏ pretrain ngoài, pretrain nội bộ**: gộp 17,1 h + dữ liệu clone VieNeu (giọng-mình-xấp-xỉ) pretrain đa-điều-kiện rồi fine-tune giờ thật | Không vướng license/gated, giọng không bị pha | Rủi ro "tự học cái sai của chính mình" (mel mờ di truyền); bắt buộc giữ split kiểm chứng giờ-thật-thuần |

### Q4. Model TTS nào cho ĐỘ RÕ CHỮ tiếng Việt cao nhất để sinh dữ liệu bù (kể cả model không clone được)?

| # | Lựa chọn | Vì sao | Cảnh báo |
|---|----------|--------|----------|
| A | **VieNeu-TTS v3 Turbo** (đang dùng) | Thiết kế riêng cho Việt (sea-g2p), 48 kHz, clone 3–5 s, Apache-2.0, RTF 0,12 Mac — tổng hòa rõ-chữ + tiện + license tốt nhất hiện tại | Không có WER công bố → tự đo CER bằng PhoWhisper/ASR trên mẫu sinh trước khi tin |
| B | **Piper vi_VN-vais1000 (VITS đơn giọng)** | Đơn giọng VITS thường **ít nuốt chữ** hơn model autoregressive lớn; ONNX 63 MB chạy realtime Mac; license voice thường thoáng | Máy móc, sai thanh điệu ở câu dài; chỉ hợp sinh câu ngắn phủ ô hiếm |
| C | **F5-TTS-Vietnamese / EraX-Smile-F5** (thử cả hai, đo CER đối đầu) | Họ F5 có tiếng rõ chữ tốt (Seed test-hard CER ~8,7% bản gốc); đã Việt hoá + clone | GPU mới chạy nổi; license weights chưa sạch → chỉ sinh dữ liệu nội bộ |
| D | **Chatterbox Multilingual** (sau khi xác minh có vi) | MIT + watermark sẵn (truy xuất nguồn dữ liệu sinh — điểm cộng quản trị dữ liệu) | Chưa xác minh vi; test CER trước, đừng tin quảng cáo "23 ngôn ngữ" |

> Nguyên tắc chung: **mọi audio sinh ra chỉ được vào train sau khi qua cổng CER** (ASR PhoWhisper/ZipFormer-30M-6000h) + kiểm tra thủ công mẫu NGÃ/HỎI — model sinh càng mượt càng dễ chèn lỗi thanh điệu mà nghe lướt không phát hiện.

---

## Phần 4 — Bảng "đáng thử ngay" (Top 5)

| # | Việc gì | Vì sao | Chi phí | Rủi ro |
|---|---------|--------|---------|--------|
| 1 | **Đo CER đối đầu 3 nguồn sinh bù** (VieNeu-v3-Turbo vs Piper-vi vs F5-TTS-Vietnamese) trên 200 câu phủ ô hiếm, bằng PhoWhisper-large | Chủ dự án ưu tiên rõ chữ; chọn đúng nguồn sinh trước khi đổ hàng nghìn câu vào train | ~1 ngày Mac (VieNeu+Piper local; F5 cần GPU thuê/Colab) | Thước ASR có bias giọng miền → dùng 2 ASR chéo nhau |
| 2 | **Augmentation phòng/mic + val-khác-miền** (RIR/noise/codec trên 17,1 h; tách val cùng/khác video) | Trị chênh val−train +26% mà không cần dữ liệu ngoài, không vướng license | Vài giờ code + retrain ngắn | Augmentation quá tay mờ mel thêm → đo val_rough kèm |
| 3 | **Xin quyền viVoice + tải PhoAudiobook, pretrain thử 1 vòng ngắn** rồi fine-tune 17,1 h | Hai bộ pretrain tốt nhất cho TTS Việt; trả lời dứt điểm "pretrain ngoài có hơn train từ đầu không" | Gated access (email cơ quan) + ~169 GB + GPU thuê pretrain | NC-license: weights pretrain dính NC → **sản phẩm cuối vẫn phải train lại hoặc xin phép**; coi đây là thí nghiệm |
| 4 | **Kéo Common Voice vi (CC0, ~23 h) + FLEURS vi (CC-BY, ~12 h) vào pipeline đo CER/thanh** (không cần vào train) | Hai bộ license sạch duy nhất → làm bộ đo chuẩn, tránh lệ thuộc bộ NC | Tải nhẹ (<5 GB) | Nhỏ, đa giọng → chỉ đo, không train giọng |
| 5 | **Baseline MMS-TTS-vie fine-tune giọng mình trong 1 buổi** (recipe 20 phút) | Có ngay mốc "giọng-mình-chạy-được" để mọi cải tiến ViFastTts phải thắng; chạy Mac/CPU | ~0,5 ngày | Trần thấp, dễ nản nếu so trực tiếp với base 40k — nhớ vai trò của nó là **mốc dưới**, không phải đích |

---

## Câu hỏi còn mở

1. WER/CER tiếng Việt thực tế của VieNeu-v3-Turbo, F5-TTS-Vietnamese, EraX-Smile, Chatterbox-multilingual trên cùng bộ câu (đặc biệt câu ≤10 từ + thanh NGÃ/HỎI)? — chưa model nào công bố.
2. License thật của weights F5-TTS-Vietnamese / EraX-Smile / Chatterbox-vi (nếu có) khi dùng audio sinh ra để train model thương mại?
3. Mâu thuẫn license BUD500 (GitHub Apache-2.0 vs HF CC-BY-NC-SA-4.0) — hỏi VietAI để chốt trước khi pretrain.
4. GigaSpeech2-vi, VietSpeech-1100h, LSVSC, ViMD, FOSD: license chính xác + chất lượng transcript đủ pretrain TTS không?
5. VietPrism (9/2026) đã public download chưa, license gì?
6. Kokoro-Vietnamese và quangvu3/coqui-xtts-vi: chất lượng + license weights?
7. VAIS-1000 gốc ở đâu (để có thêm dữ liệu đơn/ít giọng đọc sạch)?
8. Có bộ audiobook giọng **nam Bắc** đơn giọng, license thoáng nào ngoài PhoAudiobook không (quý nhất cho pretrain giữ miền Bắc)?
9. Ngưỡng trộn dữ liệu sinh (10%? 20%?) mà CER holdout bắt đầu xấu đi — phải đo trên chính pipeline.
10. Có nên mua bản quyền sách nói (Voiz/Fonos) hoặc đặt thu studio giọng chính chủ để có 3–5 h "vàng" không — bài toán chi phí/lợi ích chưa có số.

## Nguồn tổng hợp chính

- Catalogue: https://github.com/lienlt10/awesome-vietnamese-speech-datasets
- Paper PhoAudiobook (ACL 2025): https://aclanthology.org/2025.acl-short.81/ · HF: https://huggingface.co/datasets/thivux/phoaudiobook
- Paper viVoice/GigaSpeech2/VietMed/VietPrism/Qwen3-TTS: arXiv 2406.11546 · LREC-COLING 2024 (VietMed) · arXiv 2609.30005 · arXiv 2601.15621
- Model cards HF: `pnnbao-ump/VieNeu-TTS-v3-Turbo` · `facebook/mms-tts-vie` · `coqui/XTTS-v2` · `rhasspy/piper-voices` · `hexgrad/Kokoro-82M` · `suno/bark` · `Qwen/Qwen3-TTS-*`
- Repos: VieNeu-TTS · viet-tts · F5-TTS-Vietnamese · viF5TTS · Kokoro-Vietnamese · OpenVoice · Chatterbox · GPT-SoVITS · fish-speech · CosyVoice · Zonos2 · Qwen3-TTS
- Leaderboard ASR Việt: https://huggingface.co/datasets/VietAudio-team/Vietnamese-asr-leaderboard · ASR SOTA: `hynt/Zipformer-30M-RNNT-6000h` (VLSP 2025 quán quân)

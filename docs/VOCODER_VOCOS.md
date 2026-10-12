# Vocos thật + fine-tune (R7/R8) — quy trình chốt

> Quyết định: **thay generator hybrid (`ISTFTNetVocos`, upsample ×4) bằng Vocos thật**
> (`models/vocos.py`: ConvNeXt 512/1536/8, head iSTFT hop mel, **không upsample**) và
> **khớp biểu diễn mel với checkpoint MIT `charactr/vocos-mel-24khz`** trước khi tính lại
> stats/cache. Căn cứ đo đạc: R7-1 (mel/filterbank) và các benchmark trong tài liệu này.
> Cập nhật: 12/10/2026.

## 0. Vì sao đổi (số đo thật)

| Hạng mục | Hybrid trong repo | Vocos thật + fine-tune |
| --- | --- | --- |
| Nạp checkpoint pretrained | ✗ (khác shape: 384 vs 512, conv-head vs Linear-head) | ✓ 13,53M tham số, MIT |
| Zero-shot copy-synthesis (val50, tool `eval_vocoder`) | — | mel L1 **0,153** (thô) · F0-RMSE 85,7 cents (bỏ quãng tám) · VUV lỗi 4,3% |
| So vocoder lite 46k tự train | F0 116 cents · VUV 15,1% | ✓ tốt hơn ngay cả khi chưa fine-tune |
| Thời gian train | 100k ≈ 18–19,5h | **10k ≈ 1,8h** (0,653 s/bước MPS, b8·crop64·D ch16) |
| Suy luận ONNX CPU (5s audio) | RTF 0,0089 | **RTF 0,0031** (~2,9–3,3×) |
| ONNX export | 2 graph + iSTFT numpy | y hệt (real/imag + iSTFT numpy), metadata hop đúng |

Lộ trình (R8, cập nhật 10/2026 sau nghe thử trên @codelung): **tải về dùng ngay làm
bước 0** — bản zero-shot đã nghe "khá ổn" trên chính giọng, nên **fine-tune mel thật
5–10k trở thành TÙY CHỌN** (chỉ chạy khi muốn tối ưu timbre/nhiễu, hoặc khi bước 0 hụt ở
`eval_clips_v1`). **GTA 2–5k bước vẫn nên làm** sau khi acoustic train xong: lúc đó
vocoder nhận **mel dự đoán** (mượt/lệch) — phân bố mà checkpoint pretrained chưa từng
thấy; đây là rủi ro cao nhất nếu bỏ hẳn fine-tune. Fine-tune không thay được việc sửa
acoustic nếu acoustic làm mượt thanh điệu (xem §5).

## 1. Tải checkpoint (một lần)

```bash
python tools/fetch_vocos_pretrained.py        # → artifacts/pretrained/vocos-mel-24khz/
```

- Ghim revision HF `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21`, ghi `manifest.json`
  (sha256 từng file + license MIT). File `pytorch_model.bin` sha256 bản repo đã đo:
  `97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a`.
- Config trỏ `vocoder_training.pretrained_path` + `pretrained_sha256`; **train sẽ tự
  kiểm sha256** (lệch → báo lỗi) và ghi sha đó vào checkpoint fine-tune cạnh `mel_hash`.
- Dùng ngay không cần train: `synthesize`/`export-vocoder-onnx` nhận thẳng
  `artifacts/pretrained/.../pytorch_model.bin` (loader hiểu cả 3 định dạng: checkpoint
  repo `{vocoder: …}`, file HF thô, state dict trần).

## 2. Biểu diễn mel phải khớp checkpoint (R7-1)

| Thành phần | Repo trước đây | Checkpoint Vocos | Việc đã sửa |
| --- | --- | --- | --- |
| n_fft/win/hop/n_mels | 1024/1024/256/100 | y hệt | — |
| f_min/f_max | 0/12000 | 0/12000 (`None → sr//2`) | — |
| thang mel | HTK | HTK (torchaudio) | — |
| **norm filterbank** | **Slaney** `2/(f_{m+2}−f_m)` | **None (tam giác thô)** | `mel.norm: none` |
| **log** | `log(x + 1e-5)` | `log(clamp(x, 1e-7))` | `log_floor: 1e-7` + `log_floor_mode: clamp` |
| padding/cửa sổ | center/reflect/hann | y hệt | — |
| chuẩn hoá | z-score per-bin | không dùng | **khử z nội bộ trong generator** (`mel_mean/mel_std`) |

Số đo R7-1: lệch repo−checkpoint là **−3,03…−5,87 nats mỗi bin** (hằng số theo bin do
Slaney) + khác biệt floor ở frame yếu. Nếu giữ mel cũ, copy-synthesis bằng checkpoint lệch
**~4,3 nats** (khảo sát R7-1); sửa đúng thì **~0,15** (val50, đo bằng `tools/eval_vocoder.py`).
Sau khi sửa, mel repo **khớp torchaudio tới
5,6e-5** (test: `tests/test_features_representation.py::test_mel_kieu_vocos_khop_torchaudio`).

**Đổi các khoá mel trên = đổi `mel_hash` → tính lại stats + cache** (làm trước khi cache
v2; xem §3). Loss của vocoder sống ở **không gian thô** (bỏ `normalize_path`:
`VocosGenerator.loss_mel_config`), mel weight **45** đúng thang gốc; acoustic vẫn học
trong không gian z như cũ.

## 3. Quy trình chạy (sau khi có dữ liệu v2 + eval_clips_v1)

```bash
# 0) Tải checkpoint pretrained (nếu chưa) — §1

# 1) Stats mel v2 (theo dõi cảnh báo tỉ lệ bị kẹp ở floor; >1% bin cao → lọc frame có tiếng)
python tools/compute_mel_stats.py --manifest data/codelung/train.jsonl \
    --config configs/small_codelung.yaml --out data/codelung/mel_stats.json --sample 2000
#    (nếu cảnh báo: thêm --min-log-energy -6)

# 2) Cache feature (key tự đổi theo mel_hash)
vifasttts cache-features --manifest data/codelung/train.jsonl \
    --config configs/small_codelung.yaml --cache-dir cache/codelung --workers 8

# 3) Bước 0 (zero-shot) → đánh giá bằng CÙNG script cho mọi mốc
vifasttts train-vocoder --config configs/small_codelung.yaml \
    --manifest data/codelung/train.jsonl --output runs/vocoder_codelung_v3 --max-steps 0
python tools/eval_vocoder.py --config configs/small_codelung.yaml \
    --checkpoint runs/vocoder_codelung_v3/last.pt \
    --manifest data/codelung/eval_clips_v1.jsonl --output artifacts/vocoder_eval/step_0 --clips 50

# 4) Fine-tune mel thật (D warm-up 1k → GAN; 10k bước ≈ 1,8h MPS) — chạy theo khối như acoustic
python tools/train_vocoder_chunked.py --config configs/small_codelung.yaml \
    --manifest data/codelung/train.jsonl --output runs/vocoder_codelung_v3 \
    --total-steps 10000 --chunk 5000

# 5) Đánh giá mốc 1k/2k/5k/10k (wav + metrics, dùng CPU)
for s in 1000 2000 5000 10000; do
  python tools/eval_vocoder.py --config configs/small_codelung.yaml \
      --checkpoint runs/vocoder_codelung_v3/last.pt \
      --manifest data/codelung/eval_clips_v1.jsonl --output artifacts/vocoder_eval/step_$s --clips 50
done

# 6) Sau khi acoustic xong: GTA 2–5k bước trên mel teacher-forced (config riêng, lr thấp hơn)
```

Ghi chú fine-tune (config `vocoder_training` đã đặt sẵn):
`d_warmup_steps: 1000` (G đóng băng) · G lr **5e-5**, D lr **2e-4** · cosine 10k ·
`adv_loss: lsgan` + `mrd_weight: 0.1` (đúng recipe đã sinh checkpoint) · `crop_frames: 64`
(~0,68s) · D MPD 2/3/5/7/11 + MRD ch16 · `stft_weight: 0`. Chạy kiểu khối qua
`tools/train_vocoder_chunked.py` (resume + chống phình MPS); milestone nghe thử:
`tools/vocoder_milestones.py` (mel L1 giờ tính trong đúng không gian Vocos).

## 4. Cổng nghiệm thu

| Metric | Công cụ | Ngưỡng |
| --- | --- | --- |
| mel L1 (không gian Vocos) | `tools/eval_vocoder.py` | fine-tune **< bước 0** (đo trên eval_clips_v1, loudness-normalized) |
| log-F0 RMSE (cents, bỏ quãng tám) | idem (pyworld) | không tệ hơn bước 0; xem §5 cho thanh điệu |
| VUV lỗi | idem | idem |
| CER (Whisper-MLX) synth vs gốc | (khi có ASR) | chênh ≤ 1 điểm |
| Thanh hầu hoá ngã/nặng | `tools/voice_quality.py` | `hnr_creak_delta` ≤ ~2 dB (synth không "mượt hoá" rõ) |
| Tone acc từng thanh (classifier E3) | (khi có) | giảm ≤ 2 điểm so với audio gốc |

Nghe cặp tối thiểu (R8-1 bước 1, làm được ngay với bước 0):

```bash
python tools/copy_synthesis.py --config configs/small_codelung.yaml \
    --checkpoint artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin \
    --input-dir data/minimal_pairs --output artifacts/minimal_pairs/step_0
# → <stem>_orig.wav / <stem>_synth.wav + phonation.json (HNR creak Δ)
```

**Chạy thử sớm trên dữ liệu v1 (10/2026)** — trước khi dữ liệu v2 xong, đã chạy zero-shot
trên chính giọng @codelung (checkpoint pretrained, không fine-tune) bằng đúng tool này:

| Nhóm clip | mel L1 (thô) | F0-RMSE (bỏ quãng tám) | VUV lỗi | `hnr_creak_delta` |
| --- | --- | --- | --- | --- |
| val50 (4 video val) | 0,153 | 85,7 cents | 4,3% | +0,10 dB |
| general (40 clip train ngẫu nhiên) | 0,134 | 96,7 cents | 12,1% | +0,17 dB |
| ngã/nặng (20 clip ≥2 âm tiết NGA/NANG) | 0,138 | 96,2 cents | 9,0% | +0,23 dB |

- Không clipping; creak **không bị làm mượt** (Δ ≤ 0,23 dB, dưới ngưỡng 2 dB) — tín hiệu tốt cho R8-1.
- Nhóm train (lẫn tạp âm nền) có VUV lỗi cao hơn val — cần nghe để phân biệt "vocoder bỏ voicing"
  hay "tracker F0 bám nhầm tạp âm của bản gốc".
- File nghe A/B: `artifacts/vocoder_eval/preview_listen/` (10 cặp + README) và metrics đầy đủ ở
  `artifacts/vocoder_eval/preview_<nhóm>/{metrics.json,voice_quality.json}`.
- **Đính chính**: con số mel L1 0,243 nêu ở bản trước là từ script probe cũ có lỗi đếm frame
  (so đúng 1 frame); số đúng theo tool là ~0,13–0,17. F0/VUV không bị ảnh hưởng.
- Hai lỗi thật của code được phát hiện & sửa nhờ chạy thử này: `denormalize` broadcast sai
  `[T, M] × [1, 1, M] → [1, T, M]` (metric sai) và buffer khác device với mel (eval MPS).
  Cả hai đã có test hồi quy trong `tests/test_vocos_generator.py`.
- Baseline **chính thức** vẫn phải đo lại trên dữ liệu v2 + `eval_clips_v1` (như review yêu cầu).

## 5. Rủi ro còn lại & hướng xử lý

- **Creaky voice (thanh ngã, nặng)**: Vocos không có nguồn F0 tường minh; GAN có thể làm
  mượt chu kỳ không đều. Thứ tự xử lý: (1) fine-tune mel thật; (2) nếu chưa đạt — tăng tỉ
  lệ mẫu ngã/nặng trong fine-tune; (3) chỉ khi vẫn kém mới tính vocoder có nguồn F0
  (kiểu HiFTNet/NSF), chấp nhận chậm hơn. **Lỗi mượt thanh điệu do acoustic model thì
  vocoder không sửa được** — bật kênh `phonation` (HNR/tilt/glottalization, đang tắt)
  khi smoke small nếu ngã/nặng yếu trên TTS từ text.
- **fp16 cho vocoder**: đo trên checkpoint thật + clip thật — waveform lệch max ~7e-4
  (mean ~4e-5), nhờ phép kẹp `exp(mag) ≤ 1e2` giữ trong graph. Mặc định vẫn **fp32**
  (model 13,5M, RTF 0,003); nếu xuất fp16 phải chạy test so fp32 trước.
- **MPS**: mel loss/MR-STFT chạy trực tiếp trên MPS (đã bỏ vòng copy CPU); test parity
  `tests/test_mps_parity.py` khóa sai số ≤ ~5e-4 tương đối (đo thật ≤ 6e-5) trước khi tin.
- Hybrid (`generator: vocos`, `lite`) giữ làm fallback, không đầu tư thêm.

## 6. Ứng viên theo dõi (R9) — khảo sát 10/2026, không đổi kế hoạch

Nguyên tắc: ứng viên nào có checkpoint 24 kHz thì phải vượt cổng **trên eval_clips_v1**
bằng chính `tools/eval_vocoder.py` (mel L1, log-F0 RMSE, VUV, + CER khi có ASR, + cặp
tối thiểu mả/mã/mạ qua `tools/copy_synthesis.py`/`voice_quality.py`) **và** ONNX export
được, RTF CPU ≤ 3× Vocos (đo thật 0,0031). Bảng của paper không thay được số đo này.

**Bằng chứng công bố** (LibriTTS, † = dùng checkpoint công khai; CPU là giây/giây audio
— bảng của RNDVoC, máy của họ):

| Model | Tham số | MACs (G/5s) | CPU (×real-time) | PESQ | MCD | Pitch RMSE | V/UV F1 | UTMOS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Vocos†** | 13,5M | **6,35** | **0,0066 (152×)** | 3,615 | 3,105 | 35,58 | 0,948 | 3,566 |
| RNDVoC-shared† | 3,14M (ckpt 3,94M) | 37,2 | 0,0669 (15×) | **4,226** | **1,914** | **23,97** | **0,970** | 3,657 |
| RNDVoC-nonshared | 9,48M | 28,3 | 0,0770 (13×) | 4,085 | 2,251 | 27,04 | 0,961 | 3,607 |
| BigVGAN-112M† | 112M | 454 | 0,667 (1,5×) | 4,269 | 1,871 | 24,28 | 0,967 | 3,744 |
| HiFi-GAN-V1† | 14,0M | 166 | 0,167 (6×) | 3,056 | 4,284 | 52,53 | 0,921 | 3,511 |

Kết luận khảo sát:

- **Không có kiến trúc nào có checkpoint 24 kHz vừa NHANH hơn vừa TỐT hơn Vocos.** Họ
  "tốt hơn" (RNDVoC/DegVoC) nặng **~5,4–5,9× compute và ~10× thời gian CPU của Vocos**
  (vẫn real-time ~15×, tiệm cận BigVGAN-112M với ~3–4M tham số) → trượt tiêu chí
  "nhanh hơn ≤3×" nếu số paper giữ nguyên; vẫn nên đo trên M6 khi tới R9-1.
- **RNDVoC** (Range-Null decomposition, IJCAI 2025 + bản mở rộng): ứng viên DUY NHẤT có
  code MIT + checkpoint 24k/100 bin/2M bước trên Hugging Face (`AndongLi/RNDVoC`,
  3,94M tham số). Lưu ý: **mel của họ là librosa (Slaney scale + Slaney norm) + `log(clamp(x)·C)`**
  — KHÁC Vocos (HTK, norm=None), nên phải trích mel riêng khi đánh giá (lặp lại bài học
  R7-1); muốn dùng production là đổi mel lần nữa (stats/hash/acoustic). Deps cũ
  (torch 2.3+cu118, nnAudio, ptflops), ONNX chưa ai kiểm.
- **DegVoC** (AAAI 2026, 3,89M, cùng nhóm tác giả, `DegVoC` MIT): chưa có checkpoint
  công khai → không zero-shot được; compute ~45,6 G/5s (≈3× Vocos).
- **WaveNeXt 1** (Vocos bỏ iSTFT → head linear): checkpoint công khai chỉ 22 kHz/80 bin
  (không khớp); **WaveNeXt 2** (5/2026) mới có paper + bản reproduce chưa train xong
  (không weights); claim nhanh là so với HiFi-GAN/WaveFit, không so Vocos.
- **PhaseGAN** (500K tham số, ~1 GMAC, 9/2026), **SCNet** (subband condition +
  magnitude-aware phase loss), **EffVOC** (low-delay), **Harmonic Discriminator**
  (12/2025): mới có paper/demo, chưa có code + weights → watch list.
- **"Revisiting Vocos"** (7/2026): Vocos mạnh về magnitude, yếu về **phase**; lớp 1D conv
  cản trở dự đoán phase-difference → hướng nâng cấp *tại chỗ* (phase loss kiểu SCNet,
  inductive bias tốt hơn) đáng A/B trên fine-tune hiện tại trước khi đổi kiến trúc.
- **HiFTNet**: nguồn F0 tường minh giúp đường cao độ (sắc/huyền/hỏi) nhưng **có thể tệ
  hơn ở creak** của ngã/nặng, không drop-in (cần mạng F0 riêng). Chỉ cân nhắc nếu R8-1
  cho thấy Vocos mất **đường cao độ**.

Hành động R9-1 (chạy sau khi Vocos fine-tune xong): RNDVoC zero-shot trên eval_clips_v1
(mel theo cách của họ) → so F0/VUV/CER + nghe + đo RTF M6 + thử ONNX export. Chỉ cân nhắc
đổi nếu hơn rõ và tốc độ chấp nhận được (có thể nới gate cho batch offline; giữ Vocos
cho đường real-time).

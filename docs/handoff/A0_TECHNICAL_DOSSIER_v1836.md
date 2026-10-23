# A0 — Hồ sơ kỹ thuật để phân tích cải tiến (v18.36)

> **Vòng đọc, không phải vòng train.** Ngày 23/10/2026 · HEAD `0796e9f` · production = **P2**.
> Phạm vi: đọc source/config/artifact đã có; **không sửa code, không train, không cài package,
> không chấm audio mới**. Mọi số đo dưới đây lấy từ artifact/doc đã tồn tại; số nào tự tính từ
> manifest/checkpoint thì ghi rõ phép tính.
>
> Ký hiệu: **[CODE]** = đọc trực tiếp source · **[SỐ]** = đọc JSON/log/handoff đã có ·
> **[TÍNH]** = tự đếm/tổng hợp từ manifest hoặc checkpoint (không chấm audio) · **[CHƯA]** = chưa có.
>
> **Không chạy `pytest`/`ruff` trong vòng này** vì chỉ thêm một tệp Markdown; không có thay đổi code.
> Các lệnh Python đã dùng chỉ để đọc checkpoint/manifest và đếm tham số/thống kê dữ liệu.

---

## 0. Tóm tắt sự thật đã kiểm chứng

1. **Production hiện tại là P2**: `configs/production_v1831.json` trỏ
   `artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt`, config `configs/small_codelung_v32.yaml`.
   Checkpoint này **không phải** `runs/adv_p2_20k/ckpt_2000.pt` nguyên bản: `model` của nó **trùng khít
   từng tensor** với `ema` của `runs/adv_p2_20k/ckpt_2000.pt` (max diff = 0,0). Nói cách khác,
   **P2 production = EMA của P2@2k**, không phải trọng số “live”.
   ⚠️ `docs/handoff/TRANG_THAI.md:9-10` vẫn ghi “Production acoustic: G1 … rollback P2” —
   **dòng này đã cũ**; config production + commit `984d8cf` (rollback P2) + yêu cầu vòng này
   xác nhận P2. Không dùng dòng TRANG_THAI để suy ra production.
   ⚠️ Trong `configs/production_v1831.json`, trường `note` cũng còn câu “Đổi sang G1 theo A0 v18.31 …”;
   trường này **đã cũ**. Các trường có hiệu lực: `label` (“giữ P2”), `acoustic_checkpoint` (P2),
   `previous_candidate` (G1).
2. **Acoustic là 7,045,071 tham số** (7,05M): encoder 3,128,064 · decoder 1,564,032 · postnet 1,240,164 ·
   pause 333,890 · duration 333,697 · còn lại ~0,44M. **Vocoder Vocos 13,531,650** (13,53M).
   Tổng inference ~20,58M. ONNX hiện có: acoustic ~31,6 MB, vocoder ~54,2 MB.
3. **Mel contract**: 24 kHz · n_fft/win 1024 · hop 256 · 100 bin · f_min 0 · f_max 12.000 ·
   thang HTK · filterbank **norm=none** · **log-magnitude** · floor `1e-7` kiểu **clamp** ·
   `center=True` · z-score per-bin bằng `data/codelung/mel_stats.json`.
4. **Acoustic xuất mel z-score**; Vocos tự khử z bằng `mel * std + mean` rồi iSTFT. Không có bước
   đổi power/magnitude nào khác giữa hai model.
5. **Decoder CÓ nhận kênh vuv dự đoán** (logit) qua `variance_proj`, cùng f0/energy/phonation —
   không phải “không có kênh vuv” như ghi trong `A0_V18_GROUP1.md §8`. Nhưng vuv là **cấp segment**
   (một giá trị cho cả slot), **không có vuv ground-truth/frame-level**. Đây là đính chính quan trọng.
6. **P2/G1/G2 dùng `tools/finetune_adv.py`**, không dùng adversarial của `train_v2.py`.
   Chỉ train **decoder + mel + postnet** (2,82M/7,05M); **mọi thứ còn lại** (encoder, duration, pause,
   embedding, slot_expander, ratio_mlp, variance_mlp, g_token, projection) đóng băng.
7. **Acoustic không có gradient xuyên vocoder**: loss hoàn toàn trên mel z; D là **mel discriminator**;
   vocoder không được khởi tạo trong vòng fine-tune. Vocoder train riêng (GAN waveform) và hiện **pretrained
   zero-shot** được chọn làm production.
8. **P2/G1 không khôi phục optimizer, scheduler, discriminator hay RNG**: `finetune_adv.py` chỉ nạp
   `model`; AdamW mới, D random mới, step reset 0, không scheduler (LR hằng), không lưu D.
9. **Bất đối xứng chọn checkpoint**: P2 được chấm bằng **EMA**, G1/G2/g1_1k được chấm bằng **live**
   (`Synthesizer` đọc `state["model"]`). G1 live vs EMA lệch tới 0,008 (max abs). Đây là confound
   chưa được xử lý trong các kết luận v18.33/v18.35.
10. **Đường eval free-run v18 đã làm mất dấu câu**: `tools/v1833_render_eval.py` dựng text bằng
    `" ".join(raw)` và **không nối `punctuation`**, rồi gọi `iter_synthesize_sentence`. Vì `parse_text`
    chỉ tạo gap từ dấu câu, `gaps` truyền vào model **toàn 0**; eval free-run **không có pause**; đo được
    1,69% pause so với 28,68% của audio thật. Đây là lỗi đường đo, không phải bằng chứng model mất pause.
    Lưu ý: trong sentence mode, `gaps` **luôn được truyền** nên `gate_threshold=1,01` không phải nguyên
    nhân — nguyên nhân là text mất dấu câu.
11. **Vocoder trong artifact ONNX không nhất quán với “pretrained production”**: A0 lượt 10 ghi
    `v32_adv_p2_2k_vocoder.onnx` dựng từ `runs/vocoder_vocos_stftc/last.pt` — chính là E4-C đã **TRƯỢT CỔNG**
    (§45.4), trong khi kết luận production là Vocos pretrained zero-shot. Không có `mel_hash` trong metadata
    ONNX để kiểm tra.
12. **Dữ liệu v32**: 18.226 segment train / 3.378 val / 2.611 holdout; 13/12/9 video; **không trùng video**
    giữa train-val-holdout. Tổng ~21,57 h segment (16,06 + 3,03 + 2,48). Đây là số giờ **sau khi cắt segment**,
    không phải 51 h của corpus gốc.
13. **Chọn val theo video, không theo câu**: 12 video ít giờ nhất = val (cap 150 clip/video), phần dư = holdout;
    13 video còn lại = train. Mọi bộ eval v18.31–v18.35 đều lấy từ **val** (300/36/110 câu), không lấy train;
    holdout hầu như chỉ dùng cho CER.
14. **Rè tập trung ở slot C**: Q2 (300 TF) cho buzz C-mũi **67,7%** (P2) / **72,7%** (GAN) và C-tắc
    **41,6% / 44,3%**; flat_V C-tắc P2 +69…96% so copy-synth. Đây là chỗ cần ưu tiên, không phải toàn câu.
15. **Không có “C5” định nghĩa thành văn trong repo cho vòng V18**. C5 có định nghĩa là ràng buộc
    kiến trúc ONNX ở `docs/TTS_ARCHITECTURE_RESEARCH.md §1.1` (và một C5 khác, không liên quan, trong
    `SYLLABLE_SYSTEMS_RESEARCH.md`). Trong handoff V18, “C5” được nhắc như **bộ 30 câu người thật
    chưa đủ điểm** — xem §6.1.

---

## 1. Kiến trúc và đường inference thực tế

### 1.1 Sơ đồ text → WAV

```text
text
 └─ frontend/pipeline.parse_text()                  # normalizer → token → ParsedSyllable
     ├─ apply_custom_reading()                      # data/codelung/custom_reading.json
     ├─ normalize_text(dialect)                     # số/ngày/viết tắt/English fallback
     ├─ safe_parse_syllable()                       # orthography + abstract + status
     └─ resolve_oov_token() (nếu resolve_oov)       # acronym/English/tên chữ cái
 └─ ComponentVocabs.encode()                        # [1,N,7] int64
 └─ ViFastTtsModelV2.forward()                      # models/v2_slot5.py
     ├─ ComponentEmbedding: 7 bảng × 64 → concat 448 → Linear 192
     ├─ Encoder ConformerLite ×4 (192, 4 head, ffn 768, conv k15)
     ├─ duration ConvPredictor → log_d [1,N] → d_syl [1,N]
     ├─ ratio MLP → [1,N,4] (O/M/N/C)
     ├─ pause ConvPredictor → gate [1,N], log_gap [1,N]
     ├─ distribute_slots() (nếu không có seg_durations) → d_seg [1,N,4]
     ├─ slot_durations = [G_lead, (O,M,N,C,G) × N] → [1,1+5N]
     ├─ LengthRegulator → expanded [1,T,192]
     ├─ variance head (segment MLP, 9 đầu ra) → s_k[4], vuv, energy, phonation[3]
     ├─ f0 contour = f0_log_mean + f0_log_std × Σ s_k·basis   → [1,T] log-F0
     ├─ conditioning = expanded + variance_proj(f0,vuv,energy,phonation) + position_proj(pos)
     ├─ Decoder ConformerLite ×2 → mel_pre [1,T,100]
     ├─ Postnet 5 conv residual → mel [1,T,100] (z-score)
     └─ VocosGenerator: denormalize → ConvNeXt 512/1536/8 → Linear 1026
         → (magnitude, phase) [1,513,T] → iSTFT n_fft 1024/hop 256 → waveform
```

Nguồn: `src/vifasttts/frontend/pipeline.py:40-95`; `src/vifasttts/models/v2_slot5.py:56-350`;
`src/vifasttts/models/conformer.py:87-128`; `src/vifasttts/models/vocos.py:117-192`;
`src/vifasttts/infer.py:283-328`.

### 1.2 Module nào thực sự tồn tại / không có

| Có trong đường chạy | Không có / không dùng |
|---|---|
| `ComponentEmbedding` 7 bảng (onset 24, medial 2, nucleus 15, coda 11, tone 6, boundary 4, punctuation 6 = **68 ID**) | Không có text encoder ký tự/BPE; input là âm tiết có cấu trúc |
| `ConformerLite` encoder 4 block + decoder 2 block | Không có autoregressive decoder / attention tự hồi quy |
| `ConvPredictor` duration, pause | Không có **module F0 predictor riêng**; F0 là 4 hệ số DCT từ `variance_mlp` |
| `SegmentMLP` cho slot_expander / ratio / variance | Không có **module energy predictor riêng**; energy là một đầu ra của cùng MLP |
| `LengthRegulator` + `slot_positions` | Không có frame-level phonation/vuv predictor; chỉ cấp segment |
| `mel` Linear + `postnet` 5 conv | Không có speaker/style/dialect embedding (đúng C1) |
| `tone_classifier` (6 lớp) | `tone_logits` **không dùng** lúc inference; chỉ là loss phụ |
| Vocos ConvNeXt + iSTFT head | Không có diffusion/flow/VAE/sampling (đúng C3/C4) |
| `MultiScaleMelDiscriminator` (train-only) | Không có waveform discriminator trong acoustic training |
| `MultiPeriodMultiResolutionDiscriminator` (vocoder train-only) | Không có vocoder trong graph acoustic / checkpoint P2 |
| `gate_bias` (nếu `gate_conditioning: true`) | v32 **không bật**; `gate_bias` không tồn tại trong checkpoint |
| `cross_slot_context` (nếu bật) | v32 **không bật** |
| `variance_override` (tham số forward) | Không được expose qua `Synthesizer`/CLI; chỉ dùng trong probe |

### 1.3 Tham số thật

**[TÍNH]** từ `state_dict` của checkpoint P2 (`torch.load(... )["model"]`):

| Khối | Tham số | % |
|---|---:|---:|
| encoder | 3.128.064 | 44,4% |
| decoder | 1.564.032 | 22,2% |
| postnet | 1.240.164 | 17,6% |
| pause | 333.890 | 4,7% |
| duration | 333.697 | 4,7% |
| slot_expander | 126.528 | 1,8% |
| variance_mlp | 91.209 | 1,3% |
| embedding | 90.560 | 1,3% |
| ratio_mlp | 89.665 | 1,3% |
| g_token | 24.768 | 0,35% |
| mel | 19.300 | 0,27% |
| variance_proj | 1.344 | 0,02% |
| tone_classifier + role_emb + position_proj + silence_emb | ~1.700 | 0,02% |
| **Tổng acoustic** | **7.045.071** | 100% |

Vocoder: **13.531.650** tham số (`VocosGenerator`). Discriminator mel (train-only): **715.587**.
Tổng inference = **20.576.721**.

### 1.4 Shape, đơn vị, miền giá trị tại các điểm chuyển tiếp

| Điểm | Shape | Đơn vị / miền | Nguồn |
|---|---|---|---|
| `features` | `[B,N,7]` int64 | ID rời rạc; N = số âm tiết | `v2_slot5.py:213` |
| `x` sau encoder | `[B,N,192]` | hidden | `conformer.py:123-128` |
| `log_duration` | `[B,N]` | log frame; d_syl = `round(exp(x)−1).clamp_min(1)` | `v2_slot5.py:223-234` |
| `durations` | `[B,N]` long | frame mel (1 frame = 10,67 ms) | idem |
| `ratio` | `[B,N,4]` | softmax có mask active; tổng = 1 trên slot active | `v2_slot5.py:216-220` |
| `seg_durations` / `d_seg` | `[B,N,4]` long | frame; thứ tự **O,M,N,C** | `v2_slot5.py:237-240` |
| `slot_durations` | `[B,1+5N]` long | frame; phần tử 0 = `G_lead`, rồi 5 slot/âm tiết | `v2_slot5.py:254-259` |
| `expanded` | `[B,T,192]` | T = Σ slot_durations | `v2_slot5.py:268` |
| `f0_frames` | `[B,T]` | **log-F0 tự nhiên** (log Hz); = 0 ở slot G; **> 0 cả ở frame vô thanh không thuộc G** vì lấy từ contour | `v2_slot5.py:289-290` |
| `vuv` trong decoder | `[B,T]` | **logit dự đoán cấp segment**, không phải xác suất; hằng trong slot | `v2_slot5.py:301-309` |
| `energy` trong decoder | `[B,T]` | log-power thô (mean ≈ −1,03; std ≈ 1,77) | `v2_slot5.py:302-305`; `f0_stats_v2.json` |
| `phonation` | `[B,T,3]` | 3 kênh thô (hnr/glottalization/jitter) | `v2_slot5.py:306-309` |
| `conditioning` | `[B,T,192]` | hidden + variance + position | `v2_slot5.py:313` |
| `mel_pre` / `mel` | `[B,T,100]` | **z-score per-bin**; không clamp ở đầu ra | `v2_slot5.py:318-321` |
| `tone_logits` | `[B,N,6]` | logits; không dùng inference | `v2_slot5.py:332` |
| `pause_gate` | `[B,N]` | sigmoid 0…1 | `v2_slot5.py:244-250` |
| `pause_log_gap` | `[B,N]` | log1p(frame gap); speed chia sau khi gate chọn | idem |
| Vocoder input | `[B,T,100]` | z-score; `denormalize` → log-magnitude thô | `vocos.py:160-171` |
| Vocoder spectrum | `[B,513,T]` | magnitude = `exp(head).clamp(max=1e2)`, phase rad | `vocos.py:102-107` |
| Waveform | `[B,(T−1)·256]` | float; iSTFT center=True; **không phải T·256** | `vocos.py:109-114`; đo bằng `torch.istft` |

Ghi chú quan trọng:
- **Không dùng `f0 > 0` làm mask hữu thanh** (F4): f0 đã nội suy qua vô thanh; phải dùng `vuv > 0,5`.
  `features.py:242-246` giữ `vuv` gốc, chỉ nội suy `f0`.
- **Không có clamp 70–400 Hz ở đầu ra model**. 70–400 chỉ là giới hạn của pyworld lúc trích feature.
  `v2_slot5.py:289-290` không clamp.
- **Nguồn mel z**: `data/codelung/mel_stats.json` có `min_log_energy: -6.0` — thực tế là **frame có
  năng lượng > −6**, không phải “mọi frame”; xem §2.6.

### 1.5 Đường production vs validation/TF vs benchmark

| Đường | Hàm / file | Duration/gap | Gate | Pause | Trim | Hậu xử lý | LUFS | α |
|---|---|---|---|---|---|---|---|---|
| **PyTorch chunk (CLI mặc định)** | `Synthesizer.synthesize` → `iter_synthesize` (`infer.py:330-397`) | model tự dự đoán | 0,5 | `pause_seconds(chunk)` giữa các chunk theo dấu câu | không | `postprocess` fade 10/20 ms; `peak_normalize` tùy chọn | CLI mặc định **−16**; library `None` | 0 |
| **PyTorch sentence (đường eval v18)** | `iter_synthesize_sentence` (`infer.py:449-504`) | model tự dự đoán; `gaps` từ dấu câu truyền vào model | **1,01** (không có tác dụng vì `gaps` luôn được truyền) | gap trong graph + `fill_gap_silence` | `_trim_edge_silence` −45 dB | eval gọi trực tiếp: **không fade/LUFS** | không | 0 |
| **PyTorch steady** | `iter_synthesize_steady` (`infer.py:399-447`) | model tự dự đoán | 1,01 | cố định 0,25/0,55/0,08 s | có | `postprocess` + CLI LUFS | CLI −16 | 0 |
| **TF eval** | `tools/eval_acoustic.py`, `v1831_tf`, `v183_fuse_probe` | **GT durations/seg/gaps/lead** | không dùng gate | GT gaps | không | không | không | 0 |
| **Free-run eval v18** | `tools/v1833_render_eval.py:62` → `iter_synthesize_sentence` | model tự dự đoán | 1,01 | **text bị mất dấu câu ⇒ gaps = 0** | có trim | không | không | 0 |
| **ONNX** | `runtime/onnx_engine.synthesize_onnx` (`:213-305`) | model tự dự đoán; v2 nhận `speed`/`gate_threshold` | 0,5 (hoặc 1,01 nếu `--steady`) | chỉ pause giữa chunk; **không sentence-gap** | không | fade + peak/LUFS | CLI −16 | không có α |

**Hệ quả cần nhớ**: “production” trong các kết luận v18.31–v18.35 là **`iter_synthesize_sentence`
gọi trực tiếp**, không phải CLI mặc định (CLI mặc định là chunk mode `iter_synthesize`). Hai đường
khác nhau ở gate, pause, trim và hậu xử lý. Xem §4.8 và §8 H1.

### 1.6 Toàn bộ hậu xử lý

| Hậu xử lý | Mặc định | Code | Ghi chú |
|---|---|---|---|
| `sharpen_mel` (sh25) | **α = 0** (không đổi) | `runtime/postprocess.py:44-58` | α=1,5 từng được dùng trong demo; §38.2: tăng α **khuếch đại rè** |
| Fade in/out | 10 ms / 20 ms | `postprocess.py:21-33` | tuyến tính, tránh “cụp” |
| Peak normalize | tắt; nếu bật −3 dBFS | `postprocess.py:11-18`, `infer.py:393` | chỉ khi `--peak-normalize` |
| LUFS cả file | CLI mặc định **−16**; library `None` | `cli.py:354-355`, `infer.py:45-52` | ITU-R BS.1770-4 + limiter true-peak −1 dBTP |
| Chunk loudness 6d | tắt | `infer.py:258-281` | dịch log-mel về mức hữu thanh mục tiêu, ±4 dB |
| F0 scale 8b | 1,0 (tắt) | `infer.py:291-299` | k > 1 giãn biên độ quanh mean câu; đo thực tế gần như không đổi audio |
| Duration clamp | tắt | `infer.py:187-231` | kẹp 1–2 âm tiết đầu vào p90 corpus; “rồi” 42 → 30 frame |
| Gap silence fill | bật ở sentence | `infer.py:233-256` | ghi `silence_mel_v2.json` vào frame gap |
| Trim edge silence | chỉ steady/sentence | `infer.py:69-87` | ngưỡng −45 dB, margin 2 frame (20 ms) |
| Ghép câu | `chunk_text(max_chars=400)` | `frontend/chunking.py` | pause theo dấu câu; sentence mode đọc cả chunk |
| Clipping PCM | `np.clip(-1,1)` | `infer.py:30-42` | PCM16 |
| ONNX | fade + peak/LUFS; **không** α, sentence-gap, chunk-loudness, f0-scale, duration-clamp | `onnx_engine.py:252-255` | đường ONNX nghèo hơn PyTorch |

---

## 2. Hợp đồng mel giữa acoustic và vocoder

### 2.1 Tạo mel target

`configs/small_codelung_v32.yaml:23-38`:

```yaml
mel:
  n_fft: 1024
  win_length: 1024
  hop_length: 256
  n_mels: 100
  f_min: 0
  f_max: 12000
  norm: none
  log_magnitude: true
  log_floor: 1e-7
  log_floor_mode: clamp
  normalize_path: data/codelung/mel_stats.json
```

`data/features.py:155-195`:

```python
window = torch.hann_window(win, device=wav.device)
spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win,
                  window=window, return_complex=True, center=True)
magnitude = spec.abs()
power = magnitude ** 2
bank = mel_filterbank(..., norm=mel_config.get("norm", "slaney"))
magnitude_mel = torch.einsum("mc,bct->btm", bank, magnitude)
if log_floor_mode == "clamp":
    mel = torch.log(magnitude_mel.clamp_min(floor))
...
energy = torch.log(power.mean(dim=1) + 1e-6)
mel = (mel - mean) / std
```

Hợp đồng thực tế:

| Mục | Giá trị | Nguồn |
|---|---|---|
| Sample rate | 24.000 Hz | config |
| FFT / win | 1024 / 1024 | config |
| Hop | 256 (10,667 ms) | config |
| Window | Hann (`torch.hann_window(1024)`, periodic) | `features.py:172` |
| Center / padding | `center=True`; `torch.stft` mặc định `pad_mode='reflect'` | `features.py:173` |
| Số frame | `T = 1 + len(wav)//256` | `features.py` docstring; đo bằng `torch.stft` |
| Mel bins | 100 | config |
| f_min / f_max | 0 / 12.000 Hz | config |
| Mel scale | HTK `2595·log10(1+f/700)` | `features.py:19-26` |
| Filterbank norm | **none** (tam giác thô) | config `norm: none`; `features.py:29-57` |
| Đại lượng | **magnitude** cho mel; **power** cho energy | `features.py:174-190` |
| Log | log tự nhiên; floor `1e-7` kiểu clamp | config |
| Chuẩn hoá | `(mel − mean)/std` per-bin | `features.py:191-195` |
| Stats | `data/codelung/mel_stats.json`: 2.000 clip, 1.501.457 frame, `min_log_energy=-6`, clamp 0% | JSON |
| mean range | −4,013 … +1,080 | **[TÍNH]** từ JSON |
| std range | 0,844 … 1,647 | **[TÍNH]** từ JSON |

### 2.2 Acoustic output và inverse normalization

- `v2_slot5.py:318-321`: `mel_pre = self.mel(decoded)`; nếu postnet: `mel = mel_pre + postnet(mel_pre)`.
- Không có denorm trong acoustic; **mel xuất ra là z-score**.
- `vocos.py:160-171`:
  ```python
  def denormalize(self, mel):
      if not self.mel_normalized:
          return mel
      shape = (1,) * (mel.dim() - 1) + (self.mel_mean.numel(),)
      mean = self.mel_mean.reshape(shape).to(mel.device)
      std = self.mel_std.reshape(shape).to(mel.device)
      return mel * std + mean
  ```
- `VocosGenerator.__init__` nhận `stats_path` từ `mel.normalize_path` (`istftnet.py:222-230`), đăng ký
  buffer `mel_mean`/`mel_std`; buffer này **bake được vào ONNX**.
- Không có clamp/min/max sau denorm; chỉ head clamp `exp(magnitude) ≤ 1e2` (`vocos.py:102-107`).

### 2.3 Vocoder input và checkpoint

| Mục | Giá trị | Nguồn |
|---|---|---|
| Kiến trúc | Vocos thật: ConvNeXt 512/1536/8 + head iSTFT hop mel | `models/vocos.py:27-29`, `:117-158` |
| Input | mel `[B,T,100]` **z-score** | `vocos.py:188-192` |
| Output | waveform `[B,(T−1)·256]` | `vocos.py:109-114`; đo `torch.istft` |
| Checkpoint nguồn | `charactr/vocos-mel-24khz`, MIT, revision `0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21` | `docs/VOCODER_VOCOS.md §1` |
| File | `artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin` | config `vocoder_training.pretrained_path` |
| SHA256 | `97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a` | config + doc |
| Tham số | 13.531.650 | **[TÍNH]** |
| Config kỳ vọng | `hidden 512`, `intermediate_dim 1536`, `generator_blocks 8`, `generator vocos_pretrained` | `small_codelung_v32.yaml:108-141` |
| Vocoder dùng trong eval v18 | **pretrained zero-shot** (`Synthesizer(cfg, path)` không truyền `vocoder_checkpoint`) | `tools/v1835_screen.py:124`; `infer.py:168-185` |
| Vocoder trong artifact ONNX | `runs/vocoder_vocos_stftc/last.pt` = **E4-C đã trượt cổng** | `A0_TURN10_RESULT_21_10.md:70-72`; `ACOUSTIC_V2.md §45.4` |

### 2.4 Có bước chuyển đổi mel nào giữa acoustic và vocoder không?

**Không**, ngoài affine z→raw bên trong Vocos. Không có:
- đổi power ↔ magnitude,
- đổi thang mel ↔ linear,
- resample,
- clamp biên độ,
- chuẩn hoá lại theo câu.

`sharpen_mel` là tuỳ chọn **trong z-space**, mặc định α=0 nên là identity. ONNX vocoder graph đã bake
`denormalize`, nên input ONNX vẫn là z.

### 2.5 Đã kiểm chứng khớp / chưa kiểm chứng

| Mục | Trạng thái | Bằng chứng |
|---|---|---|
| Mel repo khớp torchaudio `MelSpectrogram` (norm none, clamp 1e-7) | **ĐÃ** | `tests/test_features_representation.py::test_mel_kieu_vocos_khop_torchaudio`, sai số 5,6e-5 (doc) |
| `mel_hash` P2 khớp config/stats | **ĐÃ** | P2 `mel_hash = e065776b45b74b874c483dfdc68c1f9c5e45918b`; **[TÍNH]** `mel_config_hash(canonical_mel_config(v32))` cho cùng hash |
| `denormalize` là nghịch đảo của `(mel−mean)/std` | **ĐÃ** | `tests/test_vocos_generator.py::test_vocos_denormalize_and_loss_config` |
| Copy-synth mel L1 trong z | **ĐÃ** | `v1827_domain.json`: L1(model mel, mel thật z) = 0,992; copy L1 = 0,133 |
| ONNX acoustic khớp PyTorch (mel) | **ĐÃ** | `tests/test_onnx_runtime.py` (doc) |
| ONNX metadata có `n_mels=100`, sr 24k, n_fft 1024, hop 256 | **ĐÃ** | đọc metadata 3 file ONNX |
| ONNX lưu `mel_hash` để chống lệch stats | **CHƯA** | `export.py:177-194`, `:320-331` chỉ ghi sr/n_fft/hop/n_mels/arch/f0; **không có mel_hash** |
| ONNX vocoder hiện có dùng đúng Vocos pretrained | **CHƯA** | A0 lượt 10 ghi dùng `runs/vocoder_vocos_stftc/last.pt` (E4-C) |
| Waveform ONNX khớp bit PyTorch | **CHƯA** | chỉ có test mel/spec; chưa có so waveform end-to-end |
| Copy-synth không làm đổi độ dài | **CHƯA** | iSTFT trả `(T−1)·256`, ngắn hơn WAV gốc tới 255 mẫu; các eval phải tự cắt/thẳng hàng |

### 2.6 Sai lệch tài liệu trong hợp đồng mel

- `A0_V18_GROUP1.md §6 K2` ghi “**Mel stats P2 = mel_stats.json (mọi frame)**”. Thực tế
  `mel_stats.json` có `min_log_energy: -6.0` — chỉ frame có năng lượng > −6, **không phải mọi frame**.
  `mel_stats_v35_speech.json` mới là bản lọc thêm `vuv>=0.5` (`frames` 556.230 vs 1.501.457).
  Kết luận “không so mel z trực tiếp P2 vs v3.5” vẫn đúng, nhưng lý do phải ghi là **khác population
  (energy-gate vs vuv-gate)**, không phải “mọi frame vs hữu thanh”.
- `dataset_v2.py:32` docstring ghi `mel [T,20]`; thực tế **100 bin** (`config.mel.n_mels=100`).
- `A0_TURN10` ghi ONNX vocoder 52 MB; file thật 54.158.240 byte (~54,2 MB thập phân / 51,7 MiB).

---

## 3. Training P2 và các nhánh GAN

### 3.1 Config nền / P2 / G1 / G2

**Acoustic nền** = `configs/small_codelung_v32.yaml` (đã dẫn ở §2.1). Các block quyết định:

```yaml
model:
  component_dim: 64
  hidden_size: 192
  encoder_layers: 4
  decoder_layers: 2
  attention_heads: 4
  ffn_size: 768
  conv_kernel_size: 15
  dropout: 0.1
  duration_layers: 3
  variance_layers: 3
  use_phonation: true
  use_postnet: true

training:
  batch_size: 8
  learning_rate: 0.0002
  weight_decay: 0.0001
  max_steps: 60000
  phase_a_steps: 2000
  phase_a_lr_scale: 0.25
  phase_b_warmup_steps: 2000
  lr_schedule: cosine
  min_lr_ratio: 0.05
  lr_decay_steps: 60000
  early_stop_patience: 0
  grad_clip: 1.0
  mixed_precision: false

loss_weights:
  mel: 1.0
  mel_pre: 0.2
  mel_sharp: 1.0
  duration: 1.0
  ratio: 1.0
  f0: 1.0
  f0_delta: 0.5
  f0_prior: 0.1
  vuv: 0.5
  energy: 0.5
  phonation: 0.5
  tone: 0.1
  pause_gate: 0.5
  pause_gap: 0.5

adversarial:          # đường train_v2, KHÔNG dùng cho P2/G1/G2
  enabled: false
  d_warmup_steps: 1000
  mode: bce
  d_learning_rate: 0.0001
  ramp_steps: 2000
  spectral_norm: true
  mel_adv_weight: 1.0
  feature_matching_weight: 1.0
  scales: [1, 2, 4]
  channels: 64
  layers: 3
```

**P2 / G1 / G2** không có YAML riêng; chúng là các lần chạy `tools/finetune_adv.py` với cùng
`configs/small_codelung_v32.yaml`, khác nhau ở checkpoint khởi tạo và `--fm-weight`:

| Run | Checkpoint khởi tạo | Steps | `adv_weight` | `fm_weight` | `d_warmup` | `g_lr` / `d_lr` | Ghi chú |
|---|---|---|---:|---:|---:|---|---|
| P2 `runs/adv_p2_20k` | `runs/acoustic_v32/best.pt` (step 59k) | 20.000 (log dừng 7.600) | 0,3 | 0 | 300 | 2e-5 / 1e-4 | fixed, no ramp, EMA 0,999 |
| P2 4k/6k | tiếp P2 live | 4.000/6.000 | 0,3 | 0 | 300 | idem | cùng run, optimizer tiếp tục |
| G1 `runs/v1830_g1` | P2 **EMA** `_ckpt/step_02000.pt` | 2.000 | 0,3 | 0 | 300 | idem | D mới, optimizer mới |
| G1b `runs/v1830_g1b` | G1 **live** `ckpt_2000.pt` | 2.000 | 0,3 | 0 | 300 | idem | bão hoà |
| G2 `runs/v1830_g2` | P2 **EMA** `_ckpt/step_02000.pt` | 2.000 | 0,3 | **1,0** | 300 | idem | feature matching |

Nguồn: `runs/adv_p2_20k/run.json`, `runs/v1830_g1/run.json`, `runs/v1830_g2/run.json`,
`runs/v1830_g1b/run.json`. Các args còn lại: `batch_size 8` · `device mps` · `seed 42` ·
`ema 0,999` · `log_every` 100 (P2) / 200 (G1/G2) · `val_every` 1000 (P2) / 500 (G1/G2) ·
`save_every` 2000 (P2) / 1000 (G1/G2) · `val_batches 10` · `hard_stops` true (P2) / false (G1/G2).

### 3.2 Source thực hiện generator/discriminator update

**Đường P2/G1/G2: `tools/finetune_adv.py`.**

```python
TRAIN_PREFIXES = ("decoder.", "mel.", "postnet.", "gate_bias.")   # :48
for name, param in model.named_parameters():
    param.requires_grad_(name.startswith(TRAIN_PREFIXES))          # :217-218
...
opt_g = AdamW(trainable, lr=args.g_lr, betas=(0.5, 0.9), weight_decay=0.0)  # :243
opt_d = AdamW(discriminator.parameters(), lr=args.d_lr,
              betas=(0.5, 0.9), weight_decay=0.0)                  # :244
...
real_c, fake_c = sample_crops(moved["mel"], outputs["mel"], gap, rng)      # :297
real_out = discriminator(real_c)
fake_out = discriminator(fake_c.detach())
d_loss = hinge_d(real_out, fake_out)                                 # :301-304
...
if step > args.d_warmup:                                              # :315
    g_adv = hinge_g(discriminator(fake_c))                            # :319
    total = rec + eff * g_adv + ramp * args.fm_weight * g_fm ...      # :345
```

- **D update**: mỗi bước, kể cả trong 300 bước warm-up; `fake_c.detach()`.
- **G update**: chỉ sau `step > d_warmup`; D bị `requires_grad_(False)` trong lúc tính G.
- **Tỉ lệ**: 1 D : 1 G sau warm-up; 1 D : 0 G trong 300 bước đầu.
- **Không ramp** (`ramp_steps=0` → `ramp=1` ngay sau warm-up).
- **Không lưu D**: `torch.save` chỉ ghi `model`, `ema`, `step`, `ema_decay`, `arch` (`:385-391`);
  không có `d_last.pt` cho các run này.

**Đường `train_v2.py` (T4.1) — chỉ dùng khi `--adversarial`:**
- D = `MultiScaleMelDiscriminator`; loss BCE; D update mỗi bước từ `adv_start`; G sau `d_warmup`;
  ramp tuyến tính `ramp_steps`; `d_last.pt` có lưu D + optimizer D (`train_v2.py:451-485`, `:556-559`).
- Config v32 đặt `adversarial.enabled: false`, nên P2/G1/G2 **không đi đường này**.
- Lịch sử: 3 lần phân kỳ ở probe T4.1 (`ACOUSTIC_V2.md §46.4`); đã bỏ, thay bằng `mel_sharp`.

### 3.3 Tất cả loss đang hoạt động

#### a) Loss nền của acoustic (`v2_losses.slot_losses`)

| Loss | Công thức / reduction | Mask | Trọng số v32 |
|---|---|---|---:|
| `mel` | L1 `abs(pred−real)`, mean qua bin + frame | `frame_lengths`; **không loại frame lặng trong row** (trừ khi `mel_silence_mask`) | 1,0 |
| `mel_pre` | như trên trên `mel_pre` | idem | 0,2 |
| `mel_sharp` | mean của `abs(Δtime_pred − Δtime_real)` và `abs(Δfreq_pred − Δfreq_real)` | cặp frame hợp lệ + `frame_lengths` | 1,0 |
| `duration` | MSE(`log_duration`, `log1p(durations)`) | không (mọi N) | 1,0 |
| `ratio` | CE soft target `seg_ratio` trên `log_softmax` có mask inactive | `syllable_valid` + active | 1,0 |
| `f0` | L1 `abs(seg_shape − target_z)` | `f0_weight × f0_k_mask` | 1,0 |
| `f0_delta` | L1 của đạo hàm contour tại pos 0,25/0,5/0,75 | `f0_weight` | 0,5 |
| `f0_prior` | MSE shape nucleus với prior theo thanh | `nucleus_weight` | 0,1 |
| `tone_disc` | CE phân biệt thanh từ khoảng cách tới prior | nucleus + k_mask | **0,0** (tắt) |
| `vuv` | BCEWithLogits trên `seg_vuv_logits` vs `seg_vuv≥0,5` | `active` (segment) | 0,5 |
| `energy` | L1 `seg_energy` vs target | `active` | 0,5 |
| `phonation` | L1 3 kênh | `seg_phonation_mask` | 0,5 |
| `tone` | CE `tone_logits` vs tone ID | không | 0,1 |
| `pause_gate` | BCEWithLogits, `pos_weight` + trọng số khối lượng gap | `pause_mask` | 0,5 |
| `pause_gap` | MSE `pause_log_gap` vs `log1p(gap)` | chỉ vị trí `gate=1` | 0,5 |

Nguồn: `v2_losses.py:22-27`, `:150-276`. Các chi tiết:
- `masked_l1` chia cho **số phần tử không padding** (`losses.py:13-29`); padding không vào loss.
- `mel_sharp` chia riêng cho `pair_mask.sum()` và `mask_float.sum()` (`v2_losses.py:192-195`).
- `pause_gate` dùng `PAUSE_POS_WEIGHT=1,0` (hành vi cũ) và `PAUSE_MASS_SCALE=10`, clip 40 frame;
  config v32 **không** đặt `pause_pos_weight`, nên P2 dùng 1,0 (`v2_losses.py:36-40`).
- `f0` target `s_k = DCT4(z)/√n`; mask Addendum 6: N luôn (weight = voiced ratio), M/C nếu ratio ≥0,5,
  O chỉ nếu onset ∈ `F0_VOICED_ONSETS` và ratio ≥0,5 (`v2_targets.py:136-190`).

#### b) Loss của P2/G1/G2 (`finetune_adv.py`)

`weights = {k: 0.0 for k in DEFAULT_WEIGHTS}` rồi `update({"mel":1.0,"mel_pre":0.2,"mel_sharp":1.0})`
(`finetune_adv.py:247-248`). Vì `slot_losses` vẫn tính mọi thành phần rồi nhân 0, **chỉ 3 loss tái tạo
có hiệu lực**; cộng thêm:

```text
G: L_rec + 0,3 · hinge_G(fake) + fm_weight · L_fm
D: hinge_D(real, fake.detach())
hinge_D = mean(relu(1−D(real))) + mean(relu(1+D(fake)))
hinge_G = −mean(D(fake))
```

`fm_weight = 0` cho P2/G1; `1,0` cho G2. `L_rec = mel 1,0 + mel_pre 0,2 + mel_sharp 1,0`.
Không có waveform loss, không có MR-STFT trong acoustic, không có energy/lowband (P2/G1/G2).

### 3.4 Loss tính trên không gian nào?

| Loss | Không gian |
|---|---|
| `mel`, `mel_pre`, `mel_sharp` | **mel z-score** per-bin (không phải log thô, không phải dB) |
| `adv`, `fm` (acoustic) | **mel z-score** |
| `f0`, `f0_delta`, `f0_prior` | hệ số DCT của log-F0 **đã z-score** |
| `energy` | log-power thô (target `energy` từ feature, mean −1,03) |
| `duration` | log frame |
| `pause_gate/gap` | xác suất / log1p frame |
| Vocoder GAN (train riêng) | waveform + mel **thô** (Vocos loss_mel_config bỏ normalize_path) |

**Không có loss nào trên waveform trong acoustic.** `gradient` không đi xuyên vocoder.

### 3.5 Discriminator nhìn gì, crop/padding/mask

- Input: **mel z-score** `[B,T,100]`; real = `moved["mel"]`, fake = `outputs["mel"]`.
- Kiến trúc: 3 nhánh scale 1/2/4; mỗi nhánh `avg_pool1d` thời gian (nếu scale>1) rồi 3 conv
  `kernel 5, stride 2`, channel 64→128→256, spectral norm, head conv → 1 (`discriminators.py:305-373`).
- Crop: `sample_crops` lấy **64 frame**; thử 8 vị trí ngẫu nhiên, ưu tiên cửa sổ có **≥80% frame
  không thuộc gap**; nếu không đạt, lấy vị trí tốt nhất (`finetune_adv.py:63-82`).
- Real/fake dùng **cùng start**.
- **Không dùng `frame_lengths`** trong `sample_crops`; batch được pad 0 tới `Tmax`
  (`dataset_v2.py:215-238`). Vì `SameNBatchSampler` chỉ gộp theo **N**, không theo T, các row ngắn hơn
  có thể có cửa sổ crop rơi vào vùng padding. Đây là **rủi ro kỹ thuật chưa đo** (xem §3.8).
- Không có mask nào khác trong D; D không thấy text, F0, energy, vuv, waveform.

### 3.6 Optimizer / LR / scheduler / clipping / precision / freeze

| Mục | P2/G1/G2 (`finetune_adv`) | Ghi chú |
|---|---|---|
| Optimizer G | AdamW, `lr=2e-5`, `betas=(0.5,0.9)`, `weight_decay=0` | `:243` |
| Optimizer D | AdamW, `lr=1e-4`, `betas=(0.5,0.9)`, `weight_decay=0` | `:244` |
| Scheduler | **không có** (LR hằng) | không cosine |
| Grad clip G | 1,0 | `:350` |
| Grad clip D | 1,0 (hard-code) | `:306`; config `grad_clip_d 5.0` **không dùng** ở đây |
| Precision | fp32; không AMP/autocast | config `mixed_precision: false`; script không autocast |
| Freeze | mọi thứ ngoài `decoder.`, `mel.`, `postnet.`, `gate_bias.` | `:48`, `:217-218` |
| Trainable | 2,82M / 7,05M | decoder 1,564 + mel 0,019 + postnet 1,240 |
| EMA | 0,999, khởi tạo từ trọng số nạp | `:246`, `:357-360` |
| D accuracy log | `(D(real)>D(fake)).mean()` trung bình scale | `:99-104` |
| `hard-stops` | chỉ khi bật; val +10%, D<0,05 500 bước, NaN | P2 bật, G1/G2 tắt |

### 3.7 Khi khởi tạo từ P2: khôi phục gì?

**Chỉ trọng số model.** `finetune_adv.py:206-215`:

```python
state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model = build_model(...)
load_model_state(model, state.get("model", state))
if hasattr(model, "load_f0_stats") and state.get("f0_stats"):
    model.load_f0_stats(dict(state["f0_stats"]))
```

Sau đó:
- `step = 0` (`:265`) — **không** cộng step cũ.
- Optimizer G/D **mới** (`:243-244`).
- D **random init** (`:238-241`).
- EMA **mới** từ trọng số nạp (`:246`).
- RNG: seed `random/np/torch` (`:187-189`), nhưng **không lưu/khôi phục RNG state**.
- Không có scheduler để khôi phục.
- Không có `d_last.pt` cho các run `finetune_adv`.

**Khác với `train_v2.py` resume**: `train_v2.py` khôi phục `optimizer`, `step`, `best_val`,
`no_improve`, `val_manifest_hash`, và tính LR theo `lr_at(step)` (`:326-355`, `:409-418`). Vì vậy
“P2-run tiếp tục” (P2 4k/6k) và “nhánh G1” là **hai chế độ khác nhau**, không chỉ khác trọng số loss.

### 3.8 Log loss/gradient đầu–giữa–cuối

#### P2 (`runs/adv_p2_20k/train_log.csv`; `rec = mel + 0.2·mel_pre + 1.0·mel_sharp`)

| Step | rec | mel | mel_pre | mel_sharp | adv | grad_ratio | d_loss | d_acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1,7775 | 0,9737 | 0,9891 | 0,6059 | 0,0000 | nan | 1,2025 | 1,0000 |
| 500 | 1,2472 | 0,5598 | 0,5502 | 0,5773 | 0,8114 | 2,8117 | 1,4113 | 0,9479 |
| 1.000 | 1,5501 | 0,7656 | 0,8074 | 0,6230 | 1,1255 | 2,2377 | 0,5951 | 0,9896 |
| 2.000 | 1,6185 | 0,8303 | 0,8364 | 0,6210 | 0,2238 | 1,5309 | 1,1528 | 0,9479 |
| 4.000 | 1,6538 | 0,8425 | 0,8035 | 0,6506 | 0,9196 | 2,8069 | 0,7685 | 0,9792 |
| 6.000 | 1,4574 | 0,7273 | 0,7231 | 0,5855 | 1,1770 | 1,4856 | 0,9636 | 0,9583 |
| 7.600 | 1,4948 | 0,7321 | 0,7037 | 0,6219 | 0,9459 | 4,6080 | 0,7468 | 0,9844 |

- `grad_ratio` = ‖grad adv trên decoder‖ / ‖grad rec trên decoder‖ (`finetune_adv.py:327-337`).
  Giá trị 1,5–4,6 ⇒ **gradient adversarial trên decoder lớn hơn gradient tái tạo 1,5–4,6 lần**,
  dù trọng số chỉ 0,3. Đây là lý do cần theo dõi, không chỉ nhìn `rec`.
- P2 run **không có `last.pt`**; log dừng ở 7.600, không có ghi chú lý do trong `run.json`.
  Checkpoint còn: 2k/4k/6k. Production dùng 2k EMA.

#### G1 (`runs/v1830_g1/train_log.csv`)

| Step | rec | mel | mel_pre | mel_sharp | adv | grad_ratio | d_loss | d_acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 1,3885 | 0,6539 | 0,8278 | 0,5691 | 0,0000 | nan | 1,6129 | 0,7188 |
| 400 | 1,7318 | 0,9605 | 0,7864 | 0,6141 | 0,4898 | 0,4365 | 1,1852 | 0,9115 |
| 600 | 1,4065 | 0,6938 | 0,6867 | 0,5754 | 0,5461 | 5,5247 | 1,1510 | 0,9219 |
| 1.000 | 1,4988 | 0,7372 | 0,8025 | 0,6011 | 0,9353 | 4,9560 | 1,1585 | 0,9167 |
| 2.000 | 1,6020 | 0,8219 | 0,8400 | 0,6121 | 0,4263 | 1,3903 | 1,1718 | 0,8958 |

#### G2 (`runs/v1830_g2/train_log.csv`; thêm `fm`)

| Step | rec | mel | mel_pre | mel_sharp | adv | fm | grad_ratio | d_loss | d_acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 1,3885 | 0,6539 | 0,8278 | 0,5691 | 0,0000 | 0,0000 | nan | 1,6129 | 0,7188 |
| 400 | 1,7317 | 0,9608 | 0,7862 | 0,6136 | 0,4873 | 0,2989 | 0,4288 | 1,1780 | 0,9115 |
| 1.000 | 1,5022 | 0,7403 | 0,8034 | 0,6012 | 0,9375 | 0,2511 | 4,8070 | 1,0850 | 0,9479 |
| 2.000 | 1,6132 | 0,8296 | 0,8418 | 0,6152 | 0,3876 | 0,2510 | 1,4225 | 1,0834 | 0,8958 |

#### Base v32 (`runs/acoustic_v32/history.csv`, step 59k — checkpoint P2 khởi tạo)

Các thành phần thô (chưa nhân trọng số) tại step 59.000:

```text
mel 0,7152 · mel_pre 0,7661 · mel_sharp 0,5650 · duration 0,0768 · ratio 0,8910
f0 0,4851 · f0_delta 5,5655 · f0_prior 0,3085 · tone_disc 1,6057 (weight 0)
vuv 0,0730 · energy 0,6628 · phonation 0,4562 · tone 0,00079
pause_gate 0,1819 · pause_gap 0,0818
val_mel 1,5128 · val_mel_pre 1,5680 · val_mel_sharp 0,6660 · val_f0_delta 6,7384
```

`f0_delta` là số hạng lớn nhất sau trọng số (5,5655 × 0,5 = **2,78**), gấp ~4 lần mel (0,715);
đây là vấn đề cân bằng loss đã ghi ở `AGENT0_INFO_21_10.md §A2/B5`, chưa sửa trong P2 (P2 đặt
`f0_delta` weight = 0 vì chỉ dùng 3 loss tái tạo).

### 3.9 Khác biệt thực tế giữa “P2-run tiếp tục” và “nhánh G1”

| Điểm | P2 4k/6k | G1 1k/2k |
|---|---|---|
| Điểm xuất phát | P2 **live** @2k (cùng run, optimizer tiếp tục) | P2 **EMA** @2k (checkpoint production), optimizer **mới** |
| Optimizer state | Adam moments tiếp tục | Adam moments = 0 |
| Step / LR | step 4k/6k, LR hằng 2e-5 | step reset 1/2k, LR hằng 2e-5 |
| D | D của P2 run tiếp tục (cùng D) | D **random mới** |
| EMA | EMA của P2 run | EMA **mới** từ P2 |
| Dữ liệu / seed | cùng manifest, seed 42 | cùng manifest, seed 42 |
| Kết quả chấm | `eval_4000/6000.pt` = **live** | `ckpt_*.pt` = **live** |
| Ý nghĩa | “thêm 2k/4k bước từ P2 live” | “2k bước mới từ P2 EMA, D mới” |

Vì P2 production là **EMA**, so sánh `P2 2k` với `G1` là so **EMA vs live**. So sánh `P2 4k` với `G1`
là so **live 4k** với **live 2k+2k**, nhưng vẫn khác điểm xuất phát (live 2k vs EMA 2k) và optimizer.

### 3.10 Rủi ro kỹ thuật đã thấy trong source (chưa đo)

1. **D có thể nhìn padding**: `sample_crops` không dùng `frame_lengths`; batch pad 0 tới Tmax.
   Với batch 8 cùng N nhưng T khác nhau, một cửa sổ 64 frame có thể chứa vùng padding của row ngắn.
   D có thể học “0 = real” thay vì học cấu trúc mel. Chưa có thống kê tỉ lệ crop chứa padding.
2. **D không được lưu**: không thể resume đúng D; mọi nhánh GAN là D mới.
3. **Không scheduler**: G1/G2 không phải “cosine tiếp nối”; là 2k bước LR hằng.
4. **EMA/live asymmetry** ở §0.9 và §3.9.
5. **`hard_stops` của P2 bật nhưng run dừng 7.600 không có lý do ghi lại**; không có `last.pt`.
6. **Không có `resolved_config.json`** trong `runs/adv_p2_20k`, `runs/v1830_g1/g2` — chỉ có
   `run.json` (args). Muốn tái lập phải đọc `run.json`.

---

## 4. Dữ liệu và alignment

### 4.1 Số giờ / số câu / video

**[TÍNH]** trực tiếp từ `data/codelung/v32/*_clean_v2_dur_seg_slot4.jsonl`:

| Split | Segment (row) | Clip cha | Video | Âm tiết | Type (normalized) | Frame | Giờ segment |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 18.226 | 5.807 | 13 | 220.384 | 2.552 | 5.421.000 | 16,06 |
| val | 3.378 | 1.011 | 12 | 41.087 | 1.549 | 1.021.241 | 3,03 |
| holdout | 2.611 | 842 | 9 | 34.746 | 1.369 | 838.547 | 2,48 |
| **tổng** | **24.215** | **7.660** | **25** | **296.217** | — | **7.280.788** | **21,57** |

Cách tính giờ: `frame * 256 / 24000 / 3600`. Đây là **giờ sau khi cắt segment** (mỗi row là một segment
3–10 s), không phải 51,73 h corpus gốc. Trước khi cắt: `train_clean` 27,28 h · `val_clean` 4,90 h ·
`holdout_clean` 3,93 h (`docs/DATASET_V2.md §2b`), sau đó `cut_segments` bỏ đuôi vụn và cắt tại gap dài.

Thời lượng/gap cấp âm tiết:

| Split | dur median | dur p95 | dur max | gap=0 | gap 1–2 | gap ≥3 | gap max | lead ≠ 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 15 frame | 32 | 88 | 150.862 | 3.102 | 66.420 | 60 | 0 |
| val | 15 | 32 | 77 | 26.865 | 664 | 13.558 | 60 | 0 |
| holdout | 14 | 32 | 84 | 23.392 | 509 | 10.845 | 60 | 0 |

- `gap` là số frame lặng **sau** mỗi âm tiết, gồm cả gap đuôi row; tổng gap chiếm **33,6%** frame train.
- `lead_frames = 0` cho mọi row segment (`cut_segments.py:148`); lặng đầu clip đã bị cắt ở segment đầu.
- Không có gap nội bộ > 60 frame (0,64 s): `cut_segments --max-gap-frames 60` biến gap dài thành
  **ranh giới segment**; `long_gaps` train 19.324, val 3.579, holdout 2.739 (`cut_segments_report.json`).
- Mục tiêu segment 3–10 s: `min_frames 281`, `max_frames 938`; segment hữu thanh < 23 frame bị bỏ
  (`dropped_tiny`: train 2.274, val 420, holdout 359). `cut_segments_report` ghi `max_seg` 998 frame
  (bao gồm gap đuôi) ⇒ **không phải giới hạn cứng 10 s**. Số segment tạo ra: train 18.226 từ 5.807 clip
  (13.677 pause cuts, 5.289 clip bị chia), val 3.378 từ 1.011 (2.614 cuts), holdout 2.611 từ 842 (1.969 cuts).

### 4.2 Chia train/val/test: theo video, không theo câu

- 12 video ít giờ nhất = val (cap 150 clip/video), phần dư = holdout; 13 video còn lại = train
  (`docs/DATASET_V2.md §1`; `tools/build_split_v2.py`).
- **[TÍNH]** giao video: train∩val = 0 · train∩holdout = 0 · val∩holdout = 0. Giao clip cha cũng = 0.
- 12 video val: `2b65WyiFC9k, 4ITcvEa_Acc, 50swDIypplU, BD83pScFukQ, EM_ObHWvd94, EUNlJU1BuzE,
  VeU_85bZlrY, _k5A863eLQE, lRojc8KDJ6k, o8RjAmBtyZo, vEtfty4Cc6Y, z0zSgR_oie4`.
- 9 video holdout là tập con của 12 video val (cùng video, clip khác): `2b65WyiFC9k, 4ITcvEa_Acc,
  BD83pScFukQ, EM_ObHWvd94, EUNlJU1BuzE, VeU_85bZlrY, o8RjAmBtyZo, vEtfty4Cc6Y, z0zSgR_oie4`.

**Các bộ eval v18 có thuộc train/val không?**

| Bộ | Số câu | Nguồn | Train? | Val? |
|---|---:|---|---|---|
| v18.31 free/TF | 300 | val manifest, stratified round-robin 12 video | không | **có** |
| v18.32 blind | 36 | val, 12 video × 3 bucket | không | **có** |
| v18.33 judge | 300 | val, 25/video | không | **có** |
| v18.35 screen | 110 | val, 11 video × 10, loại câu đã dùng | không | **có** |
| CER holdout | 20/100 | holdout | không | không (holdout) |

⇒ **Không có leakage từ train**, nhưng **tái sử dụng val rất nhiều** (300 + 36 + 300 + 110 câu, có loại trừ
ở v18.35). Không có test set độc lập cho quyết định “chọn bản thắng”; holdout chỉ dùng CER. Đây là
hạn chế của quy trình, không phải lỗi dữ liệu.

### 4.3 Transcript, durations, gaps, F0/voicing

**Transcript**
- Nguồn: phụ đề YouTube json3 của kênh `@codelung`, làm sạch caption cuộn/nhãn `[Music]`, gộp câu
  `max_gap 0,35 s` / `max_chars 220`, cắt clip 1–20 s (`docs/YOUTUBE_DATASET.md §4`).
- Không dùng ASR để tạo transcript train.
- Lọc fallback: 13.215 clip/51,73 h → 12.230 clip/47,48 h sau khi bỏ 6 video khách (937 clip) + 48 clip
  nhiễm; `overrides_tech_vi.json` 570 mục; fallback 1,22% → **0,80%** âm tiết; `train_clean` 24,39 → 27,28 h
  (`docs/DATASET_V2.md §1–§2b`).
- MFA corpus v2: 9.619 clip · 0 fallback · 2.877 từ · 120 phone.

**Durations / gaps**
- `tools/mfa_to_durations.py`: `durations` = số frame **hữu thanh** cấp âm tiết từ word tier; khoảng lặng
  tách thành `gaps` (sau từng âm tiết, phần tử cuối = đuôi clip) + `lead_frames`; bất biến
  `lead + Σdurations + Σgaps == T`.
- `tools/cut_segments.py`: cắt tại pause/ranh giới âm tiết, ưu tiên pause; segment gồm pause đuôi;
  `frame_range` cắt lặng đầu/đuôi; đuôi < 0,25 s bị bỏ; `max-gap-frames 60` (v32) biến gap dài thành
  ranh giới; tail gap cap 60.
- `gaps` cuối row là nhãn thật (A2), không bị bỏ như manifest cũ; `pause_targets` dùng mọi vị trí có gap
  (`v2_targets.py:89-108`), gate target = `gap >= 3` frame.

**F0 / voicing**
- `configs/small_codelung_v32.yaml:41-45`: `method: pyworld`, `f_min 70`, `f_max 400`, `interpolate: true`.
- `features.py:222-246`: `pyworld.harvest` + `stonemask`; `frame_period = 1000*hop/sr = 10,667 ms`;
  target `T = 1 + len//hop`; nội suy tuyến tính **log-F0** qua vùng vô thanh; **vuv giữ 0**.
- `f0_stats_v2.json`: `f0_log_mean 4,803942` · `f0_log_std 0,167907` · `energy_mean −1,025423` ·
  `energy_std 1,768254`; shape mean/std 4 hệ số; prior theo 6 thanh.
- `v2_targets.f0_targets`: `voiced = vuv>0,5 & f0>1e-3`; `s_k = DCT4(z)/√n`; mask Addendum 6.
- **Không dùng `f0>0` làm voicing**: f0 đã nội suy; chỉ `vuv>0,5` (F4). `f0` có thể >0 ở frame vô thanh
  không thuộc gap, vì contour `s_k` vẫn sinh giá trị.

### 4.4 Silence đầu/cuối, pause trong câu, vô thanh, frame không F0

| Trường hợp | Cách xử lý | Hệ quả |
|---|---|---|
| Lặng đầu clip | `cut_segments` cắt; `lead_frames=0` cho row segment | model không học lead; `G_lead=0` khi train segment |
| Lặng đuôi | gộp vào gap cuối; cap 60 frame (v32) | model học gap đuôi ≤0,64 s; pause dài hơn do runtime |
| Gap nội bộ > 60 frame | **cắt thành segment mới**, không đưa vào model | model không bao giờ thấy pause >0,64 s trong row |
| Gap 1–2 frame | vẫn là gap, gate target = 0 (vì <3) | không kích hoạt pause gate; vẫn có frame lặng trong mel loss |
| Gap ≥3 frame | gate target = 1; `pause_log_gap = log1p(gap)` | BCE + MSE; mass weight |
| Frame vô thanh | `vuv=0`; f0 nội suy; vẫn có mel target | mel L1 vẫn tính (trừ khi `mel_silence_mask` bật, P2 không bật) |
| Frame không có F0 (không voiced nào) | f0=0 sau interp? `estimate_f0_pyworld` nếu `voiced.any()` false thì giữ f0 gốc 0; `vuv=0` | F0 loss mask theo `vuv`/ratio; không dùng f0>0 |
| Lặng kỹ thuật số | W1: 3,64% frame RMS < −90 dBFS; 22,6% nền −90…−50 | model thường render nền trung gian, không im kỹ thuật số |

### 4.5 Bộ lọc dữ liệu và thống kê bị loại

| Bộ lọc | Số loại | Nguồn |
|---|---:|---|
| Video loạt khách (6 video) | 937 clip | `DATASET_V2.md §1` |
| Clip nhiễm tiếng khác (diarization `other_s ≥0,5s`) | 48 clip | idem |
| Fallback text | 1,22% → 0,80% âm tiết; 570 overrides | `DATASET_V2.md §2b` |
| Segment quá ngắn (<0,25 s hữu thanh) | train 2.274 · val 420 · holdout 359 | `cut_segments_report.json` |
| MFA QC | 9.619 clip, 0 fallback, 120 phone; loại clip lỗi phone >500 ms / âm tiết >1 s / lệch từ / C4 score | `DATASET_V2.md §5` |
| Loudness | 22 clip chạm trần +14 dB; 333 clip giảm >12 dB; sau chuẩn hoá mọi video −26,00 dBFS | `DATASET_V2.md §2` |
| Clip QC v31 | 7.440 clip, 0 error; SNR p5 37,8 dB · p50 57,5 · min 13,8; noise floor p50 −79,2 dB · p95 −59,0 | `artifacts/clip_qc/v31_report.json` |

### 4.6 Batch / crop / padding

- `SlotDataset` một mẫu = một segment; `min_syllables=2` cho train (MPS conv backward N=1 abort).
- `SameNBatchSampler` gộp index **cùng N**; `collate_slot_batch` **raise** nếu N khác
  (`dataset_v2.py:200-214`). Batch 8.
- Frame pad 0 tới `Tmax`; `frame_lengths` lưu độ dài thật (`dataset_v2.py:215-238`).
- Reconstruction loss `masked_l1` loại padding (`losses.py:13-29`); `mel_sharp` mask cặp frame;
  `vuv/energy/phonation` mask theo `active` segment. **Padding không vào loss tái tạo.**
- **Nhưng** D crop không dùng `frame_lengths` (§3.5) ⇒ padding có thể vào **loss adversarial**.
- `val` batch 1 để metric ổn định; base `validate_batches 40`; finetune `val_batches 10`.
- `augment_edges` không bật trong v32 (không có key `training.augment_edges`).

### 4.7 Âm lượng / nhiễu khác nhau giữa video

- Trước chuẩn hoá: lệch tới **22 dB** giữa video; gain per-clip p5/p50/p95 = **−10,9 / −1,8 / +10,2 dB**;
  22 clip chạm trần +14 dB; 333 clip giảm >12 dB. Sau chuẩn hoá: **mọi video −26,00 dBFS**
  (`DATASET_V2.md §2`). Đây là loudness, không phải noise.
- Nhiễu/nền theo video (W1, 2.300 row, 666.291 frame): 3,64% frame im kỹ thuật số (<−90 dBFS),
  22,61% nền nhiễu (−90…−50). Tập trung theo video: `5KnS0TGxWLg` 13,2% · `YFfCmXJk` 10,2% ·
  `VGYWEeV2CzM` 11,06% · `nh4wz4wlK4Q` 8,67% · `VeU` 14,33% · `BD83pScFukQ` 14,05%;
  17/20 prefix ~0%. Có **461 gap digital** và **2.175 gap nền** (`artifacts/bench/w1_gate.json`).
- Hệ quả thước N1: `v1829_l2_strat.json` cho thấy **9/12 prefix có gate**; bin <−50 **chỉ tồn tại ở video
  có gate** (`n1_nogate = 0` frame). Vì vậy N1 là phép đo **chỉ trên video gate**, không đại diện toàn bộ
  (ghi trong `A0_V18_GROUP1.md §21`).

### 4.8 Lỗi text/audio/alignment đã biết nhưng chưa sửa

| Lỗi | Bằng chứng | Trạng thái |
|---|---|---|
| **Eval free-run làm mất dấu câu** | `v1833_render_eval.py:62` `" ".join(raw)`; `row["text"]` cũng không có dấu câu; `iter_synthesize_sentence` nhận `gaps=0`; pause 1,69% vs 28,68% | **chưa sửa**; ảnh hưởng mọi kết luận pause/duration C→D |
| Đọc chữ cái theo tên Việt, người đọc theo kiểu Anh-Việt | D1.4: b/c/p/t/v/đ/r thắng kiểu Anh-Việt 0% tên Việt; CTC 5,3% trên từ đã biết ⇒ thước CTC không dùng được | **chưa sửa**; cần trang nghe/answers |
| `in xờ pếch` / `đê mô` | Là phiên âm tiếng Anh (inspector/demo), không OOV; entry custom_reading bị revert vì đổi 3→4 âm tiết làm lệch durations | **chưa sửa** |
| “rồi” bị kéo dài | thật median 17 frame, model 42 frame; pause sau “rồi” 29 frame | giảm bằng gate 1,01 + `--duration-clamp` (mặc định tắt) |
| “ách”, “treo” quá hiếm | train 220.384 token/2.552 type; `ách` 1 lần, `treo` 1 lần; 62 âm tiết trong 40 câu drift <10 lần | **chưa sửa**; lỗ hổng độ phủ |
| Coda NH/N | MFA dict có biến thể `nh → n`; chưa xác nhận dict nào dùng; centroid ngược chiều (xanh 395 Hz < xan 450 Hz) | **chưa sửa** (AGENT0_INFO A3/B4) |
| F0 halving trong nhãn | 0,78% (HUYEN 0,5%) | không sửa hàng loạt; chỉ log |
| ASR real CER cao | 300 câu: real macro CER 0,664 / micro 0,413; P2 macro 0,542 / micro 0,291 | **không phải lỗi audio**; reference/ASR; đã loại CER khỏi luật v18.34 |
| `dataset_v2` docstring `[T,20]` | `dataset_v2.py:32`; thực tế 100 bin | **chưa sửa** (doc) |
| `TRANG_THAI` ghi production G1 | `TRANG_THAI.md:9-10`; config production ghi P2 | **chưa sửa** (doc) |
| P2/G1 metadata lệch | P2 có `mel_hash` + `f0_stats`; G1 không có cả hai; `Synthesizer` fallback `f0_stats_path` (`infer.py:142-155`), `verify_mel_hash` bỏ qua khi thiếu hash (`features.py:110-126`) | chưa sửa; G1 load bằng fallback config |
| P2 run dừng 7.600 | `runs/adv_p2_20k` không có `last.pt`; log dừng 7.600, không lý do | chưa điều tra |
| P1 checkpoint v1 | `ckpt_4000/6000` thiếu `arch` ⇒ factory đoán `v1_syl`; phải bọc metadata | đã bọc cho eval, chưa sửa gốc |
| P2 4k/6k thiếu `arch` | `ckpt_4000/6000` gốc thiếu metadata; đã bọc thành `eval_4000/6000.pt` | đã bọc cho eval, chưa sửa gốc (`A0_V18_GROUP1 §8`) |

---

## 5. Bằng chứng phân biệt acoustic và vocoder

### 5.1 Định nghĩa A/B/C/D

| Ký hiệu | Định nghĩa | Cùng câu? | Cùng vocoder? | Ghi chú |
|---|---|---|---|---|
| **A** | WAV thật (audio gốc) | — | — | mốc tham chiếu, không phải “sạch tuyệt đối” |
| **B** | mel thật → Vocos (copy-synth), **GT duration** | A/B cùng row | Vocos pretrained | copy-synth không phải waveform sạch: iSTFT trả `(T−1)·256`, đã thêm bất tuần hoàn |
| **C** | mel acoustic **TF** (GT duration/seg/gaps/lead) → Vocos | cùng row với B | cùng Vocos | C và B cùng số frame |
| **D** | production free-run → Vocos | cùng row | cùng Vocos | **text eval bị mất dấu câu** + gate 1,01; khác cả duration/gap |

Các bộ số dưới đây **không chấm mới**; chỉ đọc `artifacts/bench/*.json`.

### 5.2 Cùng 300 segment (v18.31)

| Bản | flat_V | flat_8k | HNR (dB) | <−50 dBFS | mean dB | duration ratio |
|---|---:|---:|---:|---:|---:|---:|
| A — WAV thật (cùng 300 câu) | 0,1112 | 0,0548 | −2,22 | 27,93% | −44,00 | — |
| B — copy-synth **TF** | 0,1191 | 0,0573 | −2,81 | 28,19% | −44,10 | — |
| C — P2 **TF** | 0,1878 | 0,0941 | −4,06 | — ★ | — | — |
| C — G1 **TF** | 0,1459 | 0,0724 | −3,98 | — ★ | — | — |
| C — G2 **TF** | 0,1370 | 0,0655 | −3,92 | — ★ | — | — |
| D — P2 free | 0,1698 | 0,0741 | −3,79 | 2,19% | −30,04 | 0,566 ± 0,146 |
| D — G1 free | 0,1425 | 0,0611 | −3,71 | 1,65% | −30,48 | 0,563 ± 0,148 |
| D — G2 free | 0,1430 | 0,0588 | −3,67 | 1,45% | −29,77 | 0,565 ± 0,147 |
| D — G1b free | 0,1626 | 0,0656 | −3,92 | 2,33% | −30,02 | 0,565 ± 0,147 |

Nguồn: `v1831_tf.json`, `v1831_free300.json`. ★ Cột `<−50` của TF **không phải** % frame `<−50` của
model: N1 dùng **mask copy-synth** (28,187% frame của copy) và áp cùng mask cho P2/G1/G2; các bảng
`v1831_tf`/`v1830_g12_tf` không xuất `levels` tổng cho model. Vì vậy ô này để “—”, tránh đọc nhầm.
Lưu ý A và D là hai điều kiện khác nhau (A là audio thật; D là free-run).
⚠️ Các số **BAK/UTMOS/CER của v18.33** cũng đo trên **D** (text mất dấu câu, pause 1,69%). BAK đo nền
nên có thể ít bị ảnh hưởng bởi pause hơn UTMOS, nhưng **chưa có thí nghiệm tách**; vì vậy kết luận
“G1 kém BAK” cần đọc cùng cảnh báo này, không chỉ cảnh báo EMA/live.
⚠️ A dùng **89.294 frame** (toàn audio thật), B/C dùng **88.998 frame** (`frame_range` GT); lệch 296 frame
(~0,3%). Vì vậy A/B không phải tập frame khớp tuyệt đối; đây là hạn chế nhỏ của bảng, không đổi hướng.

### 5.3 Trần copy-synth 40 segment (K1) và tách voiced/unvoiced

| | flat hữu thanh | Δ so A | flat vô thanh | Δ so A | HNR hữu thanh | Δ so A | % voiced |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — WAV thật | 0,1224 | — | 0,2688 | — | −1,93 | — | 61,7% |
| B — copy-synth | 0,1337 | **+9,2%** | 0,2759 | +2,6% | −2,51 | **−0,58** | 61,2% |
| C — P2 @2k | 0,1668 | +36,3% | 0,2545 | −5,3% | −3,74 | −1,82 | 68,3% |
| C — GAN2k | 0,1754 | +43,3% | 0,2529 | −5,9% | −3,75 | −1,83 | 70,9% |

Nguồn: `v181_frame_probe.json`; định nghĩa voiced = autocorr ≥0,45; HNR 1–5 kHz. Kết luận:
- Vocoder chỉ đóng góp **+9,2% flat hữu thanh** và **−0,58 dB HNR**; phần lớn khoảng cách còn lại là acoustic.
- P2 **rè hơn thật ở hữu thanh** (+36%) nhưng **kém ma sát ở vô thanh** (−5,3%); hai chiều triệt tiêu khi lấy trung bình.

### 5.4 Theo nhóm coda / slot (TF, 300 segment)

| Slot/nhóm | Frame | copy flat | P2 flat | Δ P2 | GAN flat | Δ GAN | Buzz P2 | Buzz GAN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C — tắc | 1.434 | 0,2003 | 0,3388 | +69% (300 TF; 40 TF cũ +96%) | 0,3364 | +68% | 41,6% | 44,3% |
| C — mũi | 4.634 | 0,0932 | 0,1726 | +85% | 0,1460 | +57% | **67,7%** | **72,7%** |
| C — bán NA | 4.453 | 0,1465 | 0,2225 | +52% | 0,2076 | +42% | 45,5% | 46,7% |
| N — mở | 13.092 | 0,1452 | 0,2111 | +45% | 0,2075 | +43% | 45,2% | 49,5% |
| N — mũi | 7.285 | 0,1029 | 0,1653 | +61% | 0,1497 | +45% | 37,5% | 34,8% |
| N — tắc | 3.185 | 0,1463 | 0,2415 | +65% | 0,2239 | +53% | 49,1% | 52,1% |

Nguồn: `d314_d3_n1.json` (flat) và `q12_n1_buzz.json` (buzz, định nghĩa GT-vô-thanh bằng autocorr WAV thật <0,45).
**Buzz nặng nhất ở C-mũi**, không phải C-tắc; C-tắc nặng ở **flat** (nhòe hài). Hai hiện tượng khác nhau.

### 5.5 Nền / mức to (N1)

| Bin RMS (q12, đã sửa) | Frame | copy flat | P2 flat | GAN flat | Δ RMS P2 | Δ RMS GAN |
|---|---:|---:|---:|---:|---:|---:|
| >−20 dBFS (bin 0) | 2.252 | 0,1473 | 0,2284 | 0,2249 | **−5,9 dB** | −6,7 dB |
| <−50 dBFS (bin 4) | 25.398 | 0,4828 | 0,4391 | 0,4231 | **+12,6 dB** | +15,1 dB |

Nguồn: `q12_n1_buzz.json` (sau khi sửa lỗi bin), `g0.json`. Đọc đúng:
- Ở frame rất nhỏ (<−50), model **to hơn copy ~12,6 dB** và **không phẳng hơn copy** (0,439 vs 0,483).
  Vậy “rè nền” = **nền quá to**, không phải “nền phẳng”.
- Ở frame rất to (>−20), model **nhỏ hơn copy 5,9 dB** nhưng **phẳng hơn** (0,228 vs 0,147) — hướng ngược,
  cho thấy model không tái tạo được nhóm frame cực to.
- Ở free-run, `<−50` chỉ 1,6–2,3% vì **không có pause** (D), không thể so trực tiếp với TF 28%.

### 5.6 Rè xuất hiện ở đâu: A→B, B→C hay C→D?

| Chặng | Bằng chứng | Kết luận |
|---|---|---|
| **A→B** (Vocos) | copy-synth: **+9,2%** flat_V trên bộ 40 (300 câu: +7,1%), HNR −0,58 dB; L1 tai người “rè” copy = **1,00** (9/9 câu); §39: jitter NGÃ copy **+39%** so thật, HNR copy chỉ −0,15 | Vocos thêm **rung nhẹ ở creak** và **+7…9% flat**, nhưng **không phải nguồn rè chính**; copy-synth vẫn là mốc “sạch” tương đối |
| **B→C** (acoustic) | TF P2 vs copy: flat_V +57,7%, HNR −1,25 dB; S1 median MAE 3,71 dB (300 TF; bộ 40 cũ 3,56 dB); L1 tai người P2 “rè” **2,89**; §39: free-run HNR −1,02 so thật trong khi copy −0,15 | **Acoustic/mel dự đoán là nguồn chính của ồn/rè**; Vocos chỉ là bộ render |
| **C→D** (free-run) | P2 free flat_V 0,1698 < TF 0,1878; HNR −3,79 > −4,06; pause 1,69% (300, v18.33) vs 28,29% (36, v18.32); text mất dấu câu | **Không có bằng chứng free-run làm tăng rè**; khác biệt chính là **mất pause + đổi độ dài**, và đường đo bị confound. Cơ chế “nghịch lý nền” **chưa xác định** |
| **GAN G1/G2** | flat_V giảm 22–27%, HNR tốt hơn, **flat_V G1 thắng 88% câu**, nhưng BAK −0,40 / UTMOS −0,10 / CER +0,063 | “Phẳng hơn” **không đồng nghĩa tốt hơn**; giám khảo tự động không xác nhận |
| **GTA Vocos** | copy-check L1 0,341–0,529 vs 0,133; flat/HNR tệ hơn P2 | Vocoder học “bám mel vào” nhưng mel vào đã mờ ⇒ không nét hơn; **trượt** |
| **E4-C Vocos** | mel L1 −11,3% nhưng F0 RMSE −0,36% (<3%), octave +9,6%, creak +3,8% | “Bám mel hơn” ≠ “đúng hơn”; **trượt** |

**Không coi copy-synth là waveform sạch tuyệt đối**: A→B đã +9,2% flat_V, −0,58 dB HNR, jitter +39% ở NGÃ;
copy-synth còn ngắn hơn WAV gốc tới 255 mẫu do iSTFT `(T−1)·256`. Vì vậy mọi “trần Vocos” phải ghi rõ
là **trần của Vocos pretrained**, không phải trần vật lý.

### 5.7 Residual / phổ theo speech-pause

- **Chưa có** phân tích residual waveform (A−B, B−C, C−D) trong artifact.
- Có phân tích phổ theo **voiced/unvoiced** (`v181_frame_probe`), theo **slot/coda** (`d314_d3_n1`),
  theo **mức to** (`q12_n1_buzz`, `g0`), và theo **video** (`v1829_l2_strat`). Không có residual.
- `v1830_metric_corr.json`: tương quan với điểm tai (45 ô): flat_8k rho 0,377 (trong câu 0,80) ·
  HNR −0,460 (−0,70) · inter-harmonic −0,478 (−0,90) · flat_V 0,313 (0,50) · S1 0,421 (0,50) ·
  N1 median 0,301 (0,50). Chỉ 9 câu/45 ô ⇒ **định hướng**, không kết luận chắc.

---

## 6. Mục tiêu và giới hạn triển khai

### 6.1 C5 nguyên văn và các tiêu chí chấp nhận hiện hành

**C5 trong bảng ràng buộc kiến trúc TTS** — `docs/TTS_ARCHITECTURE_RESEARCH.md §1.1`
(lưu ý: `docs/SYLLABLE_SYSTEMS_RESEARCH.md` cũng có một “C5” khác, nói về kích thước hệ âm tiết;
không liên quan tiêu chí triển khai):

> **C5** | ONNX là định dạng triển khai chính (CPU, ARM, Apple Silicon, NVIDIA) | Ưu tiên op phổ thông
> (`Conv`, `ConvTranspose`, `MatMul`, `LayerNorm`, `GELU`/`SiLU`, `Softmax`). Tránh op hiếm/custom,
> tránh op phụ thuộc trạng thái (KV-cache, `Loop`, `Scan`), cảnh giác với `STFT`/`ISTFT`.

Các ràng buộc liên quan, nguyên văn:

> **C4** | Deterministic: cùng input + cùng `speed` ⇒ cùng output (byte-identical trên cùng phần cứng/numerics)
> | Không dùng sampling, không dropout khi suy luận, không stochastic duration predictor, không VAE latent
> sampling, không diffusion/flow nhiều bước có nhiễu.
>
> **C6** | Suy luận nhanh | Số lần forward cố định và nhỏ (1 lần cho acoustic + 1 lần cho vocoder).
> Tránh NFE (number of function evaluations) lớn.
>
> **C7** | Dữ liệu vừa phải (5–50 giờ, một người đọc) | Kiến trúc không đòi hỏi pretraining quy mô lớn
> (LLM/codec LM/SLM discriminator). Mô hình ~30–80M tham số là hợp lý.
>
> **C8** | 24 kHz | Cấu hình `n_fft=1024`, `hop=256`, `n_mels=100`.
>
> **C10** | Đọc đúng > không lặp/nuốt > thanh điệu > chất giọng > tự nhiên > biểu cảm.

**“C5” trong ngữ cảnh V18**: không có tệp định nghĩa. Chỉ có tham chiếu:

> “Không dùng < −50 % làm bằng chứng rè tốt/xấu; không đổi thước chính (**C5 chưa đủ 30 câu có điểm**);
> G1 giữ nhãn CHƯA XÁC NHẬN TAI; rollback P2 nguyên vẹn.”
> — `docs/handoff/A0_V18_GROUP1.md:745`

> “**Bộ 30 câu người thật** (người dùng thu) → chuẩn vàng + preference pairs.”
> — `docs/handoff/V11_agent2.md:148`

⇒ **Chưa có** định nghĩa C5 thành văn cho V18 và **chưa có** 30 câu có điểm. Đây là khoảng trống
lớn nhất của quy trình đánh giá bằng tai.

**Tiêu chí chấp nhận đang dùng:**

1. **Cổng ngoài train** (`docs/EVAL_TOOLS.md §4`):
   `mel_z ≤ mốc cũ · rough không giảm · C ≥ 0,78 · Δcreak không phình · CER ≤ 0,22 · (tuỳ chọn) tai A/B thắng/hoà`.
2. **Luật quyết định v18.33** (`A0_V18_GROUP1.md §25.0`, đăng ký trước):
   - G1 thắng nếu (1) ΔBAK trung vị > 0 **và** CI95 không chứa 0; (2) ΔUTMOS ≥ 0 hoặc CI95 không dưới −0,02;
     (3) ΔCER ≤ +0,002 hoặc CI95 không trên +0,002; (4) ≥2/3 video ΔBAK ≥ 0.
   - P2 thắng nếu đối xứng (ΔBAK CI95 < 0, hoặc UTMOS/CER xấu rõ).
   - Còn lại: chưa phân thắng bại, giữ nguyên.
3. **Luật sàng lọc v18.35** (`A0_V18_GROUP1.md §26.0`):
   ứng viên ĐẠT nếu ΔBAK trung vị > 0 + CI95 không chứa 0 + ΔUTMOS CI95 dưới > −0,02 + ≥2/3 video ΔBAK ≥ 0;
   không ai đạt ⇒ giữ P2, kết thúc.
4. **Production hiện tại**: `configs/production_v1831.json` ghi P2 + gate §25.0 với
   BAK CI `[−0,456, −0,350]`, UTMOS CI `[−0,143, −0,058]`, CER CI `[0,047, 0,096]`, `0/12` video.

### 6.2 Benchmark inference đã có

Nguồn: `A0_TURN10_RESULT_21_10.md §2.3` (Mac mini M6, ONNX Runtime CPU, **2 intra-op threads**,
audio 11,33 s, batch 1):

| Phần | Thời gian | RTF |
|---|---:|---:|
| Acoustic | 0,026 s | **0,002** |
| Vocoder | 0,043 s | **0,004** |
| Tổng (gồm iSTFT numpy) | 0,069 s | **0,006** (~166× realtime) |

- **Tất định**: md5 ×3 cùng input giống hệt (`9051f4d9013ba1b1f3198d7d90fa839f`).
- Đổi threads 1/2/4: 1 và 2 giống nhau; 4 khác **0,53% mẫu**, lệch tối đa 1 LSB (−90 dBFS);
  mặc định nhiều luồng vs 1 luồng: 2,5% mẫu, lệch ≤4 LSB (−78 dBFS). Production nên ghim
  `VIFASTT_INTRA_THREADS` (`onnx_engine.py:56` mặc định 2).
- **Chưa có**: RAM inference, cold start, batch 1/4/8, độ dài 3/10/30 s, latency first-chunk.
  Định nghĩa C6 yêu cầu đo các mục này (`TTS_ARCHITECTURE_RESEARCH.md §1.2`), nhưng chưa có artifact.
- Số cũ trong `AGENT0_INFO_21_10.md B6`: RTF ≈0,08 (CPU 2 luồng, acoustic+vocoder+iSTFT), khác lần đo
  0,006; có thể do model/độ dài/đo lần đầu. **Không dùng 0,08 làm số hiện tại** khi chưa rõ điều kiện.

### 6.3 Yêu cầu bắt buộc: đã chốt hay còn là giả định?

| Yêu cầu | Trạng thái | Nguồn / mức xác nhận |
|---|---|---|
| Một giọng, một phương ngữ/checkpoint | **đã chốt** (C1) | `TTS_ARCHITECTURE_RESEARCH.md`; dataset một kênh |
| Non-autoregressive | **đã chốt** (C3) | idem |
| Deterministic | **đã chốt** (C4); đo md5 ×3 | idem + A0 lượt 10 |
| ONNX là định dạng triển khai chính | **đã chốt** (C5) | idem; hiện 2 graph |
| 24 kHz | **đã chốt** (C8) | config |
| Suy luận 1 acoustic + 1 vocoder | **đã chốt** (C6) | code: 1 forward acoustic + 1 vocoder |
| Model ~30–80M | **giả định/khuyến nghị** (C7), không phải cổng cứng | hiện 20,58M (dưới khuyến nghị) |
| RTF ≤0,3 chấp nhận, ≤0,1 tốt | **giả định của agent** | `AGENT0_INFO_21_10.md B6:178`; không thấy user xác nhận |
| Offline hoàn toàn | **đã có** (không network lúc inference) | `runtime/`; pretrained local |
| Streaming | **đã có ở mức chunk**, không phải frame | `iter_synthesize`/`iter_synthesize_onnx`; latency chưa đo |
| ONNX fp16/int8 cho v2 | **chưa hỗ trợ**: chỉ fp32 | `export.py:244-251` raise `NotImplementedError` |
| ONNX 2 graph + iSTFT numpy | **đã chốt có chủ đích** | `DEPLOYMENT.md §64-71` |
| Giữ P2, không đổi production | **user xác nhận trong vòng này** | yêu cầu đầu vào |
| Không sửa code/train/benchmark | **user xác nhận trong vòng này** | idem |
| α = 0 toàn cục | **user đã chốt 21/10** | `A0_TURN6_RESULT_21_10.md §0` |
| 30 câu người thật (C5) | **chưa có** | `V11_agent2.md:148`; `A0_V18_GROUP1.md:745` |

### 6.4 Các thay đổi đã thử, kết quả, lý do dừng

| Nhánh / thay đổi | Kết quả | Lý do dừng / quyết định | Nguồn |
|---|---|---|---|
| GAN T4.1 trong `train_v2` (BCE, adv 1,0, ramp 2k) | 3 lần phân kỳ ở probe (~7,7e6); adv 165, mel 129 sau ~300 bước | bỏ, thay bằng `mel_sharp` tất định | `ACOUSTIC_V2.md §46.4`, `§28` |
| `mel_sharp` (T4.1 ổn định) | rough 0,363 @59k; giữ trong v32/P2 | nhận | `small_codelung_v32.yaml:145` |
| P1 adaptive λ | P1 2k–8k; drift p90 3,55→5,05; “từ to lên” tăng | không chọn; P2 fixed 0,3 tốt hơn | `A0_TURN6 §1-5` |
| P2 fixed adv 0,3, fm 0 | 2k: drift p90 4,67; 4k 6,27; 6k 7,43 | vượt cổng 2 mốc liên tiếp ⇒ dừng P2 20k, giữ 2k | `A0_TURN6 §4` |
| P2 4k/6k | flat_V 0,2119/0,1899; S1 0,421/0,462; HNR −3,69/−3,72 | kém P2 2k trên 3 thước ⇒ không chọn | `A0_V18_GROUP1 §8` |
| G1 (2k từ P2 EMA) | flat_V 0,1459; HNR −3,98; S1 +2,1%; BAK −0,40; UTMOS −0,10; CER +0,063 | ban đầu chọn theo flat_V/HNR; sau rollback P2 vì giám khảo tự động | `A0_V18_GROUP1 §22-25` |
| G2 (fm 1,0) | flat_V 0,1370; S1 +6,3% | trượt cổng S1 ≤5% | idem |
| G1b (2k tiếp) | flat_V 0,1626; HNR −3,92; <−50 2,33% | bão hoà ⇒ dừng nhánh | `A0_V18_GROUP1 §23` |
| g1_1k / g2_1k / g2_2k / p2_4k screen | không bản nào đạt luật BAK+UTMOS+video | giữ P2, kết thúc vòng | `A0_V18_GROUP1 §26` |
| GTA Vocos 5k | copy-check 0,341–0,529 vs 0,133; flat_V/HNR tệ hơn P2 | trượt cổng | `ACOUSTIC_V2.md §41.1` |
| E4-C Vocos weighted MR-STFT 5k | mel L1 −11,3%; F0 RMSE −0,36% (<3%); octave +9,6%; creak +3,8% | trượt cổng; giữ pretrained | `ACOUSTIC_V2.md §45.4` |
| E1 energy loss | drift p90 4,95→3,76 (xa thật), GV EMA 0,672 <0,72 | trượt GV | `A0_TURN10 §6` |
| E2 energy + lowband | GV 0,6722; drift chưa cải thiện | trượt GV | idem |
| F0 / Fc / Fd (2k, không GAN) | F0 flat +17%, S1 +8%; Fc ≈ F0; Fd S1 10,6 dB | cả 3 trượt cổng flat/S1 | `A0_V18_GROUP1 §21` |
| P2 + expander (−45/−40, trần −12/−18/−24) | flat gần như không đổi; S1 xấu 4,3–6,2 dB | không chọn | `A0_V18_GROUP1 §19-21` |
| chunk loudness 6d | drift p90 4,95→4,22 (đều hơn, xa thật) | mặc định tắt | `A0_TURN10 §4` |
| `f0_scale` 1,25/1,45 | audio gần như không đổi (84/98/115 → 84/98/114 Hz) | tắt; cần sửa kiến trúc | `A0_TURN10 §12` |
| `duration_clamp` | “rồi” 42→30 frame | mặc định tắt; steady gate 1,01 | `A0_TURN10 §15` |
| sh25 α=1,5 | RMS spike +38% ở v3.2; khuếch đại rè | bỏ α, mặc định 0 | `A0_TURN6 §0-2`; `ACOUSTIC_V2 §38.2` |
| Data v3.1 cap pause 0,7 s | mel/val/creak xấu; pause cap là thủ phạm | bỏ, quay v32 | `ACOUSTIC_V2 §50` |
| Data v3.3 DFN | không vượt b8 | giữ b8/v32 | `DATA_V33_DFN` |
| Data v3.4 coda/onset | không vượt v32 | giữ v32 | `DATA_V34_CODA` |
| Dropnoise / lọc video nhiễu | mel tốt hơn nhưng creak/CER không đạt | không dùng làm production | `ACOUSTIC_V2 §48` |
| Oversample / melsharp | trượt CER/rè | giữ b8 | `ACOUSTIC_V2 §47` |
| E4a clean creak F0 | jitter NẶNG −64%, nhưng 2 cổng mel val cũ trượt | từ chối | `ACOUSTIC_V2 §45.5` |
| P2 run 20k | log dừng 7.600, không `last.pt` | chưa rõ lý do; giữ checkpoint 2k | `runs/adv_p2_20k` |

---

## 7. Thông tin chưa có (không được coi là “đã kiểm chứng”)

1. **C5/V18**: không có định nghĩa thành văn, không có 30 câu người thật có điểm.
2. **RAM inference** (RSS/VRAM), cold start, batch 1/4/8, độ dài 3/10/30 s, latency first-chunk.
3. **Ma trận 2×2 EMA/live** cho P2 và G1 trên cùng bộ câu.
4. **Free-run có dấu câu**: chưa có bản eval giữ `punctuation` + gate 0,5; mọi kết luận pause hiện tại
   dựa trên text mất dấu câu + gate 1,01.
5. **Tỉ lệ crop D chứa padding** và ảnh hưởng lên D/G.
6. **Frame-level vuv/phonation target**; hiện chỉ cấp segment.
7. **Residual waveform A−B/B−C/C−D**; chưa có.
8. **ONNX mel_hash**; chưa có. Không chứng minh được ONNX vocoder dùng đúng stats.
9. **Quyết định vocoder production**: pretrained zero-shot hay E4-C? Config production không ghi;
   ONNX artifact dùng E4-C.
10. **D checkpoint** cho P2/G1/G2; không lưu, không resume được.
11. **P2 `last.pt`** và lý do dừng 7.600.
12. **Số người nói**: chỉ biết một giọng chủ kênh sau khi loại 6 video khách + 48 clip; không có
    thống kê diarization đầy đủ cho toàn bộ 25 video.
13. **Tỉ lệ lỗi transcript so bản gõ tay**: chưa có; CER thật 0,664 macro/0,413 micro là CER của ASR
    so reference, không phải lỗi transcript.
14. **Đánh giá tai G1**: chưa có (G1 luôn nhãn “CHƯA XÁC NHẬN TAI”).
15. **Đo per-video loudness sau v32**: có loudness.jsonl gốc và W1 digital; chưa có bảng per-video
    sau cắt segment.
16. **Đo `flat_V`/HNR trên cùng 300 câu cho cả A/B/C/D có dấu câu**; hiện A/B/C/D khác điều kiện.
17. **Script sinh `v1831_tf.json` / `v1831_free300.json` không có trong `tools/`** (đã grep toàn repo;
    chỉ còn JSON). Không tái lập được đúng lệnh đã sinh 300 câu nếu không đọc lại handoff.
18. **Không có log/commit cho lần chạy `v1831`**; chỉ có `meta` trong JSON (mode/segments/alpha/variants).
19. **P2/G1/G2 không log đủ thành phần loss**: `finetune_adv.py` chỉ ghi `mel/mel_pre/mel_sharp/adv/fm/
    grad_ratio/d_loss/d_acc` (`train_log.csv`), không ghi `f0/duration/pause/...`; muốn có phải sửa logger
    (không làm trong vòng này).

---

## 8. Tối đa 3 giả thuyết cải tiến (chỉ đề xuất phép thử, chưa thực thi)

### H1 — “Mất pause trong eval free-run là lỗi đường đo, không phải lỗi model”

**Bằng chứng ủng hộ**
- `tools/v1833_render_eval.py:62`: `text = " ".join(str(s.get("raw") ...))` — **không nối
  `s["punctuation"]`**. `tools/v1832_tf_free_diag.py:96`, `tools/v1828_gta_eval.py:269` cũng vậy.
- `frontend/pipeline.parse_text` chỉ gắn `punctuation`/boundaries từ dấu câu **trong text**; nếu text
  không có dấu câu thì `gaps` trong `iter_synthesize_sentence` toàn 0 (`infer.py:482-497`).
- Trong sentence mode, `gaps` **luôn được truyền** vào model (`infer.py:495-497`), nên `gate_threshold`
  không tham gia quyết định pause; `gate=1,01` không phải nguyên nhân. Nguyên nhân là `gaps=0`.
- Đo: free-run P2 pause **1,69%** / duration ratio 0,566; GT thật pause **28,68%**; TF P2 pause **28,29%**
  (36 câu v18.32). Chênh gần đúng bằng lượng pause bị bỏ.
- Top-level `row["text"]` cũng **không có dấu câu** (ví dụ row `2b65WyiFC9k_0001__s00`: text không có
  dấu chấm, nhưng syllable cuối có `punctuation="."`).

**Bằng chứng còn thiếu**
- Chưa đo bản free-run **có dựng lại dấu câu** trên cùng checkpoint/câu.
- Chưa biết người dùng thường nhập text có dấu câu hay không (giả định: có).
- Chưa đo tác động lên BAK/UTMOS (pause có thể ảnh hưởng tự nhiên, không chỉ duration).

**Phép thử nhỏ nhất phân biệt đúng–sai**
- Lấy đúng 300 câu v18.31 (hoặc 36 câu v18.32), cùng P2, cùng Vocos, cùng `iter_synthesize_sentence`.
- Dựng 2 text: (a) `raw join` như hiện tại; (b) `raw join` + nối `s["punctuation"]` vào cuối âm tiết.
- Chạy đúng 2 điều kiện (a)/(b); **không đổi gate** vì sentence mode luôn truyền `gaps`.
- Đo **pause %**, số gap, tổng duration, duration ratio, flat_V/HNR. Không chạy ASR/MOS.
- Nếu (b) làm pause % tiến gần 28% và duration ratio tiến gần 1,0 ⇒ H1 đúng, các kết luận C→D cũ
  phải đọc lại. Nếu không ⇒ H1 sai, model thật sự bỏ pause.
- (Tuỳ chọn, cùng bộ) chạy thêm chunk mode `iter_synthesize` với gate 0,5 trên text (b) để kiểm model
  có tự đặt pause hay không; đây là phép thử phụ, không thay (a)/(b).

### H2 — “So P2 EMA với G1 live là bất đối xứng; kết luận G1 kém có thể đổi nếu chấm cùng chế độ”

**Bằng chứng ủng hộ**
- `Synthesizer.__init__` nạp `state.get("model", state)` (`infer.py:137-141`); **không đọc `ema`**.
- P2 production `_ckpt/step_02000.pt` có `model` **trùng khít** `ema` của `runs/adv_p2_20k/ckpt_2000.pt`
  (max diff 0,0). Vậy P2 được chấm bằng EMA.
- G1 `runs/v1830_g1/ckpt_2000.pt` có `model` ≠ `ema` (max abs 0,0080; mean abs ~1,0e-4). Các candidate
  `g1_1k`, `g2_1k`, `g2_2k` cũng được chấm bằng `model` live.
- Log G1: EMA val 1,6485 vs live 1,6714 @2k; EMA thường tốt hơn. P2 2k: EMA 1,6167 vs live 1,6702.
- P2 4k/6k là **live**, nên so `p2_4k` với P2 2k cũng là EMA vs live.

**Bằng chứng còn thiếu**
- Chưa có điểm BAK/UTMOS/flat_V/HNR cho **G1 EMA** và **P2 live** trên cùng bộ 110/300.
- Chưa biết EMA có thực sự “gần P2” đến mức xoá khác biệt BAK hay không.

**Phép thử nhỏ nhất**
- Dùng checkpoint có sẵn, không train: chấm 4 biến thể trên cùng 110 câu screen (hoặc 300):
  P2 EMA (`_ckpt/step_02000.pt["model"]`, hiện tại) · P2 live (`runs/adv_p2_20k/ckpt_2000.pt["model"]`) ·
  G1 live (`runs/v1830_g1/ckpt_2000.pt["model"]`) · G1 EMA (`runs/v1830_g1/ckpt_2000.pt["ema"]`).
- Đo BAK/UTMOS + flat_V/HNR, ghép cặp theo câu, bootstrap theo video như v18.35.
- Nếu G1 EMA thu hẹp ΔBAK về 0 và UTMOS ≥0 ⇒ kết luận “G1 kém” chưa vững; nếu vẫn kém ⇒ củng cố P2.

### H3 — “Buzz ở coda là do vuv dự đoán cấp segment được đưa vào decoder; có thể kiểm bằng ép vuv thấp ở slot C”

**Bằng chứng ủng hộ**
- `v2_slot5.py:301-309`: `variance_parts = [f0, segment_out[...,4] (vuv logit), segment_out[...,5] (energy), phonation]`;
  `variance_proj = Linear(3+3, h)` (`:113`). Vậy decoder **có** nhận vuv logit dự đoán.
- `vuv` target là **tỉ lệ hữu thanh cấp segment** (`v2_targets.py:129-133`), không có target frame-level;
  decoder thấy một giá trị hằng cho cả slot.
- Buzz nặng nhất ở **C-mũi 67,7%** và C-tắc 41,6% (`q12_n1_buzz.json`); flat C-tắc +69…96% so copy
  (`d314_d3_n1.json`). Đây là chỗ cần can thiệp.
- `variance_override` đã có trong `forward` (`v2_slot5.py:203`, `:310`) — có thể ép kênh vuv mà **không
  cần sửa kiến trúc**.
- Đính chính: `A0_V18_GROUP1.md §8` nói “decoder KHÔNG có kênh vuv” là **sai ở mức chữ**; đúng nếu hiểu
  là “không có vuv ground-truth/frame-level”. Vì vậy hướng “thêm kênh vuv” không còn đúng; hướng đúng là
  **kiểm tra/đổi độ phân giải hoặc ép giá trị dự đoán**.

**Bằng chứng còn thiếu**
- Chưa đo vuv dự đoán thực tế ở slot C (cao hay thấp).
- Chưa biết decoder dùng kênh vuv mạnh hay yếu; thí nghiệm `f0=0` của A0 lượt 10 cho thấy kênh f0 chỉ
  ảnh hưởng ~10% audio, nên vuv có thể cũng yếu.
- Chưa có thử nghiệm ép vuv ở runtime.

**Phép thử nhỏ nhất**
- 300 segment TF, P2, một forward/segment; dùng `variance_override` để đặt **cột 1 (vuv)** = −10
  (hoặc 0) cho **slot C** (và tùy chọn 30% cuối slot N), giữ nguyên các cột khác
  (thứ tự variance: `f0, vuv, energy, phonation×3` — `v2_slot5.py:301-309`).
- Vocoder cùng config, đo `buzz_pct` (định nghĩa Q2), `flat_V`, HNR, S1; thêm đối chứng “ép vuv = +10”
  và “không ép”.
- Nếu buzz C giảm ≥20% tương đối mà flat_V toàn câu không xấu hơn 5% ⇒ H3 đáng theo; nếu không ⇒ loại.
- Chỉ khi probe này dương mới cân nhắc train frame-level vuv (tốn kém hơn).

### Các lead chưa đủ thành giả thuyết (ghi để không quên)

- **D nhìn padding** (§3.5/§3.10): cần đếm tỉ lệ crop chứa padding và D accuracy trên crop sạch/bẩn;
  nếu >5–10% crop bẩn, thử crop theo `frame_lengths` trong một probe 200 bước.
- **ONNX vocoder E4-C** (§2.3): không phải giả thuyết cải tiến, mà là **cần chốt lại artifact production**;
  nếu ONNX là đường phát hành, phải export lại từ Vocos pretrained + ghi `mel_hash`.
- **`mel_stats` energy-gate vs vuv-gate** (§2.6): nếu muốn so P2/v3.5 mel z, phải dùng cùng population;
  hiện không so được.

---

## 9. Phụ lục — file đã đọc / phép tính đã làm

**Source chính**
`src/vifasttts/infer.py` · `models/v2_slot5.py` · `models/vocos.py` · `models/istftnet.py` ·
`models/conformer.py` · `models/components.py` · `models/discriminators.py` · `models/v2_losses.py` ·
`models/slot_ops.py` · `train_v2.py` · `data/features.py` · `data/dataset_v2.py` · `data/v2_targets.py` ·
`runtime/onnx_engine.py` · `runtime/postprocess.py` · `frontend/pipeline.py` · `frontend/segments.py` ·
`export.py` · `cli.py` · `tools/finetune_adv.py` · `tools/v1833_render_eval.py` · `tools/cut_segments.py`.

**Config / checkpoint / artifact**
`configs/small_codelung_v32.yaml` · `configs/production_v1831.json` · `data/codelung/mel_stats.json` ·
`data/codelung/mel_stats_v35_speech.json` · `data/codelung/f0_stats_v2.json` ·
`data/codelung/v32/*_clean_v2_dur_seg_slot4.jsonl` · `data/codelung/v32/cut_segments_report.json` ·
`artifacts/listen_small_filt/adv_progress_p2/_ckpt/step_02000.pt` · `runs/adv_p2_20k/*` ·
`runs/v1830_g1/*` · `runs/v1830_g2/*` · `runs/v1830_g1b/*` · `runs/acoustic_v32/history.csv` ·
`artifacts/vocoder_vocos_stftc/last.pt` · `artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin` ·
`artifacts/onnx/*.onnx`.

**JSON đo đã dùng**
`v1831_tf.json` · `v1831_free300.json` · `v1832_tf_free.json` · `v1833_decision.json` ·
`v1834_audit.json` · `v1835_screen/screen.json` · `v181_frame_probe.json` · `v183_fuse.json` ·
`d314_d3_n1.json` · `q12_n1_buzz.json` · `g0.json` · `w1_gate.json` · `v1829_l2_strat.json` ·
`v1830_metric_corr.json` · `v1827_domain.json` · `v1828_gta_copy.json`.

**Phép tính tự làm (không chấm audio)**
- Đếm tham số từ `state_dict` của P2/Vocos/D.
- Đếm row/clip/video/syllable/frame/gap từ manifest v32.
- So `model` của `_ckpt/step_02000.pt` với `model`/`ema` của `runs/adv_p2_20k/ckpt_2000.pt`
  (max abs = 0,0 với `ema`).
- Kiểm `mel_config_hash` của config v32 so với `mel_hash` P2 (khớp).
- Kiểm công thức độ dài `torch.istft(center=True)` = `(T−1)·hop` bằng tensor nhỏ.
- Đọc metadata 3 file ONNX.
- Tính `mean|.|` của hai bộ mel stats (1,5102 vs 1,7605).

**Test thực chạy**: **không chạy `pytest`/`ruff`**; chỉ thêm tệp Markdown này, không sửa code.

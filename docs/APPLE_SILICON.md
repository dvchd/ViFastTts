# Apple Silicon: huấn luyện & suy luận trên Mac — khảo sát và quyết định

> Phạm vi: Mac mini M6 32GB (máy tham chiếu của repo) và các máy M-series khác.
> Số đo thực hiện ngày 11/10/2026 trên máy thật. Câu hỏi trung tâm: **có nên viết
> một luồng train/inference riêng cho Mac, song song với luồng CUDA?**
> Trạng thái tổng: `docs/PROJECT_STATUS.md`; số đo train: `docs/TRAINING_ESTIMATES.md` §3.1.

## 1. Các thư viện/đường chạy trên Apple Silicon

| Thư viện | Vai trò | Chạy trên | ANE? | Trạng thái trong repo |
| --- | --- | --- | --- | --- |
| **PyTorch + MPS** | Train + inference | GPU (Metal), CPU | Không | ✅ as-built: `training.device: auto` (cuda→mps→cpu) |
| **MLX** (Apple) | Train + inference Apple-native (unified memory, lazy) | GPU + CPU | Không | ⚠️ đã **đo đánh giá**, chưa tích hợp (xem §2) |
| **Core ML / CoreMLTools** | Inference (ANE/GPU/CPU), app iOS/macOS | ANE + GPU + CPU | **Có** | ⚠️ chưa convert trực tiếp; có thể dùng khi cần ANE thuần |
| **ONNX Runtime EPs** | Inference | CPU / CoreML (GPU+ANE) / CUDA+TensorRT (NVIDIA) | Qua CoreML EP | ✅ as-built: `synthesize-onnx --provider auto\|cpu\|cuda\|coreml\|tensorrt` |
| TensorFlow-metal | Train (hệ TF) | GPU | Không | Không xét (ngoài scope) |

Điểm cần nhớ: **PyTorch MPS và MLX đều KHÔNG dùng Neural Engine** — chỉ Core ML
(CoreMLTools hoặc CoreML EP của ONNX Runtime) mới chạm tới ANE.

## 2. Số đo trên Mac mini M6 32GB (đo thật, không phải ước lượng)

**Benchmark tương đương** — transformer encoder 6 lớp, hidden 256, ffn 1024, 4 heads,
batch 8 × 600 token (cùng cỡ encoder của model acoustic):

| Việc | MLX 0.32 | PyTorch MPS 2.14 | MLX nhanh hơn |
| --- | --- | --- | --- |
| Train step | **83 ms** | 100 ms | ~1,2× |
| Inference | **10 ms** | 21 ms | ~2,1× |

**Inference ONNX (model base, đầu vào thật, provider 1 luồng):**

| Pipeline | CPU EP | CoreML EP (GPU/ANE) |
| --- | --- | --- |
| Acoustic (6 âm tiết / 25 âm tiết) | **1,5 / 2,2 ms** | 2,3 / 3,2 ms |
| Vocoder (6 / 25 frame mel) | **0,3 / 0,7 ms** | 0,4 / 0,8 ms |

**Ghi chú kỹ thuật khi dùng CoreML EP** (theo
[ORT CoreML EP docs](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)):

- Cần `ModelFormat=MLProgram` + `MLComputeUnits=ALL` để mở GPU/ANE.
- Model acoustic có shape động (LengthRegulator): phải đặt
  `RequireStaticInputShapes=1` mới chạy ổn định; khi đó CoreML nhận ~129/807 node
  (phần còn lại CPU) — đó là lý do CoreML chưa nhanh hơn CPU ở kích thước này.
- ORT biên dịch subgraph Core ML mỗi session → nên bật cache
  (`ModelCacheDirectory`, repo dùng `~/.cache/vifasttts/coreml`, override bằng
  `VIFASTTTS_COREML_CACHE`).
- macOS có thể in log hệ thống ("Context leak detected", CoreAnalytics) — vô hại.

**Train trên Mac (bảng đầy đủ ở `docs/TRAINING_ESTIMATES.md` §3.1):** acoustic MPS
0,31 s/bước (nhanh hơn CPU 0,53); vocoder **CPU nhanh hơn MPS** (0,39 vs 0,93 s/bước)
→ auto chọn CPU cho vocoder.

## 3. Quyết định: có nên viết luồng riêng cho Mac?

**Train — KHÔNG viết luồng song song.**
Một codebase PyTorch, chọn thiết bị bằng `training.device: auto` (CUDA → MPS → CPU).
Lý do:

1. Lợi ích đo được của MLX chỉ ~1,2× cho train (encoder tương đương) — không đủ bù
   chi phí viết và **duy trì song song** một bản model + loss + dataset + export thứ hai.
2. Model nhỏ (6–32M tham số); PyTorch MPS đã chạy end-to-end với test smoke thật.
3. Export ONNX là cầu nối trung lập: train ở đâu cũng ra **cùng một artifact**.

**Khi nào nên xem lại MLX:** model ≥ ~100M tham số, cần flow-matching/LLM-style với
unified memory, LoRA fine-tuning, hoặc benchmark trên **model thật** cho thấy MLX
nhanh hơn > 1,5×. Khi đó viết `models/mlx_*.py` như một backend thứ ba, không đụng
luồng hiện tại.

**Inference — KHÔNG cần code riêng.** Cùng file ONNX, chỉ đổi **execution provider**:

| Máy | Provider nên dùng |
| --- | --- |
| Mac (M-series) | `cpu` (đo là nhanh nhất cho model này); `coreml` khi muốn thử GPU/ANE |
| NVIDIA | `cuda` hoặc `tensorrt` |
| Không có GPU | `cpu` |

Đây chính là câu trả lời cho "song song với luồng CUDA": **cùng code, cùng artifact,
khác provider/device** — không fork codebase:

```text
train (cuda hoặc mps hoặc cpu)  ──export-onnx──▶  model.onnx + vocoder.onnx
                                                        │
              ┌─────────────────────────────────────────┼──────────────────────────┐
              ▼                                         ▼                          ▼
     NVIDIA: --provider cuda/tensorrt        Mac: --provider cpu/coreml      CPU-only: --provider cpu
```

## 4. Cách dùng nhanh

```bash
# Train trên Mac: acoustic chạy MPS, vocoder CPU (auto)
vifasttts train --config configs/base_24k.yaml \
  --train-manifest data/train.jsonl --val-manifest data/val.jsonl --output runs/base

# Ép thiết bị cụ thể khi cần
vifasttts train ... --config configs/base_24k.yaml   # sửa training.device: mps|cpu

# Inference ONNX — chọn provider theo máy
vifasttts synthesize-onnx --acoustic model.onnx --vocoder vocoder.onnx \
  --text "Xin chào" --output out.wav --provider cpu      # Mac: nhanh nhất (đo)
vifasttts synthesize-onnx ... --provider coreml          # Mac: thử GPU/ANE
vifasttts synthesize-onnx ... --provider cuda            # máy NVIDIA
```

## 5. Giới hạn & hướng phát triển

- **ANE chưa được khai thác tối đa**: CoreML EP chỉ nhận phần lớn node khi mạng tĩnh;
  muốn ANE thuần cho acoustic cần export "durations là input" (shape tĩnh) hoặc
  convert thẳng sang Core ML bằng CoreMLTools (9.0) rồi chạy `compute_units=ALL`.
- **MLX**: benchmark hiện chỉ trên encoder tương đương; muốn kết luận chắc phải port thử
  model thật (việc lớn, xem §3).
- **AMP**: `mixed_precision` chỉ bật trên CUDA; MPS chạy fp32 (đủ nhanh ở cỡ này).
- **Điện năng**: Mac mini tối đa 155W — train chuẩn ~7–15k đồng tiền điện (rẻ hơn cloud
  nếu đã có máy); xem `docs/TRAINING_ESTIMATES.md` §6.

## 6. Nguồn

1. MLX — Apple machine learning research: <https://github.com/ml-explore/mlx>
   (unified memory, lazy computation, GPU+CPU; **không** ANE).
2. ONNX Runtime — CoreML Execution Provider:
   <https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html>
   (options `ModelFormat/MLComputeUnits/RequireStaticInputShapes/ModelCacheDirectory`).
3. PyTorch — MPS backend notes: <https://docs.pytorch.org/docs/2.14/notes/mps.html>.
4. Apple — Mac mini technical specifications: <https://www.apple.com/mac-mini/specs/>
   (M6: 12-core CPU, 12-core GPU, Dual 16-core Neural Engine, 153/170 GB/s).
5. Core ML Tools: <https://coremltools.readme.io/> (convert + `compute_units=ALL`).

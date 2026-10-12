"""Copy-synthesis cho file ghi âm bất kỳ — cặp tối thiểu, demo, kiểm nhanh (R8-1 bước 1).

    python tools/copy_synthesis.py --config configs/small_codelung.yaml \
        --checkpoint artifacts/pretrained/vocos-mel-24khz/pytorch_model.bin \
        --input-dir data/minimal_pairs --output artifacts/minimal_pairs/step_0

- Mỗi file vào sinh 2 file ra: `<stem>_orig.wav` (bản 24 kHz để nghe A/B) và
  `<stem>_synth.wav` (copy-synthesis) — nghe xen kẽ để phát hiện rè/kim loại/mất creak.
- Kèm `phonation.json`: HNR/tilt/glottalization/perturbation gốc vs synth
  (xem `tools/voice_quality.py`; cần pyworld — extra `f0`).
- `--checkpoint` nhận cả checkpoint `train-vocoder` (khóa `vocoder`) lẫn file HF thô
  (`pytorch_model.bin`) — dùng luôn được cho zero-shot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:  # cho phép chạy `python tools/copy_synthesis.py`
    sys.path.insert(0, str(_ROOT))


def _find_inputs(path: Path) -> list[Path]:
    """Một file WAV, hoặc mọi WAV trong thư mục (không đệ quy)."""
    if path.is_dir():
        return sorted(item for item in path.iterdir() if item.suffix.lower() == ".wav")
    return [path]


def main() -> None:
    """Chạy copy-synthesis từng file + ghi phonation.json (nếu có pyworld)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt hoặc pytorch_model.bin HF")
    parser.add_argument("--input", default=None, help="Một file WAV")
    parser.add_argument("--input-dir", default=None, help="Thư mục WAV")
    parser.add_argument("--output", required=True, help="Thư mục kết quả")
    parser.add_argument("--device", default="cpu", help="cpu (mặc định) | mps | cuda")
    parser.add_argument("--no-phonation", action="store_true", help="Bỏ đo phonation")
    args = parser.parse_args()
    if bool(args.input) == bool(args.input_dir):
        raise SystemExit("Truyền đúng một trong --input hoặc --input-dir")

    import torch
    import yaml

    from tools.eval_vocoder import load_vocoder
    from vifasttts.data.features import canonical_mel_config, extract_mel_energy, load_wav
    from vifasttts.infer import write_wav

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sample_rate = int(config.get("sample_rate", 24000))
    mel_config = canonical_mel_config(config)
    hop = int(mel_config["hop_length"])
    n_fft = int(mel_config["n_fft"])

    generator = load_vocoder(config, args.checkpoint, device=args.device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.input_dir or args.input)

    use_phonation = not args.no_phonation
    if use_phonation:
        try:
            import pyworld  # noqa: F401
        except ImportError:
            print("⚠️  Không có pyworld — bỏ phonation (cài extra 'f0' nếu muốn đo creak)")
            use_phonation = False
    compare_pair = None
    if use_phonation:
        from tools.voice_quality import compare_pair  # type: ignore[assignment]

    results: dict[str, dict[str, float]] = {}
    for wav_path in _find_inputs(source):
        wav = load_wav(str(wav_path), sample_rate)
        mel, _ = extract_mel_energy(wav, mel_config)
        with torch.no_grad():
            synth = generator(mel.unsqueeze(0).to(args.device))[0].float().cpu()
        write_wav(out_dir / f"{wav_path.stem}_orig.wav", wav.numpy(), sample_rate)
        write_wav(out_dir / f"{wav_path.stem}_synth.wav", synth.numpy(), sample_rate)
        info = f"  {wav_path.name}: {wav.numel()/sample_rate:.2f}s → {synth.numel()/sample_rate:.2f}s"
        if compare_pair is not None:
            metrics = compare_pair(wav.numpy(), synth.numpy(), sample_rate, hop, n_fft)
            results[wav_path.stem] = metrics
            delta = metrics.get("hnr_creak_delta")
            if delta is not None and math.isfinite(delta):
                info += f" · HNR creak Δ = {delta:+.2f} dB"
        print(info)

    if results:
        (out_dir / "phonation.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"Đã ghi {out_dir / 'phonation.json'}")
    print(f"Xong: {out_dir} (nghe `<stem>_orig.wav` / `<stem>_synth.wav`)")


if __name__ == "__main__":
    main()

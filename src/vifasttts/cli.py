"""CLI của ViFastTts (Typer): frontend, dữ liệu, train, suy luận, export, server.

Chạy `vifasttts --help` để xem toàn bộ lệnh; `--version` để in phiên bản.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from vifasttts import __version__
from vifasttts.config import load_config
from vifasttts.data.manifest import build_manifest
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.syllable import parse_syllable
from vifasttts.train import train_reference

app = typer.Typer(help="ViFastTts CLI")


def _version_callback(value: bool) -> None:
    """In phiên bản rồi thoát khi có `--version` (chạy trước mọi lệnh)."""
    if value:
        typer.echo(f"ViFastTts {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="In phiên bản rồi thoát",
    ),
) -> None:
    """ViFastTts — TTS tiếng Việt một giọng (frontend + model + ONNX)."""


@app.command("parse-syllable")
def parse_one(value: str) -> None:
    """Parse một âm tiết và in JSON đầy đủ (chính tả + âm vị + trạng thái)."""
    typer.echo(json.dumps(parse_syllable(value).to_dict(), ensure_ascii=False, indent=2, default=str))


@app.command("parse")
def parse(value: str, dialect: str = "north") -> None:
    """Parse cả câu theo vùng miền, in JSON từng âm tiết (kèm dấu câu/ranh giới)."""
    typer.echo(json.dumps([x.to_dict() for x in parse_text(value, dialect=dialect)], ensure_ascii=False, indent=2, default=str))


@app.command("g2p")
def g2p_cmd(
    text: str = typer.Argument(..., help="Câu cần phiên âm"),
    dialect: str = typer.Option("north", help="north, central hoặc south"),
    style: str = typer.Option("broad", help="broad hoặc narrow"),
    tone_format: str = typer.Option("contour", help="contour | number | none"),
) -> None:
    """Phiên âm IPA từng âm tiết (chỉ để phân tích, không nằm trong runtime)."""
    from vifasttts.frontend.g2p import syllable_to_phonemes, to_phoneme_string

    for item in parse_text(text, dialect=dialect):
        phonemes = syllable_to_phonemes(item, dialect, style=style, tone_format=tone_format)
        rendered = to_phoneme_string(phonemes) if phonemes else "(fallback)"
        typer.echo(f"{item.raw}\t{rendered}")


@app.command("build-manifest")
def manifest(
    metadata: str = typer.Option(..., help="File CSV 'id|text'"),
    wav_dir: str = typer.Option(..., help="Thư mục chứa WAV"),
    output: str = typer.Option(..., help="File JSONL đầu ra"),
) -> None:
    """Dựng manifest JSONL (text + âm tiết đã parse) từ CSV và thư mục WAV."""
    build_manifest(metadata, wav_dir, output)
    typer.echo(f"Đã ghi {output}")


@app.command("validate-manifest")
def validate_manifest_cmd(
    manifest: str = typer.Option(..., help="File JSONL cần kiểm tra"),
    wav_dir: str | None = typer.Option(None, help="Thư mục WAV (nếu manifest chỉ có id)"),
) -> None:
    """Kiểm manifest: WAV tồn tại, âm tiết rỗng; in coverage tone/coda/fallback."""
    from vifasttts.data.manifest import coverage_report, validate_manifest
    report = validate_manifest(manifest, wav_dir)
    typer.echo(f"Tổng {report['total']} dòng, {len(report['issues'])} vấn đề")
    for issue in report["issues"][:20]:
        typer.echo(f"  dòng {issue['line']}: {issue['id']} - {issue['issue']} ({issue.get('path', '')})")
    if len(report["issues"]) > 20:
        typer.echo(f"  ... và {len(report['issues']) - 20} vấn đề khác")
    coverage = coverage_report(manifest)
    typer.echo(f"Coverage: {coverage['syllables']} âm tiết, {coverage['fallback']} fallback")
    typer.echo(f"  tone: {coverage['tone']}")
    typer.echo(f"  coda: {coverage['coda']}")
    if coverage["missing_tones"]:
        typer.echo(f"  thiếu thanh: {coverage['missing_tones']}")


@app.command("youtube-fetch")
def youtube_fetch_cmd(
    channel: str = typer.Argument(..., help="URL kênh (/videos), playlist hoặc video YouTube của bạn"),
    out: str = typer.Option("data/youtube", help="Thư mục dataset đầu ra"),
    limit: int = typer.Option(0, help="Số video tối đa (0 = tất cả)"),
    demo: bool = typer.Option(False, help="Chạy demo: thử lần lượt video ngắn nhất tới khi có clip"),
    langs: str = typer.Option("vi,vi-orig", help="Phụ đề ưu tiên, cách nhau dấu phẩy"),
    dialect: str = typer.Option("north", help="Cách đọc số/ngày: north, central, south"),
    min_duration: float = typer.Option(1.0, help="Độ dài clip tối thiểu (giây)"),
    max_duration: float = typer.Option(20.0, help="Độ dài clip tối đa (giây)"),
    max_gap: float = typer.Option(0.35, help="Khoảng nghỉ tối đa để gộp câu (giây)"),
    max_chars: int = typer.Option(220, help="Độ dài text tối đa mỗi clip"),
    min_video_duration: float = typer.Option(0.0, help="Bỏ video ngắn hơn mức này (giây)"),
    max_video_duration: float = typer.Option(0.0, help="Bỏ video dài hơn mức này (0 = không giới hạn)"),
    title_filter: str | None = typer.Option(None, help="Chỉ lấy video có tiêu đề khớp regex"),
    keep_wav: bool = typer.Option(True, "--keep-wav/--no-keep-wav",
                                  help="Giữ WAV gốc 24 kHz để căn chỉnh lại sau này"),
    dedupe: bool = typer.Option(True, "--dedupe/--no-dedupe", help="Lọc clip trùng chữ"),
    min_rms_db: float = typer.Option(-45.0, help="Bỏ clip im lặng dưới mức RMS này (dBFS)"),
    max_fallback_ratio: float = typer.Option(0.34, help="Bỏ clip có tỉ lệ fallback cao hơn"),
    min_caption_density: float = typer.Option(2.0, help="Bỏ video có phụ đề thưa hơn (ký tự/giây)"),
    respell_codes: bool = typer.Option(True, "--respell-codes/--no-respell-codes",
                                       help="Làm sạch mã chữ-số cho dataset (v2 → vê hai, MP3 → em pê ba)"),
    overrides: str | None = typer.Option(None, help="File JSON 'mặt chữ: cách đọc' cho từ lạ (OOV)"),
    license: str = typer.Option("CC BY 4.0", help="Giấy phép mở ghi vào dataset card"),
    sleep: float = typer.Option(1.0, help="Nghỉ giữa các request (giây)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Chỉ liệt kê video, không tải"),
    recut: bool = typer.Option(False, "--recut",
                               help="Xoá kết quả cũ, cắt lại từ raw/ + subs/ (không tải lại)"),
    refresh: bool = typer.Option(False, help="Bỏ cache danh sách video (.videos.json)"),
    ytdlp_bin: str | None = typer.Option(None, "--ytdlp-bin", hidden=True,
                                          help="Lệnh yt-dlp thay thế (test/đường dẫn riêng)"),
    ffmpeg_bin: str | None = typer.Option(None, "--ffmpeg-bin", hidden=True,
                                          help="Đường dẫn ffmpeg thay thế"),
) -> None:
    """Tải audio + phụ đề kênh YouTube của bạn → clip sạch + manifest + SRT mở.

    Chi tiết luồng và giấy phép dữ liệu: `docs/YOUTUBE_DATASET.md`.
    """
    import shlex

    from vifasttts.data.youtube import FetchOptions, fetch_dataset

    overrides_map: dict[str, str] = {}
    if overrides:
        payload = json.loads(Path(overrides).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
        ):
            raise typer.BadParameter('File overrides phải là JSON dạng {"mặt chữ": "cách đọc"}')
        overrides_map = payload

    fetch_dataset(FetchOptions(
        channel=channel, out=Path(out), limit=limit, demo=demo,
        langs=tuple(lang.strip() for lang in langs.split(",") if lang.strip()),
        dialect=dialect, min_duration=min_duration, max_duration=max_duration,
        max_gap=max_gap, max_chars=max_chars,
        min_video_duration=min_video_duration, max_video_duration=max_video_duration,
        title_filter=title_filter, keep_wav=keep_wav, dedupe=dedupe,
        min_rms_db=min_rms_db, max_fallback_ratio=max_fallback_ratio,
        min_caption_density=min_caption_density, overrides=overrides_map,
        overrides_source=overrides, respell_codes=respell_codes,
        license=license, sleep=sleep, dry_run=dry_run, refresh=refresh,
        recut=recut,
        ytdlp=shlex.split(ytdlp_bin) if ytdlp_bin else None, ffmpeg=ffmpeg_bin,
    ))


@app.command("train")
def train(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    train_manifest: str = typer.Option(..., help="Manifest train (JSONL)"),
    val_manifest: str = typer.Option(..., help="Manifest validation (JSONL)"),
    output: str = typer.Option(..., help="Thư mục run"),
    max_steps: int | None = typer.Option(None, help="Ghi đè số bước tối đa (chạy thử nhanh)"),
    resume: str | None = typer.Option(None, help="Checkpoint .pt để tiếp tục (resume)"),
) -> None:
    """Huấn luyện acoustic model (device lấy từ `training.device`, auto = cuda→mps→cpu)."""
    train_reference(load_config(config), train_manifest, val_manifest, output,
                    max_steps=max_steps, resume=resume)


@app.command("train-v2")
def train_v2(
    config: str = typer.Option(..., help="File YAML cấu hình (v2: phonation + loss_weights v2)"),
    train_manifest: str = typer.Option(..., help="Manifest train `_slot4` (JSONL)"),
    val_manifest: str = typer.Option(..., help="Manifest validation `_slot4` (JSONL)"),
    output: str = typer.Option(..., help="Thư mục run"),
    warm_start: str | None = typer.Option(None, help="Checkpoint v1 (B0) để warm-start (§11)"),
    max_steps: int | None = typer.Option(None, help="Ghi đè số bước tối đa (chạy thử nhanh)"),
    resume: str | None = typer.Option(None, help="Checkpoint v2 .pt để tiếp tục (resume)"),
    reset_val_state: bool = typer.Option(False, "--reset-val-state",
                                         help="X3: resume bằng tập val KHÁC ⇒ reset best_val/no_improve"),
    device: str | None = typer.Option(None, help="Ghi đè device (auto|cpu|mps|cuda)"),
    adversarial: bool | None = typer.Option(None, "--adversarial/--no-adversarial",
                                            help="T4.1: bật/tắt mel discriminator (train-only)"),
) -> None:
    """Huấn luyện acoustic v2 (`v2_slot5`) — warm-start từ B0, giai đoạn A/B (docs/ACOUSTIC_V2.md §11)."""
    from vifasttts.train_v2 import train_slot_v2

    train_slot_v2(load_config(config), train_manifest, val_manifest, output,
                  warm_start=warm_start, max_steps=max_steps, resume=resume,
                  reset_val_state=reset_val_state, device=device,
                  adversarial_flag=adversarial)


@app.command("export-onnx")
def export_onnx_cmd(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    checkpoint: str = typer.Option(..., help="Checkpoint .pt"),
    output: str = typer.Option(..., help="File .onnx đầu ra"),
    opset: int = typer.Option(18, help="Opset ONNX (mặc định 18)"),
    precision: str = typer.Option("fp32", help="fp32 | fp16 | int8"),
) -> None:
    """Xuất ONNX acoustic (fp32/fp16/int8) với batch/độ dài động + metadata."""
    from vifasttts.export import export_onnx
    export_onnx(load_config(config), checkpoint, output, opset_version=opset, precision=precision)
    typer.echo(f"Đã xuất {output} ({precision})")


@app.command("export-vocoder-onnx")
def export_vocoder_onnx_cmd(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    checkpoint: str = typer.Option(..., help="Checkpoint vocoder .pt"),
    output: str = typer.Option(..., help="File .onnx đầu ra (mel → phổ)"),
    opset: int = typer.Option(18, help="Opset ONNX (mặc định 18)"),
    precision: str = typer.Option("fp32", help="fp32 | fp16 (so fp16 với fp32 trước khi dùng)"),
) -> None:
    """Xuất vocoder ONNX (mel → spec_real/spec_imag) để chạy `synthesize-onnx`."""
    from vifasttts.export import export_vocoder_onnx
    export_vocoder_onnx(load_config(config), checkpoint, output, opset_version=opset, precision=precision)
    typer.echo(f"Đã xuất {output} (vocoder, {precision})")


@app.command("check-profile")
def check_profile_cmd(
    text: str = typer.Argument(..., help="Câu cần kiểm tra"),
    profile: str = typer.Option("strict", help="open | strict | corpus"),
    lexicon: str | None = typer.Option(None, help="File lexicon .json (bắt buộc cho profile corpus)"),
) -> None:
    """Kiểm chính tả/ngữ âm theo profile; in mã cảnh báo cho từng âm tiết."""
    from vifasttts.frontend.lexicon import ISSUE_DESCRIPTIONS, Lexicon, check_syllable

    lex = Lexicon.load(lexicon) if lexicon else None
    items = parse_text(text)
    flagged = 0
    for item in items:
        issues = check_syllable(item, profile, lex)  # type: ignore[arg-type]
        if issues:
            flagged += 1
            readable = "; ".join(ISSUE_DESCRIPTIONS.get(code, code) for code in issues)
            typer.echo(f"  {item.raw}: {readable} [{', '.join(issues)}]")
    typer.echo(f"{len(items)} âm tiết, {flagged} cảnh báo (profile={profile})")


@app.command("build-lexicon")
def build_lexicon_cmd(
    output: str = typer.Option(..., help="File JSON đầu ra"),
    manifest: str | None = typer.Option(None, help="Manifest JSONL nguồn"),
    wordlist: str | None = typer.Option(None, help="File 1 từ/dòng (nguồn thay thế)"),
) -> None:
    """Sinh lexicon shape từ manifest hoặc wordlist cho profile corpus."""
    import re

    from vifasttts.frontend.lexicon import Lexicon

    if bool(manifest) == bool(wordlist):
        raise typer.BadParameter("Chọn đúng một trong --manifest hoặc --wordlist")
    if manifest:
        lex = Lexicon.from_manifest(manifest)
    elif wordlist is not None:
        words: list[str] = []
        word_re = re.compile(r"[A-Za-zÀ-ỹĐđ]+")
        with open(wordlist, encoding="utf-8") as handle:
            for line in handle:
                words.extend(token for token in re.split(r"[\s\-]+", line.strip()) if word_re.fullmatch(token))
        lex = Lexicon.from_words(words)
    else:  # pragma: no cover - đã chặn ở trên
        raise typer.BadParameter("Thiếu nguồn --manifest/--wordlist")
    lex.save(output)
    typer.echo(f"Đã ghi {output} ({len(lex.shapes)} shape, {len(lex.rimes)} vần)")


@app.command("cache-features")
def cache_features_cmd(
    manifest: str = typer.Option(..., help="Manifest JSONL"),
    config: str = typer.Option(..., help="File YAML cấu hình"),
    cache_dir: str = typer.Option(..., help="Thư mục cache"),
    limit: int | None = typer.Option(None, help="Giới hạn số mẫu (chạy thử)"),
    workers: int = typer.Option(0, help="Số tiến trình song song (0 = tuần tự)"),
) -> None:
    """Tính trước mel/f0/vuv cho manifest để training nhanh hơn."""
    from vifasttts.data.cache import precompute_features

    done = precompute_features(manifest, load_config(config), cache_dir, limit=limit, workers=workers)
    typer.echo(f"Đã cache {done} mẫu vào {cache_dir}")


@app.command("train-vocoder")
def train_vocoder_cmd(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    manifest: str = typer.Option(..., help="Manifest (JSONL)"),
    output: str = typer.Option(..., help="Thư mục run"),
    max_steps: int | None = typer.Option(None, help="Ghi đè số bước tối đa"),
    vocoder_checkpoint: str | None = typer.Option(None, help="Checkpoint vocoder để huấn luyện tiếp"),
    resume: str | None = typer.Option(None, help="Checkpoint .pt để tiếp tục (resume)"),
    seed: int | None = typer.Option(None, help="Ghi đè seed (đổi thứ tự batch sau khi rollback NaN)"),
    lr_scale: float = typer.Option(1.0, help="Nhân learning rate sau resume (ví dụ 0.5 khi NaN lặp lại)"),
) -> None:
    """Huấn luyện vocoder (generator lite/vocos + MPD/MRD)."""
    from vifasttts.train import train_vocoder_reference

    train_vocoder_reference(
        load_config(config), manifest, output,
        max_steps=max_steps, vocoder_checkpoint=vocoder_checkpoint, resume=resume,
        seed=seed, lr_scale=lr_scale,
    )
    typer.echo(f"Đã lưu vocoder vào {output}")


@app.command("synthesize")
def synthesize_cmd(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    checkpoint: str = typer.Option(..., help="Checkpoint .pt"),
    text: str = typer.Option(..., help="Câu cần đọc"),
    output: str = typer.Option(..., help="File WAV đầu ra"),
    dialect: str = typer.Option("north", help="north, central hoặc south"),
    speed: float = typer.Option(1.0, help="Tốc độ đọc (1.0 là gốc)"),
    vocoder_checkpoint: str | None = typer.Option(None, help="Checkpoint vocoder (tùy chọn)"),
    overrides: str | None = typer.Option(None, help="File JSON 'mặt chữ: cách đọc'"),
    peak_normalize: bool = typer.Option(False, "--peak-normalize", help="Chuẩn hoá đỉnh -3 dBFS"),
    max_chars: int = typer.Option(400, help="Độ dài tối đa mỗi đoạn (chunk theo câu)"),
    device: str = typer.Option("auto", help="auto | cpu | cuda | mps"),
) -> None:
    """Tổng hợp WAV bằng PyTorch (chunk câu dài, override, chuẩn hoá đỉnh tuỳ chọn)."""
    from vifasttts.infer import load_overrides, synthesize
    samples, sample_rate = synthesize(
        load_config(config), checkpoint, text, output,
        dialect=dialect, speed=speed, vocoder_checkpoint=vocoder_checkpoint,
        overrides=load_overrides(overrides) if overrides else None,
        peak_normalize=peak_normalize, max_chars=max_chars, device=device,
    )
    typer.echo(f"Đã ghi {output} ({samples} mẫu @ {sample_rate} Hz)")


@app.command("synthesize-onnx")
def synthesize_onnx_cmd(
    acoustic: str = typer.Option(..., help="File model.onnx (acoustic)"),
    vocoder: str = typer.Option(..., help="File vocoder.onnx (mel → phổ)"),
    text: str = typer.Option(..., help="Câu cần đọc"),
    output: str = typer.Option(..., help="File WAV đầu ra"),
    dialect: str = typer.Option("north", help="north, central hoặc south"),
    speed: float = typer.Option(1.0, help="Tốc độ đọc (xấp xỉ bằng co giãn mel)"),
    overrides: str | None = typer.Option(None, help="File JSON 'mặt chữ: cách đọc'"),
    peak_normalize: bool = typer.Option(False, "--peak-normalize", help="Chuẩn hoá đỉnh -3 dBFS"),
    max_chars: int = typer.Option(400, help="Độ dài tối đa mỗi đoạn (chunk theo câu)"),
    provider: str = typer.Option("auto", help="auto | cpu | cuda | coreml | tensorrt"),
) -> None:
    """Tổng hợp bằng ONNX Runtime + numpy (không cần PyTorch)."""
    from vifasttts.infer import load_overrides
    from vifasttts.runtime.onnx_engine import synthesize_onnx

    samples, sample_rate = synthesize_onnx(
        text, acoustic, output, vocoder_path=vocoder,
        dialect=dialect, speed=speed,
        overrides=load_overrides(overrides) if overrides else None,
        peak_normalize=peak_normalize, max_chars=max_chars, provider=provider,
    )
    typer.echo(f"Đã ghi {output} ({samples} mẫu @ {sample_rate} Hz)")


@app.command("synthesize-batch")
def synthesize_batch_cmd(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    checkpoint: str = typer.Option(..., help="Checkpoint .pt"),
    input: str = typer.Option(..., help="File văn bản: 'id<TAB>câu' hoặc mỗi dòng một câu"),
    output_dir: str = typer.Option(..., "--output-dir", help="Thư mục WAV đầu ra"),
    vocoder_checkpoint: str | None = typer.Option(None, help="Checkpoint vocoder (tùy chọn)"),
    dialect: str = typer.Option("north", help="north, central hoặc south"),
    speed: float = typer.Option(1.0, help="Tốc độ đọc (1.0 là gốc)"),
    overrides: str | None = typer.Option(None, help="File JSON 'mặt chữ: cách đọc'"),
    peak_normalize: bool = typer.Option(False, "--peak-normalize", help="Chuẩn hoá đỉnh -3 dBFS"),
    max_chars: int = typer.Option(400, help="Độ dài tối đa mỗi đoạn (chunk theo câu)"),
    device: str = typer.Option("auto", help="auto | cpu | cuda | mps"),
) -> None:
    """Tổng hợp nhiều câu từ file → thư mục WAV (nạp model một lần)."""
    from vifasttts.infer import load_overrides, synthesize_batch

    rows: list[tuple[str, str]] = []
    for line_number, raw in enumerate(Path(input).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if "\t" in line:
            sample_id, text = line.split("\t", 1)
        else:
            sample_id, text = f"line_{line_number:04d}", line
        rows.append((sample_id.strip(), text.strip()))
    if not rows:
        raise typer.BadParameter("File input không có dòng hợp lệ")
    paths = synthesize_batch(
        load_config(config), checkpoint, rows, output_dir,
        vocoder_checkpoint=vocoder_checkpoint, dialect=dialect, speed=speed,
        overrides=load_overrides(overrides) if overrides else None,
        peak_normalize=peak_normalize, max_chars=max_chars, device=device,
    )
    typer.echo(f"Đã ghi {len(paths)} file vào {output_dir}")


@app.command("serve")
def serve_cmd(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    checkpoint: str = typer.Option(..., help="Checkpoint .pt"),
    vocoder_checkpoint: str | None = typer.Option(None, help="Checkpoint vocoder (tùy chọn)"),
    host: str = typer.Option("127.0.0.1", help="Địa chỉ lắng nghe"),
    port: int = typer.Option(8000, help="Cổng"),
    device: str = typer.Option("auto", help="auto | cpu | cuda | mps"),
) -> None:
    """HTTP server tối giản: POST /synthesize (JSON → WAV), GET /health, /info."""
    from vifasttts.server import serve

    serve(load_config(config), checkpoint, vocoder_checkpoint=vocoder_checkpoint,
          host=host, port=port, device=device)


if __name__ == "__main__":
    app()

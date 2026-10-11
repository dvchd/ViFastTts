from __future__ import annotations

import json

import typer

from vifasttts import __version__
from vifasttts.config import load_config
from vifasttts.data.manifest import build_manifest
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.syllable import parse_syllable
from vifasttts.train import train_reference

app = typer.Typer(help="ViFastTts CLI")


def _version_callback(value: bool) -> None:
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
    typer.echo(json.dumps(parse_syllable(value).to_dict(), ensure_ascii=False, indent=2, default=str))


@app.command("parse")
def parse(value: str, dialect: str = "north") -> None:
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
    build_manifest(metadata, wav_dir, output)
    typer.echo(f"Đã ghi {output}")


@app.command("validate-manifest")
def validate_manifest_cmd(
    manifest: str = typer.Option(..., help="File JSONL cần kiểm tra"),
    wav_dir: str | None = typer.Option(None, help="Thư mục WAV (nếu manifest chỉ có id)"),
) -> None:
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


@app.command("train")
def train(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    train_manifest: str = typer.Option(..., help="Manifest train (JSONL)"),
    val_manifest: str = typer.Option(..., help="Manifest validation (JSONL)"),
    output: str = typer.Option(..., help="Thư mục run"),
    max_steps: int | None = typer.Option(None, help="Ghi đè số bước tối đa (chạy thử nhanh)"),
) -> None:
    train_reference(load_config(config), train_manifest, val_manifest, output, max_steps=max_steps)


@app.command("export-onnx")
def export_onnx_cmd(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    checkpoint: str = typer.Option(..., help="Checkpoint .pt"),
    output: str = typer.Option(..., help="File .onnx đầu ra"),
    opset: int = typer.Option(18, help="Opset ONNX (mặc định 18)"),
    precision: str = typer.Option("fp32", help="fp32 | fp16 | int8"),
) -> None:
    from vifasttts.export import export_onnx
    export_onnx(load_config(config), checkpoint, output, opset_version=opset, precision=precision)
    typer.echo(f"Đã xuất {output} ({precision})")


@app.command("export-vocoder-onnx")
def export_vocoder_onnx_cmd(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    checkpoint: str = typer.Option(..., help="Checkpoint vocoder .pt"),
    output: str = typer.Option(..., help="File .onnx đầu ra (mel → phổ)"),
    opset: int = typer.Option(18, help="Opset ONNX (mặc định 18)"),
) -> None:
    """Xuất vocoder ONNX (mel → spec_real/spec_imag) để chạy `synthesize-onnx`."""
    from vifasttts.export import export_vocoder_onnx
    export_vocoder_onnx(load_config(config), checkpoint, output, opset_version=opset)
    typer.echo(f"Đã xuất {output} (vocoder)")


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
) -> None:
    """Tính trước mel/f0/vuv cho manifest để training nhanh hơn."""
    from vifasttts.data.cache import precompute_features

    done = precompute_features(manifest, load_config(config), cache_dir, limit=limit)
    typer.echo(f"Đã cache {done} mẫu vào {cache_dir}")


@app.command("train-vocoder")
def train_vocoder_cmd(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    manifest: str = typer.Option(..., help="Manifest (JSONL)"),
    output: str = typer.Option(..., help="Thư mục run"),
    max_steps: int | None = typer.Option(None, help="Ghi đè số bước tối đa"),
    vocoder_checkpoint: str | None = typer.Option(None, help="Checkpoint vocoder để huấn luyện tiếp"),
) -> None:
    """Huấn luyện vocoder tham chiếu (iSTFTNetLite + MPD-lite)."""
    from vifasttts.train import train_vocoder_reference

    train_vocoder_reference(
        load_config(config), manifest, output,
        max_steps=max_steps, vocoder_checkpoint=vocoder_checkpoint,
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
) -> None:
    from vifasttts.infer import load_overrides, synthesize
    samples, sample_rate = synthesize(
        load_config(config), checkpoint, text, output,
        dialect=dialect, speed=speed, vocoder_checkpoint=vocoder_checkpoint,
        overrides=load_overrides(overrides) if overrides else None,
        peak_normalize=peak_normalize, max_chars=max_chars,
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
) -> None:
    """Tổng hợp bằng ONNX Runtime + numpy (không cần PyTorch)."""
    from vifasttts.infer import load_overrides
    from vifasttts.runtime.onnx_engine import synthesize_onnx

    samples, sample_rate = synthesize_onnx(
        text, acoustic, output, vocoder_path=vocoder,
        dialect=dialect, speed=speed,
        overrides=load_overrides(overrides) if overrides else None,
        peak_normalize=peak_normalize, max_chars=max_chars,
    )
    typer.echo(f"Đã ghi {output} ({samples} mẫu @ {sample_rate} Hz)")


if __name__ == "__main__":
    app()

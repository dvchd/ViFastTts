from __future__ import annotations

import json

import typer

from vifasttts.config import load_config
from vifasttts.data.manifest import build_manifest
from vifasttts.frontend.pipeline import parse_text
from vifasttts.frontend.syllable import parse_syllable
from vifasttts.train import train_reference

app = typer.Typer(help="ViFastTts CLI")


@app.command("parse-syllable")
def parse_one(value: str) -> None:
    typer.echo(json.dumps(parse_syllable(value).to_dict(), ensure_ascii=False, indent=2, default=str))


@app.command("parse")
def parse(value: str, dialect: str = "north") -> None:
    typer.echo(json.dumps([x.to_dict() for x in parse_text(value, dialect=dialect)], ensure_ascii=False, indent=2, default=str))


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
    from vifasttts.data.manifest import validate_manifest
    report = validate_manifest(manifest, wav_dir)
    typer.echo(f"Tổng {report['total']} dòng, {len(report['issues'])} vấn đề")
    for issue in report["issues"][:20]:
        typer.echo(f"  dòng {issue['line']}: {issue['id']} - {issue['issue']} ({issue.get('path', '')})")
    if len(report["issues"]) > 20:
        typer.echo(f"  ... và {len(report['issues']) - 20} vấn đề khác")


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
) -> None:
    from vifasttts.export import export_onnx
    export_onnx(load_config(config), checkpoint, output, opset_version=opset)
    typer.echo(f"Đã xuất {output}")


@app.command("synthesize")
def synthesize_cmd(
    config: str = typer.Option(..., help="File YAML cấu hình"),
    checkpoint: str = typer.Option(..., help="Checkpoint .pt"),
    text: str = typer.Option(..., help="Câu cần đọc"),
    output: str = typer.Option(..., help="File WAV đầu ra"),
    dialect: str = typer.Option("north", help="north, central hoặc south"),
    speed: float = typer.Option(1.0, help="Tốc độ đọc (1.0 là gốc)"),
    vocoder_checkpoint: str | None = typer.Option(None, help="Checkpoint vocoder (tùy chọn)"),
) -> None:
    from vifasttts.infer import synthesize
    samples, sample_rate = synthesize(
        load_config(config), checkpoint, text, output,
        dialect=dialect, speed=speed, vocoder_checkpoint=vocoder_checkpoint,
    )
    typer.echo(f"Đã ghi {output} ({samples} mẫu @ {sample_rate} Hz)")


if __name__ == "__main__":
    app()

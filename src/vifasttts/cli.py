from __future__ import annotations

import json
from pathlib import Path
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
def parse(value: str) -> None:
    typer.echo(json.dumps([x.to_dict() for x in parse_text(value)], ensure_ascii=False, indent=2, default=str))


@app.command("build-manifest")
def manifest(metadata: str, wav_dir: str, output: str) -> None:
    build_manifest(metadata, wav_dir, output)
    typer.echo(f"Đã ghi {output}")


@app.command("train")
def train(config: str, train_manifest: str, val_manifest: str, output: str) -> None:
    train_reference(load_config(config), train_manifest, val_manifest, output)


@app.command("export-onnx")
def export_onnx_cmd(config: str, checkpoint: str, output: str) -> None:
    from vifasttts.export import export_onnx
    export_onnx(load_config(config), checkpoint, output)
    typer.echo(f"Đã xuất {output}")


if __name__ == "__main__":
    app()

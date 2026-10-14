"""V09.B3: watchdog dựng lệnh MỘT LẦN (mảng) — khởi động đầu vs khởi động lại vs dry-run."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _fake(tmp_path: Path) -> tuple[Path, Path]:
    fake = tmp_path / "fake_train.sh"
    args_file = tmp_path / "args.txt"
    fake.write_text(f'#!/bin/bash\necho "$@" >> "{args_file}"\nexit 0\n', encoding="utf-8")
    fake.chmod(0o755)
    return fake, args_file


def _run(tmp_path: Path, out: Path, fake: Path, *, dry: bool = False, init: bool = True) -> str:
    env = {**os.environ, "TRAIN_CMD": str(fake), "WATCH_MAX_ATTEMPTS": "1"}
    if init:
        env["INIT_RESUME"] = "runs/acoustic_v2_base/last.pt"
        env["INIT_RESET_VAL_STATE"] = "1"
    if dry:
        env["DRY_RUN"] = "1"
    proc = subprocess.run(
        ["bash", str(ROOT / "tools/watch_train_v2.sh"), str(out), str(tmp_path / "log.txt"),
         "60000", "cfg.yaml", "train.jsonl", "val.jsonl"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def test_watchdog_lan_dau_resume_ngoai_va_reset_val(tmp_path):
    """(a) lần khởi động đầu: có --resume <INIT_RESUME> + --reset-val-state."""
    fake, args_file = _fake(tmp_path)
    _run(tmp_path, tmp_path / "run_a", fake)
    args = args_file.read_text(encoding="utf-8")
    assert "--resume runs/acoustic_v2_base/last.pt" in args
    assert "--reset-val-state" in args


def test_watchdog_khoi_dong_lai_khong_reset_val(tmp_path):
    """(b) đã có OUT/last.pt (step>0): resume OUT/last.pt, KHÔNG --reset-val-state."""
    fake, args_file = _fake(tmp_path)
    out = tmp_path / "run_b"
    out.mkdir()
    torch.save({"step": 100}, out / "last.pt")
    _run(tmp_path, out, fake)
    args = args_file.read_text(encoding="utf-8")
    assert f"--resume {out}/last.pt" in args
    assert "--reset-val-state" not in args


def test_watchdog_dry_run_trung_lenh_that(tmp_path):
    """(c) đầu ra DRY_RUN trùng từng ký tự với đối số script giả nhận ở lần khởi động đầu."""
    fake, args_file = _fake(tmp_path)
    out = tmp_path / "run_c"          # CÙNG thư mục OUT cho cả hai lần (dry-run không ghi gì)
    _run(tmp_path, out, fake)
    real_args = args_file.read_text(encoding="utf-8").strip()
    dry = _run(tmp_path, out, fake, dry=True)
    # dry in CẢ lệnh (gồm TRAIN_CMD); script giả nhận phần đối số ⇒ dry == "<TRAIN_CMD> <đối số>"
    assert dry == f"{fake} {real_args}"

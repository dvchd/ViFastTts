"""Tải checkpoint Vocos pretrained (MIT) về `artifacts/pretrained/vocos-mel-24khz/`.

Chạy:
    python tools/fetch_vocos_pretrained.py

- Ghim đúng revision HF (không phụ thuộc `main` đổi sau này) và ghi `manifest.json`
  (revision + sha256 từng file + license) để truy vết — xem `docs/VOCODER_VOCOS.md` §1.
- File `pytorch_model.bin` phải khớp `pretrained_sha256` ghi trong config fine-tune;
  bước train sẽ tự kiểm (train.py::`_pretrained_sha256`).
- Cần `huggingface_hub` (`pip install huggingface_hub`); file nằm ngoài git (`artifacts/`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

#: Nguồn pretrained — giấy phép MIT (xem LICENSES.md).
REPO_ID = "charactr/vocos-mel-24khz"
REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
FILES = ("config.yaml", "pytorch_model.bin")
#: sha256 của bản đã đo trong repo (R7-1, 12/10/2026) — lệch thì cảnh báo (có thể HF đổi file).
EXPECTED_SHA256 = {
    "pytorch_model.bin": "97ec976ad1fd67a33ab2682d29c0ac7df85234fae875aefcc5fb215681a91b2a",
}


def main() -> None:
    """Tải + xác minh sha256 + ghi manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/pretrained/vocos-mel-24khz",
                        help="Thư mục đích (mặc định artifacts/pretrained/vocos-mel-24khz)")
    parser.add_argument("--revision", default=REVISION, help="Revision HF (mặc định ghim của repo)")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise SystemExit("Cần huggingface_hub: pip install huggingface_hub") from exc

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "repo": REPO_ID,
        "revision": args.revision,
        "license": "MIT",
        "files": {},
    }
    for name in FILES:
        local = Path(hf_hub_download(REPO_ID, name, revision=args.revision, local_dir=str(out)))
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        size_mb = local.stat().st_size / 1e6
        manifest["files"][name] = {"sha256": digest, "bytes": local.stat().st_size}  # type: ignore[index]
        print(f"  {name}: sha256={digest[:12]}… ({size_mb:.1f} MB)")
        expected = EXPECTED_SHA256.get(name)
        if expected and expected != digest:
            print(f"  ⚠️  {name}: sha256 KHÁC bản đã đo ({expected[:12]}…) — kiểm tra revision/HF trước khi dùng!")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã lưu vào {out} (revision {args.revision[:12]}…, ghi manifest.json)")


if __name__ == "__main__":
    main()

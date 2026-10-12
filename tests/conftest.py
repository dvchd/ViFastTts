"""Cấu hình pytest: cho phép import các tool trong `tools/` theo kiểu script (top-level).

Các tool viết để chạy trực tiếp (`python tools/xxx.py`) import lẫn nhau bằng tên trần
(ví dụ `from mfa_to_durations import ...`). Khi test import `tools.xxx`, cần `tools/`
nằm trên `sys.path` để các import đó phân giải được.
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

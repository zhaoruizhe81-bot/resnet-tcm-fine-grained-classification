from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root, src

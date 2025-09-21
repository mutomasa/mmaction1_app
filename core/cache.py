from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_CACHE_DIR = Path(os.getenv("MMACTION2_APP_CACHE", Path.home() / ".cache" / "mmaction2_app"))


def get_cache_dir(name: Optional[str] = None) -> Path:
    base = DEFAULT_CACHE_DIR
    if name:
        base = base / name
    base.mkdir(parents=True, exist_ok=True)
    return base


def resolve_temp_dir() -> Path:
    return get_cache_dir("temp")

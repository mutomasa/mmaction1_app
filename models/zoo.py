from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import requests

from core.cache import get_cache_dir

LOGGER = logging.getLogger(__name__)

REMOTE_SCHEMES = ("http://", "https://")
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1] / "model_store"
MODEL_ROOT = Path(
    os.getenv("MMACTION2_APP_MODEL_DIR", str(DEFAULT_MODEL_ROOT))
).expanduser()


def _ensure_tuple(value: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class ModelEntry:
    display_name: str
    config: str | Sequence[str]
    checkpoint: str | Sequence[str]
    label_map: str | Sequence[str]
    input_format: str = "RGB"
    description: str | None = None
    mim_config: str | None = None

    def config_candidates(self) -> Tuple[str, ...]:
        return _ensure_tuple(self.config)

    def checkpoint_candidates(self) -> Tuple[str, ...]:
        return _ensure_tuple(self.checkpoint)

    def label_map_candidates(self) -> Tuple[str, ...]:
        return _ensure_tuple(self.label_map)


ZOO: Dict[str, ModelEntry] = {
    "tsn_r50_kinetics400": ModelEntry(
        display_name="TSN-R50 (Kinetics-400)",
        config=[
            "configs/mmaction/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py",
            "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py",
        ],
        checkpoint=[
            "checkpoints/mmaction/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth",
            "https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth",
        ],
        label_map=[
            "labels/mmaction/label_map_k400.txt",
            "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/tools/data/kinetics/label_map_k400.txt",
        ],
        description="Baseline RGB recognizer suitable for demos.",
        mim_config="tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb",
    ),
    "slowfast_r50_kinetics400": ModelEntry(
        display_name="SlowFast-R50 (Kinetics-400)",
        config=[
            "configs/mmaction/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py",
            "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py",
        ],
        checkpoint=[
            "checkpoints/mmaction/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth",
            "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth",
        ],
        label_map=[
            "labels/mmaction/label_map_k400.txt",
            "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/tools/data/kinetics/label_map_k400.txt",
        ],
        description="Higher quality video model; needs more resources.",
        mim_config="slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb",
    ),
}


def available_models() -> Iterable[str]:
    return ZOO.keys()


def get_model_entry(key: str) -> ModelEntry:
    try:
        return ZOO[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model key: {key}") from exc


def _hashed_filename(url: str) -> str:
    digest = hashlib.md5(url.encode("utf-8"), usedforsecurity=False).hexdigest()
    suffix = Path(url).suffix or ".bin"
    return f"{digest}{suffix}"


def ensure_remote_file(url: str, cache_subdir: str = "downloads") -> Path:
    cache_dir = get_cache_dir(cache_subdir)
    target = cache_dir / _hashed_filename(url)
    if target.exists():
        return target
    LOGGER.info("Downloading %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def _resolve_local_candidate(candidate: str, cache_subdir: str) -> Path | None:
    if candidate.startswith(REMOTE_SCHEMES):
        try:
            return ensure_remote_file(candidate, cache_subdir=cache_subdir)
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("Failed to fetch %s: %s", candidate, exc)
            return None
    path = Path(candidate)
    if not path.is_absolute():
        path = MODEL_ROOT / path
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.exists():
        return path
    LOGGER.debug("Candidate %s was not found", path)
    return None


def ensure_local_resource(resource: Sequence[str] | str, *, cache_subdir: str) -> Path:
    for candidate in _ensure_tuple(resource):
        resolved = _resolve_local_candidate(candidate, cache_subdir=cache_subdir)
        if resolved:
            return resolved
    candidates = list(_ensure_tuple(resource))
    raise FileNotFoundError(
        "Could not resolve resource from candidates: "
        f"{candidates}. Provide a local copy or set MMACTION2_APP_MODEL_DIR."
    )


def resolve_model_files(entry: ModelEntry) -> Tuple[Path, Path]:
    config_path = ensure_local_resource(entry.config_candidates(), cache_subdir="configs")
    checkpoint_path = ensure_local_resource(entry.checkpoint_candidates(), cache_subdir="checkpoints")
    return config_path, checkpoint_path


def load_label_map(entry: ModelEntry) -> List[str]:
    label_path = ensure_local_resource(entry.label_map_candidates(), cache_subdir="labels")
    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise RuntimeError(f"Label map at {label_path} is empty")
    return labels


__all__ = [
    "ModelEntry",
    "ZOO",
    "available_models",
    "get_model_entry",
    "load_label_map",
    "resolve_model_files",
    "MODEL_ROOT",
]

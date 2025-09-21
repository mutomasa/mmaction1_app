#!/usr/bin/env python3
"""Download MMAction2 configs/checkpoints into the local model store."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import requests

from models.zoo import MODEL_ROOT, ModelEntry, available_models, get_model_entry

REMOTE_SCHEMES = ("http://", "https://")
CONFIG_DIR = MODEL_ROOT / "configs" / "mmaction"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints" / "mmaction"
LABEL_DIR = MODEL_ROOT / "labels" / "mmaction"
TEMP_DIR = MODEL_ROOT / "downloads"
CHUNK_SIZE = 1 << 15  # 32KB


def ensure_dirs() -> None:
    for directory in (CONFIG_DIR, CHECKPOINT_DIR, LABEL_DIR, TEMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def primary_local_path(entry: ModelEntry, kind: str) -> Path:
    candidates = getattr(entry, f"{kind}_candidates")()
    if not candidates:
        raise ValueError(f"No {kind} candidates defined for {entry.display_name}")
    candidate = candidates[0]
    path = Path(candidate)
    if not path.is_absolute():
        path = MODEL_ROOT / path
    return path


def download_via_mim(entry: ModelEntry, overwrite: bool) -> bool:
    if not entry.mim_config:
        return False
    config_target = primary_local_path(entry, "config")
    checkpoint_target = primary_local_path(entry, "checkpoint")
    if config_target.exists() and checkpoint_target.exists() and not overwrite:
        return False

    dest = TEMP_DIR / entry.mim_config
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mim",
        "download",
        "mmaction2",
        "--config",
        entry.mim_config,
        "--dest",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"    [FAIL] mim download failed: {exc}")
        return False

    moved = False
    for source in dest.glob("*.py"):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(config_target))
        moved = True
    for source in dest.glob("*.pth"):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(checkpoint_target))
        moved = True
    shutil.rmtree(dest, ignore_errors=True)
    return moved


def download_file(url: str, dest: Path, overwrite: bool) -> bool:
    if dest.exists() and not overwrite:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
    return True


def ensure_label_map(entry: ModelEntry, overwrite: bool) -> None:
    target = primary_local_path(entry, "label_map")
    if target.exists() and not overwrite:
        return
    for candidate in entry.label_map_candidates():
        if candidate.startswith(REMOTE_SCHEMES):
            try:
                if download_file(candidate, target, overwrite=True):
                    return
            except Exception as exc:
                print(f"    [WARN] label map fetch failed: {exc}")
        else:
            src = MODEL_ROOT / candidate
            if src.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                return
    try:
        import mmaction  # type: ignore

        pkg_root = Path(mmaction.__file__).resolve().parent / ".mim" / "tools" / "data"
        fallback = pkg_root / "kinetics" / "label_map_k400.txt"
        if fallback.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback, target)
            return
    except ImportError:
        pass
    raise FileNotFoundError(f"Could not obtain label map for {entry.display_name}")


def download_model(entry: ModelEntry, overwrite: bool) -> None:
    print(f"==> {entry.display_name}")
    used_mim = download_via_mim(entry, overwrite)
    if not used_mim:
        for url in entry.config_candidates():
            if url.startswith(REMOTE_SCHEMES):
                config_target = primary_local_path(entry, "config")
                try:
                    if download_file(url, config_target, overwrite):
                        break
                except Exception as exc:
                    print(f"    [WARN] config fetch failed: {exc}")
    config_path = primary_local_path(entry, "config")
    checkpoint_path = primary_local_path(entry, "checkpoint")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not checkpoint_path.exists():
        for url in entry.checkpoint_candidates():
            if url.startswith(REMOTE_SCHEMES):
                try:
                    download_file(url, checkpoint_path, overwrite)
                    break
                except Exception as exc:
                    print(f"    [WARN] checkpoint fetch failed: {exc}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    ensure_label_map(entry, overwrite)
    print(f"    [ready] {config_path.name}, {checkpoint_path.name}")


def iter_models(args_models: Iterable[str]) -> Iterable[str]:
    if list(args_models) == ["all"]:
        return list(available_models())
    return args_models


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=["all"], help="Model keys to download")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args(argv)

    ensure_dirs()

    for key in iter_models(args.models):
        try:
            entry = get_model_entry(key)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue
        try:
            download_model(entry, overwrite=args.overwrite)
        except Exception as exc:
            print(f"    [FAIL] {exc}")
    print(f"Artifacts stored under {MODEL_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

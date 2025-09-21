from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency during tests
    cv2 = None  # type: ignore


@dataclass
class VideoMetadata:
    path: Path
    frame_count: int
    fps: float

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def probe_video(path: str | Path) -> VideoMetadata:
    if cv2 is None:
        raise ImportError("opencv-python is required for video probing")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    return VideoMetadata(path=Path(path), frame_count=frame_count, fps=fps)


def compute_sample_indices(frame_count: int, fps: float, sample_fps: int) -> List[int]:
    if frame_count <= 0:
        return []
    if sample_fps <= 0 or fps <= 0:
        return list(range(frame_count))
    step = max(int(round(fps / sample_fps)), 1)
    return list(range(0, frame_count, step))


def extract_frames(path: str | Path, indices: Sequence[int], output_dir: Path) -> List[Path]:
    if cv2 is None:
        raise ImportError("opencv-python is required for frame extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    extracted: List[Path] = []
    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue
            target = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(target), frame)
            extracted.append(target)
    finally:
        cap.release()
    return extracted


def normalise_scores(scores: Sequence[float]) -> List[float]:
    array = np.asarray(list(scores), dtype=np.float32)
    if not len(array):
        return []
    max_val = float(array.max())
    if max_val <= 0:
        return [0.0 for _ in array]
    array /= max_val
    return array.round(4).tolist()


def moving_average(values: Sequence[float], window_size: int) -> List[float]:
    if window_size <= 1:
        return list(values)
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return []
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    padded = np.pad(arr, (window_size - 1, 0), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: arr.size].round(4).tolist()

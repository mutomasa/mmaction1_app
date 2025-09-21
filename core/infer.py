from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from mmengine.utils import mkdir_or_exist

try:  # pragma: no cover - optional dependency at runtime
    from mmaction.apis import inference_recognizer, init_recognizer
except ImportError:  # pragma: no cover - handled lazily
    inference_recognizer = None  # type: ignore
    init_recognizer = None  # type: ignore

from core.cache import get_cache_dir
from core.postprocess import coerce_predictions, to_timeseries, topk_predictions
from models.zoo import ModelEntry, get_model_entry, load_label_map, resolve_model_files

LOGGER = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    model_key: str
    device: str = "cuda"
    label_map: Sequence[str] | None = None
    sample_rate: float = 4.0
    topk: int = 5
    threshold: float = 0.05
    window_size: int = 5


class ActionRecognizer:
    def __init__(self, entry: ModelEntry, device: str = "cuda", *, label_map: Sequence[str] | None = None):
        if init_recognizer is None:
            raise ImportError("mmaction2 is required. Install it via `uv pip install mmaction2`.")
        self.entry = entry
        self.device = device
        config_path, checkpoint_path = resolve_model_files(entry)
        self.model = init_recognizer(str(config_path), str(checkpoint_path), device=device)
        self.label_map = list(label_map) if label_map else None
        LOGGER.info("Loaded model %s on %s", entry.display_name, device)

    def predict_video(
        self,
        video_path: str | Path,
        *,
        sample_rate: float,
        topk: int,
        threshold: float,
        window_size: int,
    ) -> dict:
        if inference_recognizer is None:
            raise ImportError("mmaction2 is required. Install it via `uv pip install mmaction2`.")
        raw = inference_recognizer(self.model, str(video_path))
        labels = self.label_map
        predictions = coerce_predictions(raw, labels=labels)
        top_predictions = topk_predictions(predictions, topk=topk, threshold=threshold)
        timeseries = to_timeseries(predictions, sample_rate=sample_rate, window_size=window_size)
        return {
            "topk": [prediction.__dict__ for prediction in top_predictions],
            "timeseries": timeseries,
        }


def build_recognizer(config: InferenceConfig) -> ActionRecognizer:
    entry = get_model_entry(config.model_key)
    labels = list(config.label_map) if config.label_map else load_label_map(entry)
    cache_dir = get_cache_dir("checkpoints")
    mkdir_or_exist(cache_dir)
    return ActionRecognizer(entry=entry, device=config.device, label_map=labels)


def run_inference(config: InferenceConfig, video_path: str | Path) -> dict:
    recognizer = build_recognizer(config)
    return recognizer.predict_video(
        video_path,
        sample_rate=config.sample_rate,
        topk=config.topk,
        threshold=config.threshold,
        window_size=config.window_size,
    )

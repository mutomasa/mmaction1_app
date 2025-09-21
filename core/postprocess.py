from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from core.preprocess import moving_average, normalise_scores


@dataclass
class Prediction:
    label: str
    score: float


def coerce_predictions(raw: Iterable, labels: Sequence[str] | None = None) -> List[Prediction]:
    predictions: List[Prediction] = []
    for item in raw:
        if isinstance(item, dict):
            label = item.get("label")
            score = float(item.get("score", 0.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            label_idx, score = item[0], float(item[1])
            if labels and isinstance(label_idx, (int, float)):
                index = int(label_idx)
                label = labels[index] if 0 <= index < len(labels) else str(label_idx)
            else:
                label = str(label_idx)
        else:
            continue
        if label is None:
            continue
        predictions.append(Prediction(label=label, score=score))
    return predictions


def topk_predictions(predictions: Sequence[Prediction], topk: int, threshold: float) -> List[Prediction]:
    filtered = [pred for pred in predictions if pred.score >= threshold]
    filtered.sort(key=lambda item: item.score, reverse=True)
    return list(filtered[:topk])


def to_timeseries(predictions: Sequence[Prediction], sample_rate: float, window_size: int) -> List[dict]:
    scores = [pred.score for pred in predictions]
    smoothed = moving_average(scores, window_size=window_size)
    normalised = normalise_scores(smoothed)
    timeseries = []
    for idx, pred in enumerate(predictions):
        timecode = round(idx / sample_rate, 2) if sample_rate > 0 else idx
        timeseries.append({
            "timestamp": timecode,
            "label": pred.label,
            "score": pred.score,
            "smoothed": smoothed[idx] if idx < len(smoothed) else pred.score,
            "normalised": normalised[idx] if idx < len(normalised) else 0.0,
        })
    return timeseries

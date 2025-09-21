from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from core.preprocess import moving_average, normalise_scores


@dataclass
class Prediction:
    label: str
    score: float


def _resolve_label(label_idx: int | float | str, labels: Sequence[str] | None) -> str:
    if labels:
        try:
            index = int(label_idx)  # handles numpy / tensor scalars
        except (TypeError, ValueError):
            pass
        else:
            if 0 <= index < len(labels):
                return labels[index]
    return str(label_idx)


def _scores_to_list(scores) -> List[float]:
    if scores is None:
        return []
    if hasattr(scores, "detach"):
        scores = scores.detach()
    if hasattr(scores, "cpu"):
        scores = scores.cpu()
    if hasattr(scores, "numpy"):
        try:
            scores = scores.numpy()
        except TypeError:
            pass
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (list, tuple)):
        return [float(score) for score in scores]
    return [float(scores)]


def coerce_predictions(raw: Iterable | object, labels: Sequence[str] | None = None) -> List[Prediction]:
    predictions: List[Prediction] = []

    if raw is None:
        return predictions

    # mmaction の ActionDataSample など、スコアを属性として持つケース
    if hasattr(raw, "pred_score") or hasattr(raw, "pred_scores"):
        scores = getattr(raw, "pred_score", None)
        if scores is None:
            scores = getattr(raw, "pred_scores", None)
        for idx, score in enumerate(_scores_to_list(scores)):
            predictions.append(Prediction(label=_resolve_label(idx, labels), score=score))
        return predictions

    if isinstance(raw, dict):
        if "pred_score" in raw or "pred_scores" in raw:
            scores = raw.get("pred_score")
            if scores is None:
                scores = raw.get("pred_scores")
            for idx, score in enumerate(_scores_to_list(scores)):
                predictions.append(Prediction(label=_resolve_label(idx, labels), score=score))
            return predictions
        label = raw.get("label")
        score = raw.get("score")
        if label is not None and score is not None:
            predictions.append(Prediction(label=str(label), score=float(score)))
            return predictions

    if isinstance(raw, (list, tuple)):
        iterable = raw if isinstance(raw, list) else [raw]
        for item in iterable:
            if isinstance(item, dict):
                label = item.get("label")
                score = item.get("score")
                if label is None or score is None:
                    continue
                predictions.append(Prediction(label=str(label), score=float(score)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                label_idx, score = item[0], item[1]
                predictions.append(Prediction(label=_resolve_label(label_idx, labels), score=float(score)))
        return predictions

    for idx, score in enumerate(_scores_to_list(raw)):
        predictions.append(Prediction(label=_resolve_label(idx, labels), score=score))
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

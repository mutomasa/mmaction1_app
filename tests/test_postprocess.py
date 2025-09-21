from __future__ import annotations

import pytest

from core.postprocess import Prediction, coerce_predictions, to_timeseries, topk_predictions


try:  # pragma: no cover - optional dependency for runtime branch
    import torch
    from mmaction.structures import ActionDataSample
except ImportError:  # pragma: no cover
    ActionDataSample = None  # type: ignore
    torch = None  # type: ignore


def test_coerce_predictions_from_tuples():
    preds = coerce_predictions([(0, 0.8), (1, 0.2)], labels=["jump", "run"])
    assert preds[0].label == "jump"
    assert preds[0].score == 0.8


def test_topk_predictions_with_threshold():
    predictions = [Prediction(label="jump", score=0.9), Prediction(label="run", score=0.3)]
    topk = topk_predictions(predictions, topk=1, threshold=0.5)
    assert len(topk) == 1
    assert topk[0].label == "jump"


def test_timeseries_contains_smoothed_scores():
    predictions = [Prediction(label="jump", score=0.9), Prediction(label="run", score=0.3)]
    ts = to_timeseries(predictions, sample_rate=2.0, window_size=1)
    assert ts[0]["timestamp"] == 0.0
    assert "smoothed" in ts[0]


@pytest.mark.skipif(ActionDataSample is None or torch is None, reason="mmaction2 not available")
def test_coerce_predictions_from_action_data_sample():
    sample = ActionDataSample()
    sample.set_pred_score(torch.tensor([0.1, 0.9]))
    preds = coerce_predictions(sample, labels=["jump", "run"])
    assert len(preds) == 2
    assert preds[0].label == "jump"
    assert pytest.approx(preds[1].score, rel=1e-5) == 0.9

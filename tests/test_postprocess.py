from __future__ import annotations

from core.postprocess import Prediction, coerce_predictions, to_timeseries, topk_predictions


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

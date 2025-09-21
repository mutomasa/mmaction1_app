from __future__ import annotations

from core.preprocess import compute_sample_indices, moving_average, normalise_scores


def test_compute_sample_indices_downsamples():
    indices = compute_sample_indices(frame_count=100, fps=30.0, sample_fps=5)
    assert indices[0] == 0
    assert indices[1] == indices[0] + 6
    assert len(indices) > 0


def test_normalise_scores_handles_zero():
    assert normalise_scores([0.0, 0.0]) == [0.0, 0.0]


def test_moving_average_window():
    smoothed = moving_average([1.0, 2.0, 3.0], window_size=2)
    assert len(smoothed) == 3
    assert smoothed[0] <= smoothed[1]

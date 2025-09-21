from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt


def render_score_plot(timeseries: Sequence[dict]):
    if not timeseries:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No predictions", ha="center", va="center")
        ax.set_axis_off()
        return fig
    timestamps = [item.get("timestamp", idx) for idx, item in enumerate(timeseries)]
    scores = [item.get("score", 0.0) for item in timeseries]
    smoothed = [item.get("smoothed", val) for item, val in zip(timeseries, scores)]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(timestamps, scores, label="score", alpha=0.5)
    ax.plot(timestamps, smoothed, label="smoothed", linewidth=2)
    ax.set_xlabel("Seconds")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Confidence Over Time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig

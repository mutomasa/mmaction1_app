from __future__ import annotations

from pathlib import Path

from core.infer import ActionRecognizer, InferenceConfig, run_inference
from models.zoo import ModelEntry


class DummyModel:
    pass


def make_entry() -> ModelEntry:
    return ModelEntry(
        display_name="Test",
        config="config.py",
        checkpoint="checkpoint.pth",
        label_map="labels.txt",
    )


def test_action_recognizer_runs(monkeypatch, tmp_path: Path):
    entry = make_entry()

    def fake_resolve(entry: ModelEntry):
        return tmp_path / entry.config, tmp_path / entry.checkpoint

    def fake_init(config: str, checkpoint: str, device: str):
        assert config == str(tmp_path / entry.config)
        assert checkpoint == str(tmp_path / entry.checkpoint)
        assert device == "cpu"
        return DummyModel()

    def fake_infer(model, video_path):
        assert isinstance(model, DummyModel)
        assert video_path == str(tmp_path / "video.mp4")
        return [(0, 0.9), (1, 0.2)]

    monkeypatch.setattr("core.infer.resolve_model_files", fake_resolve)
    monkeypatch.setattr("core.infer.init_recognizer", fake_init)
    monkeypatch.setattr("core.infer.inference_recognizer", fake_infer)

    recognizer = ActionRecognizer(entry=entry, device="cpu", label_map=["jump", "run"])
    result = recognizer.predict_video(
        tmp_path / "video.mp4",
        sample_rate=4,
        topk=1,
        threshold=0.5,
        window_size=1,
    )
    assert result["topk"] == [{"label": "jump", "score": 0.9}]
    assert result["timeseries"]


def test_run_inference(monkeypatch, tmp_path: Path):
    config = InferenceConfig(model_key="tsn_r50_kinetics400", device="cpu", label_map=["jump"], sample_rate=2.0)

    class DummyRecognizer:
        def predict_video(self, video_path, sample_rate, topk, threshold, window_size):
            assert sample_rate == 2.0
            assert topk == config.topk
            assert threshold == config.threshold
            assert window_size == config.window_size
            return {"topk": [], "timeseries": []}

    dummy = DummyRecognizer()
    monkeypatch.setattr("core.infer.build_recognizer", lambda cfg: dummy)

    result = run_inference(config, tmp_path / "video.mp4")
    assert result == {"topk": [], "timeseries": []}

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import yaml

from core.infer import ActionRecognizer, InferenceConfig
from core.postprocess import Prediction
from core.viz import render_score_plot
from models.zoo import available_models

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "app.yaml"


def load_defaults() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


@st.cache_resource(show_spinner=False)
def predictor_factory(model_key: str, device: str) -> ActionRecognizer:
    from models.zoo import get_model_entry, load_label_map

    entry = get_model_entry(model_key)
    labels = load_label_map(entry)
    return ActionRecognizer(entry=entry, device=device, label_map=labels)


st.set_page_config(page_title="MMAction2 Action Recognition", layout="wide")
st.title("MMAction2 行動認識デモ")
defaults = load_defaults()

with st.sidebar:
    st.header("設定")
    model_choices = list(available_models())
    default_model = defaults.get("default_model", model_choices[0] if model_choices else "")
    model_index = model_choices.index(default_model) if default_model in model_choices else 0
    model_key = st.selectbox("モデル", model_choices, index=model_index)
    device = st.selectbox("デバイス", ["cuda", "cpu"], index=0 if defaults.get("default_device", "cuda") == "cuda" else 1)
    sample_fps = st.slider("サンプリングFPS", 1, 16, int(defaults.get("sample_fps", 4)))
    topk = st.slider("Top-K", 1, 10, int(defaults.get("topk", 5)))
    threshold = st.slider("スコアしきい値", 0.0, 1.0, float(defaults.get("score_threshold", 0.05)), 0.01)
    window_size = st.slider("スムージングウィンドウ", 1, 15, int(defaults.get("window_size", 5)))

st.write("アップロードした動画またはURLを使って行動認識を実行します。")

source = st.radio("入力ソース", ["動画ファイル", "URL"], horizontal=True)
video_path: Path | None = None
if source == "動画ファイル":
    uploaded = st.file_uploader("動画を選択", type=["mp4", "mov", "avi"])
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".mp4") as tmp:
            tmp.write(uploaded.read())
            video_path = Path(tmp.name)
else:
    url = st.text_input("動画URL")
    if url:
        st.info("URLからの直接ダウンロードは実環境で実装してください。")

run = st.button("推論を実行")
if run:
    if not video_path:
        st.error("動画を指定してください。")
    else:
        with st.spinner("モデルを読み込み中..."):
            recognizer = predictor_factory(model_key, device)
        with st.spinner("推論を実行中..."):
            config = InferenceConfig(
                model_key=model_key,
                device=device,
                label_map=recognizer.label_map,
                sample_rate=float(sample_fps),
                topk=topk,
                threshold=threshold,
                window_size=window_size,
            )
            result = recognizer.predict_video(
                video_path,
                sample_rate=config.sample_rate,
                topk=config.topk,
                threshold=config.threshold,
                window_size=config.window_size,
            )

        st.subheader("Top-K ラベル")
        topk_rows = [Prediction(**item) for item in result["topk"]]
        st.table({"label": [row.label for row in topk_rows], "score": [row.score for row in topk_rows]})

        st.subheader("スコア推移")
        fig = render_score_plot(result["timeseries"])
        st.pyplot(fig)

        st.subheader("動画プレビュー")
        st.video(str(video_path))
        st.success("推論が完了しました。")
else:
    st.caption("サンプル動画は `assets/samples/` に配置してください。")

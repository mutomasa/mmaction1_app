1. 目的・概要

目的：アップロードした動画またはWebカメラ映像に対して、MMAction2 の学習済みモデルで行動認識を実行し、トップKラベルとスコアの時系列を可視化する。

対象ユーザー：研究・PoC・店舗/工場の行動分析のプロトタイピング。

アプリ構成：Streamlit（UI）＋ 推論サービス層（MMAction2）＋ 補助ユーティリティ（前処理/後処理/可視化/キャッシュ）。

2. 対応環境

OS: Ubuntu 22.04/24.04（推奨）

GPU: NVIDIA（任意。GPUありならCUDAで高速化）

Python: 3.10–3.11（uv が準拠しているバージョン）

ブラウザ: 最新版 Chrome / Edge / Firefox

3. 機能要件

入力

動画ファイル（.mp4/.mov/.avi）

Webカメラ（任意・環境対応時）

推論対象フレームレート（サブサンプリング）設定

モデル

モデル選択（例：TSN/TSM/SlowFast/X3D などZooから）

事前学習済み重みの自動ダウンロード（初回のみ）

推論

単一動画のオンライン/バッチ推論

トップK分類結果（時系列 / 全体集計）

スコアしきい値・トップKの変更

出力

トップKの表・バー表示

時系列スコアの折れ線グラフ

アノテーション付きプレビュー（任意：テキストオーバレイ）

JSON（時系列結果）エクスポート

ユーティリティ

キャッシュ（モデル・前処理済みフレーム・結果）

ログ出力（INFO/DEBUG）

設定の保存/復元（YAML/JSON）

4. 非機能要件

性能：GPU時 30fps 入力でサブサンプリング 2〜4fps 推論を想定（モデル依存）

再現性：固定シード（前処理サンプリング）

保守性：モジュール分割、型ヒント、Docstring

セキュリティ：アップロードファイルはアプリ内一時領域に保存。外部送信しない。

5. ディレクトリ構成（提案）
mmaction2-streamlit/
  app/
    __init__.py
    ui.py                 # Streamlit UI
    pages/                # 将来拡張（バッチ処理など）
  core/
    __init__.py
    infer.py              # init_recognizer / inference_recognizer ラッパ
    preprocess.py         # 動画デコード・フレーム抽出
    postprocess.py        # 時系列集計・スムージング
    viz.py                # 可視化・フレームへのラベル描画
    cache.py              # ディスク/メモリキャッシュ
    config_schemas.py     # 設定バリデーション(pydantic推奨)
  models/
    zoo.py                # モデル定義（config/ckptのマッピング）
  configs/
    app.yaml              # 既定設定
  tests/
    test_infer.py
    test_preprocess.py
  assets/
    samples/              # サンプル動画(任意)
  pyproject.toml          # uv用（プロジェクト設定）
  README.md
  .env.example

6. 主要依存パッケージ

torch, torchvision（CUDA または CPU）

mmengine, mmcv（OpenMMLab 基盤）

mmaction2（行動認識）

streamlit, opencv-python, numpy, pydantic, pyyaml, matplotlib（可視化は任意）

追加：openmim（MMCV/各種依存の解決に便利）

7. セットアップ（uv 利用）
7.1 プロジェクト初期化
# Pythonツールチェイン: uv を使用
# 1) 新規プロジェクト作成
uv init mmaction2-streamlit
cd mmaction2-streamlit

# 2) 仮想環境作成・有効化
uv venv
source .venv/bin/activate

7.2 PyTorch のインストール
(A) CUDA 環境（例: CUDA 12.1）
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

(B) CPU 環境
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

7.3 OpenMMLab 依存
uv pip install -U openmim
python -m mim install "mmcv>=2.0.0"

7.4 MMAction2・その他
# mmaction2 本体とユーティリティ
uv pip install mmaction2 streamlit opencv-python numpy pydantic pyyaml matplotlib


注：環境やCUDA版の差異で依存解決が難しい場合は、mim install mmaction2 を試し、mmcv のバージョンを mmaction2 の要求に合わせて固定してください。

8. 設定（例：configs/app.yaml）
model:
  name: "tsn_r50_kinetics400"
  topk: 5
  score_threshold: 0.05
  # zoo.py 内のエントリ名に一致
  # config/ckpt は zoo.py 側で定義

inference:
  sample_fps: 2          # サブサンプリングFPS
  window_sec: 1.0        # 集計ウィンドウ
  smooth_sigma: 0.5      # ガウシアン平滑(任意)

ui:
  theme: "auto"
  max_video_seconds: 120 # 大きな動画はサマリに制限
  show_timeseries: true
  show_overlay: false

cache:
  dir: ".cache"
  enable: true

9. モデル Zoo 定義（例：models/zoo.py）
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelEntry:
    display_name: str
    config: str
    checkpoint: str
    label_map: str  # Kinetics-400 など

ZOO = {
    "tsn_r50_kinetics400": ModelEntry(
        display_name="TSN-R50 (Kinetics-400)",
        config="https://raw.githubusercontent.com/open-mmlab/mmaction2/main/configs/recognition/tsn/tsn_r50_8xb32-1x1x3-100e_kinetics400-rgb.py",
        checkpoint="https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_r50_kinetics400.pth",
        label_map="https://raw.githubusercontent.com/open-mmlab/mmaction2/main/tools/data/kinetics/label_map_k400.txt",
    ),
    # 例: SlowFast, X3D を必要に応じて追加
}


実運用時は、URL の可用性に備えて ローカルへミラーすることを推奨。

10. 推論ラッパ（core/infer.py）
from pathlib import Path
from typing import List, Dict, Any
from mmengine.utils import mkdir_or_exist
from mmaction.apis import init_recognizer, inference_recognizer

class ActionRecognizer:
    def __init__(self, config: str, checkpoint: str, device: str = "cuda"):
        self.model = init_recognizer(config, checkpoint, device=device)
        self.device = device

    def predict_video(self, video_path: str, label_map: List[str]) -> List[Dict[str, Any]]:
        # inference_recognizer は (label, score) のリストを返す（実装により差異あり）
        results = inference_recognizer(self.model, video_path)
        # 形式を統一
        return [{"label": label_map[idx], "score": float(score)} for idx, score in results]

11. 前処理・後処理（骨子）

preprocess.py

OpenCV または decord で動画読込

指定 sample_fps にサブサンプリング

MMAction2 の想定入力に整形（必要に応じてテンポラルクリップ生成）

postprocess.py

トップK抽出

スコアしきい値適用

時系列スムージング（移動平均/ガウシアン）

viz.py

スコアの折れ線グラフ（matplotlib）

フレームへのテキスト描画（OpenCV putText）

12. Streamlit UI（app/ui.py：最小動作例）
import streamlit as st
import tempfile, requests
from core.infer import ActionRecognizer
from models.zoo import ZOO

st.set_page_config(page_title="MMAction2 Action Recognition", layout="wide")

st.title("MMAction2 行動認識デモ")

# モデル選択
model_key = st.selectbox("モデルを選択", list(ZOO.keys()),
                         format_func=lambda k: ZOO[k].display_name)
entry = ZOO[model_key]

# ラベルマップ取得
@st.cache_resource
def get_label_map(url: str):
    text = requests.get(url, timeout=30).text
    return [line.strip() for line in text.splitlines() if line.strip()]

# モデル初期化（キャッシュ）
@st.cache_resource
def load_model(config: str, ckpt: str, device: str):
    return ActionRecognizer(config=config, checkpoint=ckpt, device=device)

device = "cuda" if st.checkbox("GPUを使う", value=True) else "cpu"
labels = get_label_map(entry.label_map)
recognizer = load_model(entry.config, entry.checkpoint, device)

# 入力
src = st.radio("入力ソース", ["動画ファイル", "URL"], horizontal=True)

video_path = None
if src == "動画ファイル":
    up = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi"])
    if up:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(up.read()); tmp.flush()
        video_path = tmp.name
else:
    url = st.text_input("動画URL（直接アクセス可能）")
    if url:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(requests.get(url, timeout=60).content); tmp.flush()
        video_path = tmp.name

topk = st.slider("Top-K", 1, 10, 5)
threshold = st.slider("スコアしきい値", 0.0, 1.0, 0.05, 0.01)

run = st.button("推論を実行")
if run and video_path:
    with st.spinner("推論中..."):
        result = recognizer.predict_video(video_path, labels)  # 簡易例
        result = sorted(result, key=lambda x: x["score"], reverse=True)[:topk]
        result = [r for r in result if r["score"] >= threshold]
    st.subheader("結果")
    st.table(result)
    st.video(video_path)


注：inference_recognizer の返却形式やクリップ処理は モデル・バージョンで差異があります。必要に応じて MMAction2 のサンプルに合わせて前処理（フレームスタッキング、clip_len/interval 設定など）を実装してください。

13. 実行方法
# 依存インストール（前節参照後）
uv run streamlit run app/ui.py
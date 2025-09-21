# Repository Guidelines
## Project Structure & Module Organization
This Streamlit + MMAction2 demo should follow the proposed `mmaction2-streamlit/` layout:
- `app/` houses Streamlit entry points (`ui.py`) and future `pages/`.
- `core/` packages inference (`infer.py`), preprocessing, postprocessing, visualization, and caching utilities.
- `models/` contains the model zoo definitions and download helpers.
- `configs/` stores default app settings (e.g., `app.yaml`) and environment templates.
- `tests/` mirrors `core/` with focused units; keep fixtures in `tests/data/`.
- `assets/` holds sample videos and static media shipped with the UI.
Keep shared constants/loaders in dedicated modules to avoid circular imports.

## Build, Test, and Development Commands
- `uv venv && source .venv/bin/activate` creates and activates the virtual environment.
- `uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision` installs CUDA builds; swap to the CPU index when GPU is unavailable.
- `uv pip install .` (once `pyproject.toml` is committed) installs app dependencies, including MMAction2 via OpenMMLab.
- `uv run streamlit run app/ui.py` launches the local UI.
- `uv run pytest` runs the full suite; add `-k <keyword>` for narrower loops during development.

## Coding Style & Naming Conventions
Target Python 3.10–3.11, enforce 4-space indentation, and stick to PEP 8 (`snake_case` functions, `PascalCase` classes). Annotate public functions with type hints and concise docstrings, matching the spec’s emphasis on maintainability. Keep Streamlit callbacks lightweight and push heavy logic into `core/`. Adopt `ruff` and `black` via `pyproject.toml` to keep formatting consistent before submitting changes.

## Testing Guidelines
Use `pytest` with descriptive names (`test_preprocess_extracts_frames`). Validate deterministic sampling by fixing random seeds and asserting on output shapes and score thresholds. Provide lightweight sample clips under `tests/data/` and skip GPU-only assertions when `CUDA_VISIBLE_DEVICES` is empty. Ensure new model entries carry regression tests that cover label-map handling and caching paths.

## Commit & Pull Request Guidelines
Author commits around single logical changes with imperative subjects (e.g., `Add SlowFast preset`). Summaries should cite the relevant module and risk. Pull requests must outline motivation, key changes, local test evidence (`uv run pytest`), and any configuration files touched (`configs/app.yaml`, `.env.example`). Include screenshots or console snippets when the UI or logging behavior shifts, and link tracking issues or specs for traceability.

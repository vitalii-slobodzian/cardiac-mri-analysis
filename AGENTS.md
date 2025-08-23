# Repository Guidelines

## Project Structure & Module Organization
- `segmentation/`: Fastai-based training code. Key files: `learner.py`, `helpers/` (`DiceMetric`, `NiiReader`, `ConfigLoader`, image utils). Notebooks: `dataset_builder.ipynb`, `training.ipynb`.
- `classification/`: Utilities for feature engineering, e.g., `helpers/polar_transformation.py`.
- `visualization/`: Static assets for rendering outputs (e.g., `index.html`).
- `img/`: Figures used in the README and docs.
- `.github/workflows/`: Release workflow for packaging to PyPI.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && .venv\\Scripts\\activate` (Windows) or `source .venv/bin/activate` (Unix).
- Install deps (current subset): `pip install -r segmentation/requirements.txt`.
- Format code: `pip install black && black .`.
- Run notebooks: `jupyter lab` (recommended for `segmentation/*.ipynb`).
- Optional test setup: `pip install pytest` then `pytest -q` (see Testing Guidelines).

## Coding Style & Naming Conventions
- Python 3.9+. Use 4-space indentation and keep lines concise; format with Black.
- Naming: `snake_case` for functions/vars/files, `PascalCase` for classes (e.g., `DiceMetric`, `NiiReader`).
- Modules: keep helpers in `segmentation/helpers/` and avoid circular imports; prefer explicit exports via `__init__.py`.
- Config: load YAML via `helpers/ConfigLoader.get_config(path)`; keep file paths relative and configurable.
- Types: add type hints where clear; keep public APIs minimal and documented in docstrings.

## Testing Guidelines
- Framework: `pytest`.
- Layout: place tests under `tests/`, mirror module paths. Name files `test_*.py`.
- Targets: unit-test utilities (`DiceMetric.multi_dice`, `NiiReader.parse_config`, image helpers). Use small arrays and synthetic masks for speed.
- Run: `pytest -q`. Aim for fast, deterministic tests; prefer CPU-only paths.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (e.g., "Add DiceMetric unit tests"); group related changes.
- PRs: clear description, linked issues (e.g., `Fixes #12`), before/after visuals for `visualization/` changes, and notes on data/paths.
- Checks: run Black before pushing; ensure notebooks are cleared of large outputs.
- Releases: GitHub Action builds distributions on release creation; keep versioning consistent before tagging.

## Security & Configuration Tips
- Do not commit datasets or PHI; keep raw NIFTI outside the repo. Honor `.gitignore`.
- Parameterize data paths and seeds in YAML; avoid hard-coded absolute paths.
- Validate external inputs (YAML keys, file existence) before training or inference.


# Technology Stack

## 1) Runtime Summary

| Area | Value | Evidence |
|---|---|---|
| Primary language | Python | `pyproject.toml` |
| Runtime + version | Python >=3.11, <3.14 | `pyproject.toml` |
| Package manager | uv | `uv.lock`, `README.md`, `.github/workflows/ci.yml` |
| Module/build system | Hatchling; `src/` package layout | `pyproject.toml` |

## 2) Production Frameworks and Dependencies

| Dependency | Version | Role in system | Evidence |
|---|---:|---|---|
| pandas | 2.2.3 | Tabular market-data ingestion and artifacts | `pyproject.toml` |
| numpy | 2.3.5 | Numerical features, targets, and statistics | `pyproject.toml` |
| scikit-learn | 1.8.0 | Candidate classifiers, preprocessing, metrics | `pyproject.toml`, `src/hge_gold/modeling.py` |
| PyYAML | 6.0.3 | YAML experiment configuration | `pyproject.toml`, `src/hge_gold/config.py` |
| joblib | 1.5.3 | Persisted model bundles | `pyproject.toml`, `src/hge_gold/pipeline.py` |
| Typer | 0.16.1 | Command-line interface | `pyproject.toml`, `src/hge_gold/cli.py` |
| MetaTrader5 | 5.0.6090 (Windows extra) | MT5-related optional functionality | `pyproject.toml` |

## 3) Development Toolchain

| Tool | Purpose | Evidence |
|---|---|---|
| pytest / pytest-cov | Tests and coverage | `pyproject.toml`, `.github/workflows/ci.yml` |
| ruff | Linting and import-order checks | `pyproject.toml` |
| mypy | Strict static type checking | `pyproject.toml` |
| bandit / pip-audit | Available security tooling | `pyproject.toml` |

## 4) Key Commands

```bash
uv sync --extra dev --extra mt5
uv run hge-gold --config configs/thesis.yaml
uv run pytest --cov=hge_gold --cov-fail-under=80
uv run ruff check .
uv run mypy src/hge_gold
```

## 5) Environment and Config

- Config sources: `configs/thesis.yaml`, `configs/thesis_quick.yaml`.
- Required environment variables: none discovered; input and output paths are YAML configuration fields.
- Runtime constraint: the MT5 optional dependency is Windows-only; CI uses Python 3.12.

## 6) Evidence

- `pyproject.toml`
- `uv.lock`
- `.github/workflows/ci.yml`

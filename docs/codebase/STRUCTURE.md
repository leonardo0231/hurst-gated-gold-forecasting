# Codebase Structure

## 1) Top-Level Map

| Path | Purpose | Evidence |
|---|---|---|
| `src/hge_gold/` | Pipeline source package | `pyproject.toml`, source files |
| `tests/` | Unit, contract, provenance, and end-to-end tests | `pyproject.toml`, `tests/` |
| `configs/` | Market and synthetic experiment configurations | `README.md`, `configs/thesis.yaml` |
| `artifacts/` | Generated metrics, manifests, audit reports, immutable run receipts | `README.md`, `src/hge_gold/pipeline.py` |
| `data/` | Raw/processed input and generated prediction files | `configs/thesis.yaml`, `README.md` |
| `models/` | Persisted model bundles | `src/hge_gold/pipeline.py` |
| `mt5/` | Frozen-signal replay EA and its contract | `mt5/README.md` |
| `scripts/` | Operational helper scripts, including MT5 backtest runner | `README.md` |
| `.github/workflows/` | Continuous-integration workflows | `.github/workflows/ci.yml` |

## 2) Entry Points

- Main runtime entry: `hge_gold.cli:app`.
- Secondary entry points: `src/hge_gold/__main__.py`; `scripts/run_mt5_backtest.py` documented in `README.md`.
- Entry selection: the `hge-gold` script declared in `pyproject.toml` invokes Typer's `run` command, defaulting to `configs/thesis.yaml`.

## 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|---|---|---|
| `data.py`, `data_audit.py` | OHLCV normalization and quality audit | Model selection |
| `features.py`, `targets.py` | Causal features and target timestamps | Holdout evaluation policy |
| `splits.py` | Chronological and purged split construction | Feature computation |
| `modeling.py` | Candidate fitting, out-of-fold predictions, stacker selection | File persistence |
| `evaluation.py`, `statistics.py` | Classification, bootstrap, and economic summaries | Input ingestion |
| `pipeline.py` | End-to-end orchestration and artifact persistence | Estimator implementation |
| `mt5.py` | Frozen replay signal construction | Native MQL5 inference |

## 4) Naming and Organization Rules

- Python files and functions use `snake_case`; classes use `PascalCase`.
- Modules are organized by pipeline responsibility rather than by a web/API feature.
- Imports use relative imports within `hge_gold`; no path aliases were found.

## 5) Evidence

- `pyproject.toml`
- `src/hge_gold/cli.py`
- `src/hge_gold/pipeline.py`

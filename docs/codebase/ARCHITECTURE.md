# Architecture

## 1) Architectural Style

- Primary style: batch, layered research pipeline.
- Why this classification: the CLI delegates to one orchestration function which sequences ingestion, feature/target creation, splitting, model selection, evaluation, and artifact export.
- Primary constraints: chronological causality, leakage control, immutable evidence, and fixed-signal MT5 replay.

## 2) System Flow

```text
CLI -> configuration + OHLCV loading -> causal features + targets -> purged validation + model selection -> locked evaluation/backtest -> hashed artifacts and MT5 replay signals
```

1. `cli.py` calls `run_thesis_pipeline` with a YAML configuration.
2. `pipeline.py` loads and validates OHLCV data, captures provenance, and produces a quality audit.
3. It builds causal features and a separate dataset for each configured horizon.
4. It creates development/locked partitions, purged expanding folds, and development-only CPCV diagnostics.
5. `modeling.py` selects a candidate and optional stacked meta-classifier from development evidence, then refits on development data.
6. `evaluation.py` produces locked-test metrics, bootstrap intervals, benchmarks, and frozen MT5 signal CSVs.

## 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|---|---|---|---|
| CLI | Argument parsing and user-visible paths | Research calculations | `src/hge_gold/cli.py` |
| Pipeline | Workflow orchestration and artifact assembly | Estimator-specific tuning logic | `src/hge_gold/pipeline.py` |
| Data/features/targets | Valid data, causal transformations, labels | Model selection | `src/hge_gold/data.py`, `src/hge_gold/features.py`, `src/hge_gold/targets.py` |
| Splits/modeling | Purged validation and model choice | Output file layout | `src/hge_gold/splits.py`, `src/hge_gold/modeling.py` |
| Evaluation/MT5 | Metrics, economic schedules, replay contract | Training inside MT5 | `src/hge_gold/evaluation.py`, `src/hge_gold/mt5.py` |

## 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---|---|---|
| Typed configuration dataclasses | `src/hge_gold/config.py` | Centralizes and validates experiment parameters |
| sklearn pipelines | `src/hge_gold/modeling.py` | Keeps scaling and classification coupled during fitting |
| Immutable run receipt | `src/hge_gold/pipeline.py` | Refuses overwriting an existing identified run |
| Purged chronological split | `src/hge_gold/splits.py` | Avoids target-interval overlap leakage |
| Atomic output write helpers | `src/hge_gold/pipeline.py` | Avoids partially written artifacts |

## 5) Known Architectural Risks

- The pipeline produces mutable canonical outputs plus immutable run copies; consumers must use the run receipt to identify exact evidence.
- MT5 replays externally generated signals and therefore does not verify Python-to-native model inference parity.

## 6) Evidence

- `src/hge_gold/cli.py`
- `src/hge_gold/pipeline.py`
- `src/hge_gold/splits.py`
- `mt5/README.md`

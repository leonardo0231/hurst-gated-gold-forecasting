# HGE-Hybrid Gold Forecasting Framework

Production-oriented, audit-first implementation of the locked Phase 0–11 specifications. The executable default is offline and uses deterministic, non-confidential OHLCV data. It never connects to a broker, sends an order, submits a manuscript, or stores portal credentials.

## What is implemented

- Phase 1: source validation, causal timestamps, empirical calendar, label intervals, provenance and MVP/paper-grade decision separation.
- Phase 2: 1/5/10/20-day return, statistical direction, economic action, and realized-variance targets with registered thresholds.
- Phase 3: causal gold technical, Hurst, and volatility features; GARCH is explicitly deferred; feature/target alignment and leakage audits are produced.
- Phase 4: global chronological locked test, purged walk-forward validation, fold-local preprocessing, baseline/linear/tree candidates, rule-based Hurst-gated candidate, deterministic model selection, final refit, and locked predictions without `y_true`.
- Phase 5: locked-test join, task metrics, preregistered static baselines, moving-block bootstrap, FDR correction, execution-aligned backtest, transaction costs, claim and limitation registries.
- Phases 6–7: evidence-controlled manuscript and generic submission package generation.
- Phases 8–11: status-aware governance simulation. External submission remains blocked because no journal, manual authorization, or portal receipt is supplied.
- Starlette ASGI health/status and data-contract validation endpoints.

The sample execution validates software behavior. It is not paper-grade market evidence and must not be used to make investment decisions.

## Requirements

- Python 3.11–3.13
- `uv` 0.8+
- About 1 GB free memory for the default local run

## Install

```bash
uv sync --extra dev
```

The committed `uv.lock` is authoritative for reproducibility.

## Run everything

```bash
uv run hge-gold run --config configs/pipeline.yaml
```

Use a user-provided CSV only when it has sorted, unique `date,open,high,low,close,volume` columns:

```bash
uv run hge-gold run --config configs/pipeline.yaml --source-csv /absolute/path/gold.csv
```

No external source is downloaded by the default command. Real Stooq/Yahoo/CME data requires a separate audited ingestion extension because the locked documents forbid silently treating an undocumented feed as paper-grade.

## Run the API

```bash
uv run uvicorn hge_gold.api:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status
```

Example validation request:

```bash
curl -X POST http://127.0.0.1:8000/validate-data \
  -H 'Content-Type: application/json' \
  -d '{"rows":[{"date":"2026-01-02","open":2000,"high":2020,"low":1990,"close":2010,"volume":1000}]}'
```

## Quality checks

```bash
uv run ruff check .
uv run mypy src/hge_gold
uv run pytest --junitxml=artifacts/test-results/pytest-junit.xml \
  --cov=hge_gold --cov-report=term-missing --cov-report=html:artifacts/test-results/htmlcov
uv run bandit -q -r src/hge_gold -f json -o artifacts/test-results/bandit.json
uv run pip-audit --format json --output artifacts/test-results/pip-audit.json
```

## Docker

```bash
docker compose build
docker compose up -d
curl http://localhost:8000/health
```

The container starts the read-only API. Run the pipeline before containerization if you want `/status` to return a completed execution report, or mount existing data/artifact/model volumes.

## Important output paths

- `data/processed/targets/gold_multitask_targets.parquet`
- `data/processed/features/gold_feature_matrix.parquet`
- `data/processed/modeling_base/phase3_modeling_base_gold_only.parquet`
- `data/predictions/phase4/phase4_locked_test_predictions.parquet`
- `artifacts/metadata/phase4_selected_model_map.json`
- `artifacts/metadata/phase5_locked_test_metrics_report.csv`
- `data/backtests/phase5/phase5_trade_ledger.parquet`
- `reports/phase6/manuscript/full_manuscript_draft.md`
- `reports/phase7/submission/phase7_generic_submission_package.zip`
- `reports/execution_report.json`

## Safety and scope

- Trading mode is restricted to `offline`, `simulation`, `sandbox`, or `paper`.
- Cross-market, COT, macro, safe-haven, live trading, and paper-grade claims are disabled until their required audits are supplied.
- The default cost convention is authoritative round-trip cost: 3 bps transaction cost + 2 bps slippage.
- Locked-test targets are absent from Phase 4 prediction artifacts and joined only in Phase 5.
- Phases 8–11 create empty/status-aware registries and never perform external actions.

See [architecture.md](docs/architecture.md), [data-model.md](docs/data-model.md), [api.md](docs/api.md), and [testing.md](docs/testing.md).

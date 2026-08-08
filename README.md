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

## Thesis V2

V2 is an additive, non-breaking research pipeline for the bachelor thesis. It preserves the legacy pipeline and writes its outputs to separate V2 artifact, prediction, and model paths.

The primary learning task is multi-horizon **binary gold-price direction forecasting**. For each valid observation and forecast horizon, the target is defined from the sign of the future log return:

* `1` — the future price is higher than the current price;
* `0` — the future price is unchanged or lower than the current price.

All observations with a valid future return and sufficient causal feature coverage are eligible for modeling. Future price movement magnitude is **not** used to decide whether an observation is included in the training, validation, or locked-test datasets.

An adaptive volatility-based threshold is still calculated, but it is retained only for secondary analysis. It identifies whether a realized future movement can be considered `actionable` and is also used to preserve the secondary three-class `down / flat / up` analytical label. The `is_actionable` field does not control modeling eligibility.

The current target policy identifier is:

`all_samples_binary_direction_v2_1`

Main V2 changes:

* all-sample binary direction targets for 1, 5, 10, and 20-day horizons;
* secondary adaptive actionability and three-class direction labels for analysis;
* expanded causal feature set including RSI, ATR, MACD, volume, trend, entropy, volatility, and Hurst regimes;
* chronological locked test separated from model development;
* purged walk-forward validation to prevent label overlap across temporal boundaries;
* Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting candidate models;
* learned regime gate trained from out-of-fold probabilities with best-base fallback;
* model and probability-threshold selection performed without using the locked test;
* registered acceptance gate using `Balanced Accuracy >= 0.60`, `Macro-F1 >= 0.55`, and minimum recall constraints for both classes;
* unit, split, target-alignment, modeling, causality, and end-to-end tests;
* separate manifests, predictions, model bundles, and compatibility outputs for reproducibility.

The synthetic research fixture is intended only to validate software and modeling behavior. Results produced from the fixture are **not market evidence** and must not be presented as evidence of predictive performance on real gold prices.

### Run the thesis pipeline

```bash
uv run hge-gold-v2 --config configs/thesis_v2.yaml
```

### Quick local validation

```bash
uv run hge-gold-v2 --config configs/thesis_v2_quick.yaml
```

### Run with audited real OHLCV data

```bash
uv run hge-gold-v2 --config configs/thesis_v2.yaml --source-csv /absolute/path/gold.csv
```

The real dataset must satisfy the V2 data contract and contain chronologically ordered, unique OHLCV observations. Dataset source, time range, symbol definition, and relevant preprocessing information must be recorded before results are treated as thesis evidence. Record the available provenance fields in the V2 `data` configuration, for example:

```yaml
data:
  source: market_evidence
  source_type: MT5 broker export
  symbol: XAUUSD
  timeframe: D1
  broker: Example Broker
  server: ExampleBroker-Live
  timezone: UTC
  export_date: 2026-08-08
```

The execution manifest records the resolved input path, a streaming SHA-256 fingerprint, validated row count, first and last timestamps, and these source metadata fields. Unavailable optional metadata is recorded as `null`.

### V2 outputs

* `artifacts/v2/locked_test_metrics.csv`
* `artifacts/v2/candidate_selection_metrics.csv`
* `artifacts/v2/walk_forward_fold_metrics.csv`
* `artifacts/v2/backtest_summary.csv`
* `artifacts/v2/selected_model_map.json`
* `artifacts/v2/feature_registry.json`
* `artifacts/v2/execution_manifest.json`
* `artifacts/data_quality/summary.json`
* `artifacts/data_quality/yearly_statistics.csv`
* `artifacts/data_quality/horizon_class_balance.csv`
* `artifacts/data_quality/suspicious_rows.csv`
* `data/predictions/v2/locked_test_predictions.csv`
* `models/v2/horizon_<H>_model_bundle.joblib`

See [thesis_v2_plan.md](docs/thesis_v2_plan.md) for the current thesis design and [quick_validation_report.md](reports/v2/quick_validation_report.md) for software-validation results.

The market-data quality audit is descriptive and does not filter observations or affect model selection. It records calendar anomalies, return and flash-move outliers, volume anomalies, yearly price/return distributions, class balance for each forecasting horizon, and trailing-price bull/bear regimes. Potential missing weekdays include market holidays and should be reviewed rather than treated as errors automatically.

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

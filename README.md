# HGE Gold Forecasting

Leakage-safe, audit-first research pipeline for multi-horizon binary gold-price direction forecasting. The project runs offline by default, never connects to a broker, and produces research artifacts only—not investment advice.

## Method

For each valid observation and forecast horizon, the primary target is the sign of the future log return:

- `1`: future price is higher than the current price.
- `0`: future price is unchanged or lower than the current price.

The pipeline uses causal technical, volatility, entropy, volume, and Hurst-regime features; a chronological locked test; purged walk-forward validation; Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting candidates; and a learned regime gate with a best-base fallback. Candidate selection, threshold selection, and gate selection do not use the locked test.

The default research fixture is synthetic and validates software behavior only. It is not market evidence.

## Run

Install dependencies:

```bash
uv sync --extra dev
```

Run the default thesis configuration:

```bash
uv run hge-gold --config configs/thesis.yaml
```

Run quick local validation:

```bash
uv run hge-gold --config configs/thesis_quick.yaml
```

Run with an audited OHLCV CSV:

```bash
uv run hge-gold --config configs/thesis.yaml --source-csv /absolute/path/gold.csv
```

The CSV must contain sorted, unique `date,open,high,low,close,volume` columns. Record available provenance in `data` configuration:

```yaml
data:
  source: market_evidence
  source_type: MT5 broker export
  symbol: XAUUSD
  timeframe: D1
  broker: null
  server: null
  timezone: null
  export_date: null
```

## Outputs

- `artifacts/v2/execution_manifest.json`
- `artifacts/v2/locked_test_metrics.csv`
- `artifacts/v2/candidate_selection_metrics.csv`
- `artifacts/v2/walk_forward_fold_metrics.csv`
- `artifacts/v2/backtest_summary.csv`
- `artifacts/v2/selected_model_map.json`
- `artifacts/v2/feature_registry.json`
- `artifacts/data_quality/summary.json`
- `artifacts/data_quality/yearly_statistics.csv`
- `artifacts/data_quality/horizon_class_balance.csv`
- `artifacts/data_quality/suspicious_rows.csv`
- `data/predictions/v2/locked_test_predictions.csv`
- `models/v2/horizon_<H>_model_bundle.joblib`

The execution manifest records source provenance, a streaming SHA-256 fingerprint, row span, and output hashes. The descriptive market-data audit records calendar anomalies, return and flash-move outliers, volume anomalies, yearly price/return distributions, class balance, and bull/bear regimes without filtering modeling observations.

## Quality checks

```bash
uv run ruff check src/hge_gold tests
uv run mypy src/hge_gold
uv run pytest --cov=hge_gold --cov-fail-under=80
```

See [thesis_plan.md](docs/thesis_plan.md) and [quick_validation_report.md](reports/quick_validation_report.md) for the thesis design and validation report.

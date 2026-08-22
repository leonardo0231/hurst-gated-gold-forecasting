# HGE Gold Forecasting

Leakage-controlled research software for multi-horizon XAUUSD direction forecasting. The framework separates statistical close-to-close labels from executable next-open labels, uses purged chronological validation, freezes every selection decision before evaluating the historical holdout, and exports fixed signals for MetaTrader 5 Strategy Tester replay. It is research software, not investment advice.

## Scientific design

- Primary statistical label: `sign(close[t + H] - close[t])`.
- Executable return: entry at `open[t + 1]`, exit at `open[t + 1 + H]`.
- Cost-aware `down / no-trade / up` actionability is diagnostic and never filters the binary modeling sample.
- The locked boundary is the same calendar date for all horizons. Training labels are purged wherever their information intervals overlap validation or the locked holdout.
- Candidate, threshold, and stacked-meta selection use development data only.
- Hurst is treated as a possible regime descriptor—not a directional predictor—and is tested against no-Hurst and legacy-estimator ablations.
- The 2023–2026 holdout has been seen before. Results are therefore repeated historical out-of-sample evidence, not pristine confirmation.

The default market configuration uses the checked XAUUSD D1 export from MetaQuotes-Demo. Its `volume` is broker tick volume, not centralized exchange volume. Broker server, timezone, holidays, and candle boundaries can change D1 bars, features, labels, and results.

## Run

```powershell
uv sync --extra dev --extra mt5
uv run hge-gold --config configs/thesis.yaml
```

Software-only smoke test:

```powershell
uv run hge-gold --config configs/thesis_quick.yaml
```

The quick configuration is synthetic and is never market evidence.

## MetaTrader 5 verification

The pipeline writes one frozen signal file per horizon under `data/predictions/v3/market`. Compile and replay a horizon without optimization:

```powershell
uv run python scripts/run_mt5_backtest.py `
  --terminal "D:\Programming\MetaTrader\MetaTrader 5\terminal64.exe" `
  --signals data/predictions/v3/market/mt5_replay_signals_h1.csv `
  --from-date 2023.07.03 `
  --to-date 2026.08.01 `
  --output-dir artifacts/mt5/h1 `
  --close-running-terminal
```

`mt5/HGE_SignalReplay.mq5` validates execution of already-frozen predictions. It is intentionally not native MQL5 inference and does not tune anything in the tester. See `mt5/README.md` for its contract.

## Principal outputs

- `artifacts/v3/market/execution_manifest.json`: source/config/code/runtime provenance and artifact hashes.
- `artifacts/v3/market/locked_test_metrics.csv`: classification metrics and block-bootstrap uncertainty.
- `artifacts/v3/market/ablation_metrics.csv`: formal Hurst/no-Hurst and stacker/no-stacker comparisons.
- `artifacts/v3/market/backtest_summary.csv`: cost-aware model, trend, volatility-filtered, long/short/cash, and buy-and-hold benchmarks.
- `artifacts/v3/market/cpcv_split_manifest.csv`: development-only combinatorial split audit.
- `artifacts/v3/market/validation_diagnostics.json`: implemented and explicitly deferred statistics.
- `artifacts/runs/<run-id>/`: immutable evidence copy; a trackable receipt is also written under `artifacts/v3/market`.
- `artifacts/mt5/mt5_evidence_index.json`: hashed MT5 compiler/tester evidence linked to the pipeline run.
- `data/predictions/v3/market/mt5_replay_signals_h*.csv`: fixed MT5 replay inputs.

## Quality checks

```powershell
uv run ruff check .
uv run mypy src/hge_gold
uv run pytest --cov=src/hge_gold --cov-report=term-missing
```

The acceptance goal of 60% balanced accuracy is a preregistered threshold, not a promised outcome. See `docs/research_framework_v3_report.md` for the evidence, implemented changes, and deferred work.

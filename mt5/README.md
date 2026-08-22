# MT5 frozen-signal execution validation

`HGE_SignalReplay.mq5` is a fixed-signal execution validator for MetaTrader 5. It does not train, tune, or load a Python/joblib model. Its CSV contract is exactly:

```text
entry_time,exit_time,direction,signal_id
```

Rows must have sorted unique entry times, a later exit, direction `-1` or `1`, and unique IDs. The EA fails closed on malformed data, requires D1, acts only at a new bar open, holds at most one position, and closes at the frozen exit bar open.

Run a replay after generating the market pipeline artifacts:

```powershell
uv run python scripts/run_mt5_backtest.py `
  --terminal "D:\Programming\MetaTrader\MetaTrader 5\terminal64.exe" `
  --signals data/predictions/v3/market/mt5_replay_signals_h1.csv `
  --from-date 2023.07.03 `
  --to-date 2026.08.01 `
  --output-dir artifacts/mt5/h1 `
  --close-running-terminal
```

`--close-running-terminal` explicitly authorizes the runner to close only the specified MT5 installation after collecting metadata; this is required because the MetaTrader5 Python API starts/attaches to a terminal and `shutdown()` only disconnects the API. Save other terminal work first.

The default tester model is `2` (Open prices only), which exactly represents this bar-open-only EA. Model `4` can be requested for a real-tick replay when the full period's tick archives are available. Optimization, remote agents, and cloud agents remain disabled.

Each output directory contains the original MT5 HTML report, UTF-16 tester configuration, and `mt5_evidence.json` with signal/EA/report hashes, terminal and broker metadata, tester settings, and parsed statistics.

This validates broker/tester execution of frozen signals. It is not native MQL5 inference, live trading, or a profitability claim. The historical 2023–2026 evaluation period is `historical_holdout_v1_previously_revealed` and must not be used for optimization. A new prospective period is required for confirmation.

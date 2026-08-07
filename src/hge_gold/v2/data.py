from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def generate_research_sample(n_rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic OHLCV data with learnable but non-trivial regimes.

    This fixture validates software behavior only. It is not market evidence.
    """
    if n_rows < 700:
        raise ValueError("Research sample requires at least 700 rows")
    rng = np.random.default_rng(seed)
    state = np.empty(n_rows, dtype=int)
    state[0] = 1
    for index in range(1, n_rows):
        if rng.random() < 0.965:
            state[index] = state[index - 1]
        else:
            state[index] = -state[index - 1]
    volatility_state = np.where(np.sin(np.arange(n_rows) / 55.0) > 0, 1.0, 1.45)
    innovation = rng.normal(0.0, 0.0038 * volatility_state, n_rows)
    lagged_state = np.roll(state, 1)
    lagged_state[0] = state[0]
    returns = 0.0034 * lagged_state + 0.18 * np.roll(innovation, 1) + innovation
    returns[0] = innovation[0]
    close = 1250.0 * np.exp(np.cumsum(returns))
    overnight = rng.normal(0.0, 0.0013, n_rows)
    open_ = close * np.exp(overnight)
    intraday = np.abs(rng.normal(0.0045, 0.0018, n_rows)) * volatility_state
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * np.maximum(0.70, 1.0 - intraday)
    volume = rng.lognormal(11.4 + 0.15 * (volatility_state - 1.0), 0.28, n_rows)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2014-01-02", periods=n_rows),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def load_ohlcv(source_csv: Path | None, min_rows: int, seed: int = 42) -> pd.DataFrame:
    frame = generate_research_sample(max(min_rows, 1500), seed) if source_csv is None else pd.read_csv(source_csv)
    return normalize_and_validate(frame, min_rows=min_rows)


def normalize_and_validate(frame: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    result = frame.loc[:, REQUIRED_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=False)
    for column in REQUIRED_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("Invalid date values detected")
    if len(result) < min_rows:
        raise ValueError(f"At least {min_rows} rows are required; received {len(result)}")
    if result["date"].duplicated().any():
        raise ValueError("Duplicate timestamps are forbidden")
    if not result["date"].is_monotonic_increasing:
        raise ValueError("Rows must be strictly chronological; sort the source before ingestion")
    numeric = result.loc[:, REQUIRED_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("NaN or infinite OHLCV values detected")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (result["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    row_max = result[["open", "close", "low"]].max(axis=1)
    row_min = result[["open", "close", "high"]].min(axis=1)
    if (result["high"] < row_max).any() or (result["low"] > row_min).any():
        raise ValueError("OHLC high/low invariants failed")
    result = result.reset_index(drop=True)
    result.insert(0, "row_id", np.arange(len(result), dtype=int))
    return result

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("date", "open", "high", "low", "close", "volume")
MT5_VOLUME_COLUMNS = ("tick_volume", "real_volume")
MT5_TAB_COLUMN_MAP = {
    "<DATE>": "date",
    "<OPEN>": "open",
    "<HIGH>": "high",
    "<LOW>": "low",
    "<CLOSE>": "close",
    "<TICKVOL>": "tick_volume",
    "<VOL>": "real_volume",
    "<SPREAD>": "spread",
}


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


def canonicalize_mt5_volume(
    frame: pd.DataFrame,
    volume_column: str,
) -> pd.DataFrame:
    """Map one explicitly selected MT5 volume field to canonical ``volume``.

    MT5 exposes both broker tick counts and, for some instruments and periods,
    exchange-reported volume.  They are not interchangeable, so this function
    never falls back from one field to the other.
    """
    if volume_column not in MT5_VOLUME_COLUMNS:
        raise ValueError(
            f"volume_column must be one of {list(MT5_VOLUME_COLUMNS)}; received {volume_column!r}"
        )
    if volume_column not in frame.columns:
        raise ValueError(f"Selected MT5 volume column is missing: {volume_column}")

    result = frame.copy()
    selected = pd.to_numeric(result[volume_column], errors="coerce")
    if "volume" in result.columns:
        canonical = pd.to_numeric(result["volume"], errors="coerce")
        comparable = selected.notna() & canonical.notna()
        if not np.allclose(
            selected.loc[comparable].to_numpy(dtype=float),
            canonical.loc[comparable].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "Canonical volume conflicts with the explicitly selected MT5 volume column"
            )
    result["volume"] = selected
    return result


def load_mt5_tab_export(path: Path, min_rows: int = 1) -> pd.DataFrame:
    """Deterministically transform a native MT5 tab export into canonical OHLCV."""
    raw = pd.read_csv(path, sep="\t")
    unknown_required = {
        "<DATE>",
        "<OPEN>",
        "<HIGH>",
        "<LOW>",
        "<CLOSE>",
        "<TICKVOL>",
    } - set(raw.columns)
    if unknown_required:
        raise ValueError(f"MT5 tab export is missing columns: {sorted(unknown_required)}")
    renamed = raw.rename(columns=MT5_TAB_COLUMN_MAP)
    canonical = normalize_and_validate(
        renamed,
        min_rows=min_rows,
        volume_column="tick_volume",
    )
    return canonical


def load_ohlcv(
    source_csv: Path | None,
    min_rows: int,
    seed: int = 42,
    volume_column: str | None = None,
) -> pd.DataFrame:
    frame = (
        generate_research_sample(max(min_rows, 1500), seed)
        if source_csv is None
        else pd.read_csv(source_csv)
    )
    return normalize_and_validate(frame, min_rows=min_rows, volume_column=volume_column)


def normalize_and_validate(
    frame: pd.DataFrame,
    min_rows: int,
    volume_column: str | None = None,
) -> pd.DataFrame:
    if volume_column is not None:
        frame = canonicalize_mt5_volume(frame, volume_column)
    elif "volume" not in frame.columns and any(
        column in frame.columns for column in MT5_VOLUME_COLUMNS
    ):
        raise ValueError(
            "MT5 input requires an explicit volume_column='tick_volume' or "
            "volume_column='real_volume'; implicit mixing is forbidden"
        )
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

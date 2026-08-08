from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LONG_CALENDAR_GAP_DAYS = 4
_RETURN_ROBUST_Z_THRESHOLD = 6.0
_FLASH_MOVE_LOG_RETURN_THRESHOLD = 0.05
_VOLUME_WINDOW = 63
_VOLUME_REGIME_RATIO_THRESHOLD = 3.0
_VOLUME_SCALE_SHIFT_RATIO_THRESHOLD = 5.0


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _record_suspicious_rows(
    source: pd.DataFrame,
    mask: pd.Series,
    issue_type: str,
    values: pd.Series,
    threshold: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in source.index[mask]:
        rows.append(
            {
                "row_id": int(source.at[index, "row_id"]),
                "date": source.at[index, "date"].isoformat(),
                "issue_type": issue_type,
                "value": float(values.at[index]),
                "threshold": threshold,
            }
        )
    return rows


def _yearly_statistics(
    source: pd.DataFrame,
    log_returns: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    frame = source[["date", "close", "volume"]].copy()
    frame["year"] = frame["date"].dt.year
    frame["log_return"] = log_returns
    frame["regime"] = regimes
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        returns = group["log_return"].dropna()
        prices = group["close"]
        rows.append(
            {
                "year": int(year),
                "observations": int(len(group)),
                "start_date": group["date"].iloc[0].isoformat(),
                "end_date": group["date"].iloc[-1].isoformat(),
                "close_min": float(prices.min()),
                "close_p05": float(prices.quantile(0.05)),
                "close_median": float(prices.median()),
                "close_mean": float(prices.mean()),
                "close_p95": float(prices.quantile(0.95)),
                "close_max": float(prices.max()),
                "close_std": float(prices.std(ddof=0)),
                "return_mean": float(returns.mean()) if not returns.empty else float("nan"),
                "return_std": float(returns.std(ddof=0)) if not returns.empty else float("nan"),
                "return_p01": float(returns.quantile(0.01)) if not returns.empty else float("nan"),
                "return_p05": float(returns.quantile(0.05)) if not returns.empty else float("nan"),
                "return_median": float(returns.median()) if not returns.empty else float("nan"),
                "return_p95": float(returns.quantile(0.95)) if not returns.empty else float("nan"),
                "return_p99": float(returns.quantile(0.99)) if not returns.empty else float("nan"),
                "volume_mean": float(group["volume"].mean()),
                "volume_median": float(group["volume"].median()),
                "volume_zero_rows": int((group["volume"] == 0).sum()),
                "bull_rows": int((group["regime"] == "bull").sum()),
                "bear_rows": int((group["regime"] == "bear").sum()),
                "unclassified_regime_rows": int((group["regime"] == "unclassified").sum()),
            }
        )
    return pd.DataFrame(rows)


def _horizon_class_balance(source: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    log_close = np.log(source["close"].astype(float))
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        forward_log_return = log_close.shift(-horizon) - log_close
        labeled = forward_log_return.dropna()
        up_count = int((labeled > 0.0).sum())
        down_count = int((labeled <= 0.0).sum())
        total = int(len(labeled))
        rows.append(
            {
                "horizon": int(horizon),
                "n_labeled": total,
                "up_count": up_count,
                "down_count": down_count,
                "up_rate": float(up_count / total) if total else float("nan"),
                "down_rate": float(down_count / total) if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_market_data_quality_audit(
    source: pd.DataFrame,
    horizons: tuple[int, ...],
    regime_window: int,
    output_dir: Path,
) -> dict[str, Path]:
    """Write descriptive market-data checks without altering the modeling dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(source["date"])
    close = source["close"].astype(float)
    volume = source["volume"].astype(float)
    log_returns = np.log(close).diff()
    calendar_gap_days = dates.diff().dt.days
    expected_weekdays = pd.bdate_range(dates.iloc[0].normalize(), dates.iloc[-1].normalize())
    observed_days = pd.DatetimeIndex(dates.dt.normalize())
    missing_weekdays = expected_weekdays.difference(observed_days)

    return_median = float(log_returns.median())
    return_mad = float((log_returns - return_median).abs().median())
    if return_mad > 0.0:
        robust_return_z = 0.6745 * (log_returns - return_median) / return_mad
    else:
        robust_return_z = pd.Series(0.0, index=source.index)

    rolling_volume_median = volume.rolling(_VOLUME_WINDOW, min_periods=_VOLUME_WINDOW).median()
    volume_ratio = volume / rolling_volume_median
    volume_regime = (volume_ratio >= _VOLUME_REGIME_RATIO_THRESHOLD) | (
        volume_ratio <= 1.0 / _VOLUME_REGIME_RATIO_THRESHOLD
    )
    volume_scale_ratio = rolling_volume_median / rolling_volume_median.shift(_VOLUME_WINDOW)
    volume_scale_shift = (volume_scale_ratio >= _VOLUME_SCALE_SHIFT_RATIO_THRESHOLD) | (
        volume_scale_ratio <= 1.0 / _VOLUME_SCALE_SHIFT_RATIO_THRESHOLD
    )

    trailing_price_mean = close.rolling(regime_window, min_periods=regime_window).mean().shift(1)
    regimes = pd.Series(
        np.where(
            trailing_price_mean.isna(),
            "unclassified",
            np.where(close >= trailing_price_mean, "bull", "bear"),
        ),
        index=source.index,
    )

    weekend_rows = dates.dt.dayofweek >= 5
    long_gaps = calendar_gap_days > _LONG_CALENDAR_GAP_DAYS
    outlier_returns = robust_return_z.abs() >= _RETURN_ROBUST_Z_THRESHOLD
    flash_moves = log_returns.abs() >= _FLASH_MOVE_LOG_RETURN_THRESHOLD
    zero_volume = volume == 0.0
    suspicious_rows: list[dict[str, Any]] = []
    suspicious_rows.extend(
        _record_suspicious_rows(
            source,
            weekend_rows,
            "weekend_row",
            dates.dt.dayofweek.astype(float),
            5.0,
        )
    )
    suspicious_rows.extend(
        _record_suspicious_rows(
            source,
            long_gaps.fillna(False),
            "long_calendar_gap_days",
            calendar_gap_days.fillna(0.0),
            float(_LONG_CALENDAR_GAP_DAYS),
        )
    )
    suspicious_rows.extend(
        _record_suspicious_rows(
            source,
            outlier_returns.fillna(False),
            "outlier_return_robust_z",
            robust_return_z.fillna(0.0),
            _RETURN_ROBUST_Z_THRESHOLD,
        )
    )
    suspicious_rows.extend(
        _record_suspicious_rows(
            source,
            flash_moves.fillna(False),
            "flash_move_log_return",
            log_returns.fillna(0.0),
            _FLASH_MOVE_LOG_RETURN_THRESHOLD,
        )
    )
    suspicious_rows.extend(
        _record_suspicious_rows(source, zero_volume, "zero_volume", volume, 0.0)
    )
    suspicious_rows.extend(
        _record_suspicious_rows(
            source,
            volume_regime.fillna(False),
            "volume_regime_ratio",
            volume_ratio.fillna(0.0),
            _VOLUME_REGIME_RATIO_THRESHOLD,
        )
    )
    suspicious_rows.extend(
        _record_suspicious_rows(
            source,
            volume_scale_shift.fillna(False),
            "volume_scale_shift_ratio",
            volume_scale_ratio.fillna(0.0),
            _VOLUME_SCALE_SHIFT_RATIO_THRESHOLD,
        )
    )
    suspicious_rows.extend(
        {
            "row_id": None,
            "date": day.isoformat(),
            "issue_type": "potential_missing_weekday",
            "value": None,
            "threshold": None,
        }
        for day in missing_weekdays
    )

    yearly_path = output_dir / "yearly_statistics.csv"
    class_balance_path = output_dir / "horizon_class_balance.csv"
    suspicious_path = output_dir / "suspicious_rows.csv"
    summary_path = output_dir / "summary.json"
    _yearly_statistics(source, log_returns, regimes).to_csv(yearly_path, index=False)
    _horizon_class_balance(source, horizons).to_csv(class_balance_path, index=False)
    pd.DataFrame(
        suspicious_rows,
        columns=["row_id", "date", "issue_type", "value", "threshold"],
    ).to_csv(suspicious_path, index=False)

    summary = {
        "audit_version": "market_data_quality_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_rows": int(len(source)),
        "source_start": dates.iloc[0].isoformat(),
        "source_end": dates.iloc[-1].isoformat(),
        "calendar": {
            "weekend_row_count": int(weekend_rows.sum()),
            "potential_missing_weekday_count": int(len(missing_weekdays)),
            "long_calendar_gap_threshold_days": _LONG_CALENDAR_GAP_DAYS,
            "long_calendar_gap_count": int(long_gaps.sum()),
            "maximum_calendar_gap_days": int(calendar_gap_days.max()),
        },
        "returns": {
            "robust_z_threshold": _RETURN_ROBUST_Z_THRESHOLD,
            "outlier_return_count": int(outlier_returns.sum()),
            "flash_move_log_return_threshold": _FLASH_MOVE_LOG_RETURN_THRESHOLD,
            "flash_move_count": int(flash_moves.sum()),
        },
        "volume": {
            "zero_volume_count": int(zero_volume.sum()),
            "rolling_median_window": _VOLUME_WINDOW,
            "regime_ratio_threshold": _VOLUME_REGIME_RATIO_THRESHOLD,
            "regime_change_count": int(volume_regime.sum()),
            "scale_shift_ratio_threshold": _VOLUME_SCALE_SHIFT_RATIO_THRESHOLD,
            "scale_shift_count": int(volume_scale_shift.sum()),
        },
        "bull_bear_regimes": {
            "trailing_mean_window": int(regime_window),
            "bull_count": int((regimes == "bull").sum()),
            "bear_count": int((regimes == "bear").sum()),
            "unclassified_count": int((regimes == "unclassified").sum()),
        },
        "suspicious_row_count": int(len(suspicious_rows)),
        "artifacts": {
            "yearly_statistics": yearly_path.name,
            "horizon_class_balance": class_balance_path.name,
            "suspicious_rows": suspicious_path.name,
        },
    }
    _atomic_json(summary_path, summary)
    return {
        "data_quality_summary": summary_path,
        "yearly_statistics": yearly_path,
        "horizon_class_balance": class_balance_path,
        "suspicious_rows": suspicious_path,
    }

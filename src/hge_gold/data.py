from __future__ import annotations

import json
from datetime import UTC, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import sha256_file, write_csv, write_parquet

GOLD_REQUIRED = {
    "row_id",
    "date",
    "date_index",
    "gc_open",
    "gc_high",
    "gc_low",
    "gc_close",
    "gc_price_for_return",
    "gc_price_field_for_return",
    "available_timestamp_utc",
    "decision_timestamp_utc",
    "provenance_run_id",
    "source_snapshot_id",
}


def generate_sample_market_data(n_rows: int = 900, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic, non-confidential OHLCV data with regime changes."""
    if n_rows < 500:
        raise ValueError("Sample dataset must contain at least 500 rows for all locked horizons")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_rows)
    regimes = np.select(
        [np.arange(n_rows) < n_rows * 0.35, np.arange(n_rows) < n_rows * 0.7],
        [0, 1],
        default=2,
    )
    drift = np.choose(regimes, [0.00015, -0.00005, 0.00025])
    sigma = np.choose(regimes, [0.008, 0.014, 0.010])
    shocks = rng.normal(drift, sigma)
    shocks[::197] -= 0.035
    close = 1250.0 * np.exp(np.cumsum(shocks))
    overnight = rng.normal(0, 0.002, n_rows)
    open_ = close * np.exp(overnight)
    intraday = np.abs(rng.normal(0.006, 0.003, n_rows))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * np.maximum(0.8, 1 - intraday)
    volume = rng.lognormal(mean=11.5, sigma=0.35, size=n_rows).round()
    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def validate_source(frame: pd.DataFrame) -> None:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid date values")
    if dates.duplicated().any():
        raise ValueError("Duplicate timestamps are forbidden")
    if not dates.is_monotonic_increasing:
        raise ValueError("Source timestamps must be strictly ordered")
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("NaN or Infinity detected in required source fields")
    if (numeric[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("OHLC high invariant failed")
    if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("OHLC low invariant failed")


def _utc_close(date: pd.Timestamp, hour: int) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(date.date(), time(hour, 0, tzinfo=UTC)))


def build_phase1(config: PipelineConfig, source_csv: Path | None = None) -> dict[str, Path]:
    paths = config.paths()
    raw_dir = paths.data / "raw" / "sample"
    processed = paths.data / "processed"
    metadata = paths.artifacts / "metadata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    metadata.mkdir(parents=True, exist_ok=True)

    if source_csv is None:
        source = generate_sample_market_data(seed=int(config.project["seed"]))
        source_name = "deterministic_sample_fixture"
    else:
        source = pd.read_csv(source_csv)
        source_name = "user_csv"
    source["date"] = pd.to_datetime(source["date"])
    source = source.sort_values("date").reset_index(drop=True)
    validate_source(source)
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"GC_{run_stamp}.csv"
    write_csv(raw_path, source)
    raw_hash = sha256_file(raw_path)
    provenance_id = raw_hash[:24]
    decision_hour = int(config.data["decision_hour_utc"])
    close_ts = source["date"].map(lambda value: _utc_close(value, decision_hour))
    quality_flags = json.dumps(
        {
            "flags": [
                {
                    "code": "SAMPLE_DATA",
                    "severity": "INFO",
                    "source": "phase1",
                    "details": (
                        "Non-confidential deterministic data used for verified local execution"
                    ),
                }
            ]
        }
    )

    gold = pd.DataFrame(
        {
            "row_id": np.arange(len(source), dtype=int),
            "date": source["date"].dt.normalize(),
            "date_index": np.arange(len(source), dtype=int),
            "gc_open": source["open"].astype(float),
            "gc_high": source["high"].astype(float),
            "gc_low": source["low"].astype(float),
            "gc_close": source["close"].astype(float),
            "gc_settlement": np.nan,
            "gc_volume": source["volume"].astype(float),
            "gc_open_interest": np.nan,
            "gc_price_for_return": source["close"].astype(float),
            "gc_price_field_for_return": "close",
            "gc_log_return_close": np.log(source["close"]).diff(),
            "gc_log_return_settlement": np.nan,
            "gc_log_return_selected": np.log(source["close"]).diff(),
            "gc_source": source_name,
            "gc_source_symbol": "GC",
            "gc_contract_symbol": "GC_CONTINUOUS_SAMPLE",
            "observation_date": source["date"].dt.normalize(),
            "source_timestamp_utc": close_ts,
            "available_timestamp_utc": close_ts,
            "decision_timestamp_utc": close_ts,
            "execution_timestamp_utc": close_ts + pd.Timedelta(days=1),
            "timestamp_policy": "close_assumption_mvp",
            "timestamp_quality": "assumption_not_exchange_verified",
            "is_roll_day": False,
            "suspected_roll_window": False,
            "confirmed_roll_day": False,
            "return_anomaly_candidate": np.log(source["close"]).diff().abs().gt(0.05),
            "roll_evidence_type": "none",
            "roll_method": "vendor_continuous_unknown",
            "adjustment_method": "unknown",
            "continuous_construction": "simulated_continuous_fixture",
            "is_back_adjusted": True,
            "gc_mvp_quality_score": 95.0,
            "gc_paper_readiness_score": 15.0,
            "gc_quality_grade": "MVP_ONLY",
            "gc_quality_flags": quality_flags,
            "provenance_run_id": provenance_id,
            "source_snapshot_id": raw_hash,
            "download_timestamp_utc": datetime.now(UTC).isoformat(),
        }
    )
    if not GOLD_REQUIRED.issubset(gold.columns):
        raise RuntimeError("Internal gold schema construction failed")
    if (gold["available_timestamp_utc"] > gold["decision_timestamp_utc"]).any():
        raise RuntimeError("Timestamp availability invariant failed")

    calendar = pd.DataFrame(
        {
            "row_id": gold["row_id"],
            "date": gold["date"],
            "date_index": gold["date_index"],
            "is_gold_trading_day": True,
            "calendar_source": "empirical_primary_source",
            "calendar_quality": "mvp_empirical",
            "decision_timestamp_utc": close_ts,
            "execution_timestamp_utc": close_ts + pd.Timedelta(days=1),
        }
    )
    for horizon in [1, 5, 10, 20]:
        calendar[f"next_trading_date_{horizon}"] = gold["date"].shift(-horizon)
        calendar[f"previous_trading_date_{horizon}"] = gold["date"].shift(horizon)

    interval_rows: list[dict[str, object]] = []
    dates = gold["date"].tolist()
    for index, date in enumerate(dates):
        for horizon in [1, 5, 10, 20]:
            label_end_i = index + horizon
            embargo_end_i = label_end_i + horizon
            computable = label_end_i < len(dates)
            embargo_ok = embargo_end_i < len(dates)
            drop = (
                "none"
                if computable and embargo_ok
                else (
                    "insufficient_future_horizon"
                    if not computable
                    else "insufficient_embargo_window"
                )
            )
            interval_rows.append(
                {
                    "row_id": index,
                    "date": date,
                    "horizon": horizon,
                    "label_start_date": dates[index + 1] if index + 1 < len(dates) else pd.NaT,
                    "label_end_date": dates[label_end_i] if computable else pd.NaT,
                    "label_start_date_index": index + 1 if index + 1 < len(dates) else np.nan,
                    "label_end_date_index": label_end_i if computable else np.nan,
                    "purge_start_date": dates[index + 1] if index + 1 < len(dates) else pd.NaT,
                    "purge_end_date": dates[label_end_i] if computable else pd.NaT,
                    "embargo_start_date": dates[label_end_i] if computable else pd.NaT,
                    "embargo_end_date": dates[embargo_end_i] if embargo_ok else pd.NaT,
                    "embargo_length": horizon,
                    "is_label_computable": computable,
                    "is_embargo_computable": embargo_ok,
                    "is_train_eligible": computable and embargo_ok,
                    "is_validation_eligible": computable and embargo_ok,
                    "is_test_eligible": computable,
                    "drop_reason": drop,
                }
            )
    intervals = pd.DataFrame(interval_rows)

    gold_path = processed / "gold_futures_daily.parquet"
    calendar_path = processed / "trading_calendar.parquet"
    intervals_path = processed / "label_interval_template.parquet"
    price_path = processed / "prices" / "validated_gold_price_series.parquet"
    write_parquet(gold_path, gold)
    write_parquet(calendar_path, calendar)
    write_parquet(intervals_path, intervals)
    write_parquet(price_path, gold[["date", "gc_close"]].rename(columns={"gc_close": "close"}))

    quality = pd.DataFrame(
        [
            {
                "source": source_name,
                "n_rows": len(gold),
                "duplicate_date_rate": 0.0,
                "missing_selected_price_rate": 0.0,
                "ohlc_violation_rate": 0.0,
                "final_mvp_data_quality_score": 95.0,
                "paper_grade_readiness_score": 15.0,
                "mvp_status": "CLOSED_FOR_MVP",
                "paper_grade_status": "NOT_READY",
            }
        ]
    )
    quality_path = metadata / "phase1_data_quality_report.csv"
    write_csv(quality_path, quality)
    return {
        "gold": gold_path,
        "calendar": calendar_path,
        "intervals": intervals_path,
        "prices": price_path,
        "quality": quality_path,
    }

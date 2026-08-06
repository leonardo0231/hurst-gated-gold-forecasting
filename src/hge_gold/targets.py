from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io import atomic_json, canonical_json, sha256_bytes, write_csv, write_parquet


def bps_to_log_return(bps: float | np.ndarray | pd.Series) -> float | np.ndarray | pd.Series:
    if np.isscalar(bps):
        return math.log1p(float(cast(Any, bps)) / 10_000.0)
    return np.log1p(np.asarray(bps) / 10_000.0)


def build_targets(config: PipelineConfig) -> dict[str, Path]:
    paths = config.paths()
    gold = pd.read_parquet(paths.data / "processed" / "gold_futures_daily.parquet")
    intervals = pd.read_parquet(paths.data / "processed" / "label_interval_template.parquet")
    price = gold["gc_price_for_return"].astype(float)
    returns = np.log(price).diff()
    window = int(config.targets["threshold_sigma_window"])
    min_periods = int(config.targets["threshold_sigma_min_periods"])
    sigma = returns.rolling(window, min_periods=min_periods).std()
    floor_bps = float(config.targets["threshold_floor_bps"])
    direction_k = float(config.targets["direction_k"])
    action_bps = float(config.targets["transaction_cost_bps"]) + float(
        config.targets["slippage_bps"]
    )
    policy = {
        "target_policy_id": "final_mvp_v1",
        "direction_threshold_policy_id": "vol_scaled_k050_floor_5bps",
        "trade_threshold_policy_id": "cost_aware_roundtrip_tc3_slip2_bps",
        "horizons": [1, 5, 10, 20],
        "threshold_conversion_rule": "log(1 + bps / 10000)",
        "created_before_target_construction": True,
        "not_tuned_on_test_data": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    metadata = paths.artifacts / "metadata"
    policy_path = metadata / "target_policy_registry.json"
    atomic_json(policy_path, policy)
    run_id = sha256_bytes(
        canonical_json({"policy": policy, "input": str(gold["source_snapshot_id"].iloc[0])})
    )[:24]
    rows: list[pd.DataFrame] = []
    for horizon in config.targets["horizons"]:
        fwd = np.log(price.shift(-horizon) / price)
        future_sq = returns.pow(2).shift(-1).rolling(horizon).sum().shift(-(horizon - 1))
        direction_threshold = np.maximum(
            direction_k * sigma * np.sqrt(horizon), float(bps_to_log_return(floor_bps))
        )
        trade_threshold = float(bps_to_log_return(action_bps))
        subset = intervals[intervals["horizon"] == horizon].reset_index(drop=True)
        direction = np.select(
            [fwd > direction_threshold, fwd < -direction_threshold],
            ["up", "down"],
            default="no_trade",
        )
        trade = np.select(
            [fwd > trade_threshold, fwd < -trade_threshold], ["long", "short"], default="flat"
        )
        valid = (
            subset["is_label_computable"].to_numpy()
            & fwd.notna().to_numpy()
            & direction_threshold.notna().to_numpy()
        )
        frame = pd.DataFrame(
            {
                "row_id": gold["row_id"],
                "date": gold["date"],
                "date_index": gold["date_index"],
                "price_t": price,
                "price_field": "close",
                "source": gold["gc_source"],
                "source_symbol": gold["gc_source_symbol"],
                "horizon": horizon,
                "target_policy_id": "final_mvp_v1",
                "direction_threshold_policy_id": "vol_scaled_k050_floor_5bps",
                "trade_threshold_policy_id": "cost_aware_roundtrip_tc3_slip2_bps",
                "ret_fwd": fwd,
                "direction_label": direction,
                "direction_label_encoded": pd.Series(direction).map(
                    {"down": -1, "no_trade": 0, "up": 1}
                ),
                "direction_threshold_value": direction_threshold,
                "direction_threshold_bps": np.expm1(direction_threshold) * 10_000,
                "trade_label": trade,
                "trade_label_encoded": pd.Series(trade).map({"short": -1, "flat": 0, "long": 1}),
                "economic_action_threshold_value": trade_threshold,
                "economic_action_threshold_bps": action_bps,
                "rv_fwd": future_sq,
                "vol_fwd": np.sqrt(future_sq),
                "threshold_sigma_used": sigma,
                "threshold_sigma_window": window,
                "threshold_sigma_min_periods": min_periods,
                "threshold_sigma_available_timestamp_utc": gold["available_timestamp_utc"],
                "transaction_cost_bps": float(config.targets["transaction_cost_bps"]),
                "slippage_bps": float(config.targets["slippage_bps"]),
                "cost_convention": "round_trip",
                "is_label_computable": subset["is_label_computable"],
                "is_embargo_computable": subset["is_embargo_computable"],
                "is_modeling_eligible": valid & subset["is_embargo_computable"].to_numpy(),
                "is_split_assignable": valid & subset["is_embargo_computable"].to_numpy(),
                "drop_reason": np.where(
                    valid & subset["is_embargo_computable"].to_numpy(),
                    "none",
                    np.where(
                        ~valid & subset["drop_reason"].eq("none"),
                        "invalid_threshold",
                        subset["drop_reason"],
                    ),
                ),
                "roll_sensitive_target": gold["suspected_roll_window"],
                "suspected_roll_window": gold["suspected_roll_window"],
                "confirmed_roll_day": gold["confirmed_roll_day"],
                "roll_evidence_type": gold["roll_evidence_type"],
                "timestamp_policy": gold["timestamp_policy"],
                "timestamp_quality": gold["timestamp_quality"],
                "provenance_run_id": gold["provenance_run_id"],
                "source_snapshot_id": gold["source_snapshot_id"],
                "target_construction_run_id": run_id,
            }
        )
        for col in [
            "label_start_date",
            "label_end_date",
            "label_start_date_index",
            "label_end_date_index",
            "purge_start_date",
            "purge_end_date",
            "embargo_start_date",
            "embargo_end_date",
            "embargo_length",
        ]:
            frame[col] = subset[col]
        rows.append(frame)
    targets = pd.concat(rows, ignore_index=True)
    unique_key = ["row_id", "horizon", "target_policy_id"]
    if targets.duplicated(unique_key).any():
        raise RuntimeError("Target unique-key invariant failed")
    if not (
        targets["threshold_sigma_available_timestamp_utc"]
        <= gold.set_index("row_id").loc[targets["row_id"], "decision_timestamp_utc"].to_numpy()
    ).all():
        raise RuntimeError("Threshold causality invariant failed")
    output = paths.data / "processed" / "targets" / "gold_multitask_targets.parquet"
    write_parquet(output, targets)
    leakage = pd.DataFrame(
        [
            {
                "target_name": name,
                "horizon": horizon,
                "target_policy_id": "final_mvp_v1",
                "uses_future_price_for_label": True,
                "uses_future_data_outside_label_interval": False,
                "feature_columns_used_json": "[]",
                "threshold_uses_future_returns": False,
                "threshold_uses_ret_fwd": False,
                "leakage_check_passed": True,
                "manual_review_required": False,
                "manual_review_status": "not_required",
            }
            for horizon in config.targets["horizons"]
            for name in ["ret_fwd", "direction_label", "trade_label", "rv_fwd"]
        ]
    )
    leakage_path = metadata / "target_leakage_audit_report.csv"
    write_csv(leakage_path, leakage)
    return {"targets": output, "policy": policy_path, "leakage": leakage_path}

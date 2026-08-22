from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import FeatureConfig, TargetConfig


def bps_to_log_return(bps: float) -> float:
    return math.log1p(bps / 10_000.0)


def build_horizon_dataset(
    source: pd.DataFrame,
    features: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    target_config: TargetConfig,
    feature_config: FeatureConfig,
) -> pd.DataFrame:
    close = source["close"].astype(float)
    open_price = source["open"].astype(float)
    log_price = np.log(close)
    ret = log_price.diff()
    sigma = ret.rolling(
        target_config.volatility_window,
        min_periods=target_config.volatility_min_periods,
    ).std()
    statistical_forward_return = log_price.shift(-horizon) - log_price
    entry_offset = int(getattr(target_config, "execution_lag_bars", 1))
    exit_offset = entry_offset + horizon
    executable_forward_return = np.log(open_price.shift(-exit_offset)) - np.log(
        open_price.shift(-entry_offset)
    )
    adaptive_threshold = target_config.threshold_k * sigma * np.sqrt(horizon)
    floor = bps_to_log_return(target_config.threshold_floor_bps)
    # The V3 configuration declares transaction cost plus slippage as total
    # round-trip execution drag.  The separate actionable buffer is an
    # additional safety margin, never a learned or future-derived quantity.
    round_trip_cost_bps = float(target_config.transaction_cost_bps) + float(
        target_config.slippage_bps
    )
    actionable_cost_buffer_bps = float(getattr(target_config, "actionable_cost_buffer_bps", 0.0))
    economic_floor = bps_to_log_return(round_trip_cost_bps + actionable_cost_buffer_bps)
    threshold = pd.Series(
        np.maximum.reduce(
            [
                adaptive_threshold.to_numpy(dtype=float),
                np.full(len(source), floor, dtype=float),
                np.full(len(source), economic_floor, dtype=float),
            ]
        ),
        index=source.index,
    )

    actionable_direction = np.select(
        [executable_forward_return > threshold, executable_forward_return < -threshold],
        [1, -1],
        default=0,
    ).astype(float)
    actionable_direction = pd.Series(actionable_direction, index=source.index, dtype=float).where(
        executable_forward_return.notna() & threshold.notna()
    )
    statistical_direction = np.where(
        statistical_forward_return.notna(),
        (statistical_forward_return > 0.0).astype(float),
        np.nan,
    )
    executable_direction = np.where(
        executable_forward_return.notna(),
        (executable_forward_return > 0.0).astype(float),
        np.nan,
    )
    result = features.copy()
    result["horizon"] = int(horizon)
    result["decision_row_index"] = result["row_id"]
    result["decision_available_row_index"] = result["row_id"] + 1
    result["entry_row_index"] = result["row_id"] + entry_offset
    result["exit_row_index"] = result["row_id"] + exit_offset
    result["decision_bar_timestamp"] = source["date"]
    result["decision_timestamp"] = source["date"].shift(-1)
    result["entry_timestamp"] = source["date"].shift(-entry_offset)
    result["exit_timestamp"] = source["date"].shift(-exit_offset)
    # Legacy overlap boundary for the original close-to-close statistical
    # target.  New validation code should purge on executable_label_end_index.
    result["label_end_index"] = result["row_id"] + horizon
    result["statistical_label_end_index"] = result["label_end_index"]
    result["executable_label_end_index"] = result["exit_row_index"]
    result["statistical_forward_log_return"] = statistical_forward_return
    result["statistical_direction_binary"] = statistical_direction
    result["executable_forward_log_return"] = executable_forward_return
    result["executable_direction_binary"] = executable_direction
    result["executable_direction_three_class"] = actionable_direction
    result["round_trip_cost_bps"] = round_trip_cost_bps
    result["actionable_cost_buffer_bps"] = actionable_cost_buffer_bps
    # Backward-compatible aliases keep the current statistical pipeline
    # runnable while the two targets remain explicit in persisted datasets.
    result["forward_log_return"] = result["statistical_forward_log_return"]
    result["direction_threshold"] = threshold
    result["direction_threshold_bps"] = np.expm1(threshold) * 10_000.0
    result["direction_three_class"] = result["executable_direction_three_class"]
    result["actionable_direction_three_class"] = result["executable_direction_three_class"]
    result["actionable_threshold_log_return"] = result["direction_threshold"]
    result["actionable_threshold_bps"] = result["direction_threshold_bps"]
    result["direction_binary"] = result["statistical_direction_binary"]
    result["is_actionable"] = executable_forward_return.notna() & (
        (executable_forward_return > threshold) | (executable_forward_return < -threshold)
    )
    result["is_modeling_eligible"] = (
        result["statistical_direction_binary"].notna()
        & result["statistical_forward_log_return"].notna()
        & result["executable_direction_binary"].notna()
        & result["executable_forward_log_return"].notna()
        & result["direction_threshold"].notna()
        & (result["feature_coverage"] >= feature_config.min_feature_coverage)
        & (result["executable_label_end_index"] < len(source))
    )
    result["target_policy_id"] = "separated_statistical_executable_direction_v3"
    result["feature_set_id"] = "causal_gold_features_v3_dfa"
    keep = [
        "row_id",
        "date",
        "close",
        "volume",
        *feature_columns,
        "feature_coverage",
        "horizon",
        "decision_row_index",
        "decision_available_row_index",
        "entry_row_index",
        "exit_row_index",
        "decision_bar_timestamp",
        "decision_timestamp",
        "entry_timestamp",
        "exit_timestamp",
        "label_end_index",
        "statistical_label_end_index",
        "executable_label_end_index",
        "statistical_forward_log_return",
        "statistical_direction_binary",
        "executable_forward_log_return",
        "executable_direction_binary",
        "executable_direction_three_class",
        "round_trip_cost_bps",
        "actionable_cost_buffer_bps",
        "forward_log_return",
        "direction_threshold",
        "direction_threshold_bps",
        "direction_three_class",
        "actionable_direction_three_class",
        "actionable_threshold_log_return",
        "actionable_threshold_bps",
        "direction_binary",
        "is_actionable",
        "is_modeling_eligible",
        "target_policy_id",
        "feature_set_id",
    ]
    return result[keep]

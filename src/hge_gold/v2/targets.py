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
    log_price = np.log(close)
    ret = log_price.diff()
    sigma = ret.rolling(
        target_config.volatility_window,
        min_periods=target_config.volatility_min_periods,
    ).std()
    forward_return = log_price.shift(-horizon) - log_price
    adaptive_threshold = target_config.threshold_k * sigma * np.sqrt(horizon)
    floor = bps_to_log_return(target_config.threshold_floor_bps)
    threshold = pd.Series(np.maximum(adaptive_threshold, floor), index=source.index)

    y_three = np.select(
        [forward_return > threshold, forward_return < -threshold],
        [1, -1],
        default=0,
    ).astype(float)
    y_binary = np.where(forward_return.notna(), (forward_return > 0.0).astype(float), np.nan)
    result = features.copy()
    result["horizon"] = int(horizon)
    result["label_end_index"] = result["row_id"] + horizon
    result["forward_log_return"] = forward_return
    result["direction_threshold"] = threshold
    result["direction_threshold_bps"] = np.expm1(threshold) * 10_000.0
    result["direction_three_class"] = y_three
    result["direction_binary"] = y_binary
    result["is_actionable"] = forward_return.notna() & (
        (forward_return > threshold) | (forward_return < -threshold)
    )
    result["is_modeling_eligible"] = (
        result["direction_binary"].notna()
        & result["forward_log_return"].notna()
        & result["direction_threshold"].notna()
        & (result["feature_coverage"] >= feature_config.min_feature_coverage)
        & (result["label_end_index"] < len(source))
    )
    result["target_policy_id"] = "adaptive_actionable_direction_v2_1"
    result["feature_set_id"] = "causal_gold_features_v2"
    keep = [
        "row_id",
        "date",
        "close",
        "volume",
        *feature_columns,
        "feature_coverage",
        "horizon",
        "label_end_index",
        "forward_log_return",
        "direction_threshold",
        "direction_threshold_bps",
        "direction_three_class",
        "direction_binary",
        "is_actionable",
        "is_modeling_eligible",
        "target_policy_id",
        "feature_set_id",
    ]
    return result[keep]

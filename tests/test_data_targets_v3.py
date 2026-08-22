from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hge_gold.config import FeatureConfig, TargetConfig
from hge_gold.data import canonicalize_mt5_volume, load_mt5_tab_export, normalize_and_validate
from hge_gold.targets import build_horizon_dataset


def _source(open_prices: list[float]) -> pd.DataFrame:
    opens = np.asarray(open_prices, dtype=float)
    closes = opens * np.linspace(1.0000, 1.0009, len(opens))
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=len(opens), tz="UTC"),
            "open": opens,
            "high": np.maximum(opens, closes) * 1.001,
            "low": np.minimum(opens, closes) * 0.999,
            "close": closes,
            "volume": np.arange(len(opens), dtype=float) + 100.0,
        }
    )
    return normalize_and_validate(frame, min_rows=1)


def _features(source: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = source[["row_id", "date", "close", "volume"]].copy()
    features["known_feature"] = np.arange(len(source), dtype=float)
    features["feature_coverage"] = 1.0
    return features, ["known_feature"]


def _targets(
    source: pd.DataFrame,
    *,
    horizon: int,
    transaction_cost_bps: float = 1.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    features, columns = _features(source)
    return build_horizon_dataset(
        source,
        features,
        columns,
        horizon,
        TargetConfig(
            horizons=(horizon,),
            volatility_window=2,
            volatility_min_periods=2,
            threshold_k=0.0,
            threshold_floor_bps=0.0,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        ),
        FeatureConfig(min_feature_coverage=1.0),
    )


def test_mt5_volume_mapping_is_explicit_and_never_falls_back() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=3, tz="UTC"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "tick_volume": [10, 20, 30],
            "real_volume": [1_000, 2_000, 3_000],
        }
    )

    with pytest.raises(ValueError, match="explicit volume_column"):
        normalize_and_validate(frame, min_rows=1)

    tick = normalize_and_validate(frame, min_rows=1, volume_column="tick_volume")
    real = normalize_and_validate(frame, min_rows=1, volume_column="real_volume")
    assert tick["volume"].tolist() == [10, 20, 30]
    assert real["volume"].tolist() == [1_000, 2_000, 3_000]

    conflicting = frame.assign(volume=[10, 999, 30])
    with pytest.raises(ValueError, match="conflicts"):
        canonicalize_mt5_volume(conflicting, "tick_volume")


def test_native_mt5_tab_export_has_executable_canonical_transformation(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "mt5.tsv"
    raw.write_text(
        "<DATE>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2024.01.02\t100\t102\t99\t101\t123\t0\t10\n"
        "2024.01.03\t101\t103\t100\t102\t456\t0\t12\n",
        encoding="utf-8",
    )

    result = load_mt5_tab_export(raw)

    assert result.columns.tolist() == ["row_id", "date", "open", "high", "low", "close", "volume"]
    assert result["volume"].tolist() == [123, 456]
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03"]


def test_statistical_and_executable_targets_have_exact_distinct_alignment() -> None:
    source = _source([100.0, 101.0, 103.0, 107.0, 109.0, 113.0, 127.0, 131.0])
    dataset = _targets(source, horizon=2)
    row = dataset.iloc[2]

    assert row["decision_row_index"] == 2
    assert row["decision_available_row_index"] == 3
    assert row["entry_row_index"] == 3
    assert row["exit_row_index"] == 5
    assert row["label_end_index"] == 4  # legacy close-to-close boundary
    assert row["executable_label_end_index"] == 5
    assert row["decision_bar_timestamp"] == source.loc[2, "date"]
    assert row["decision_timestamp"] == source.loc[3, "date"]
    assert row["entry_timestamp"] == source.loc[3, "date"]
    assert row["exit_timestamp"] == source.loc[5, "date"]
    assert row["statistical_forward_log_return"] == pytest.approx(
        math.log(source.loc[4, "close"] / source.loc[2, "close"])
    )
    assert row["executable_forward_log_return"] == pytest.approx(
        math.log(source.loc[5, "open"] / source.loc[3, "open"])
    )
    assert dataset.iloc[-4]["is_modeling_eligible"]
    assert not dataset.iloc[-3]["is_modeling_eligible"]
    assert not dataset.iloc[-2]["is_modeling_eligible"]
    assert not dataset.iloc[-1]["is_modeling_eligible"]


def test_realized_actionability_never_filters_otherwise_eligible_rows() -> None:
    source = _source([100.0, 100.0, 100.0, 100.0, 100.0, 102.0, 102.0, 102.0, 102.0])
    dataset = _targets(source, horizon=1)
    eligible = dataset[dataset["is_modeling_eligible"]]

    assert eligible["is_actionable"].any()
    assert (~eligible["is_actionable"]).any()
    assert eligible["direction_binary"].notna().all()
    assert eligible["executable_direction_three_class"].isin([-1.0, 0.0, 1.0]).all()
    assert (
        eligible.loc[~eligible["is_actionable"], "executable_direction_three_class"] == 0.0
    ).all()


def test_higher_round_trip_costs_cannot_increase_actionable_count() -> None:
    source = _source([100.0, 100.1, 100.3, 100.2, 100.6, 100.5, 101.0, 100.9, 101.4])
    low_cost = _targets(source, horizon=1, transaction_cost_bps=0.0, slippage_bps=0.0)
    high_cost = _targets(source, horizon=1, transaction_cost_bps=30.0, slippage_bps=20.0)

    valid_thresholds = high_cost["direction_threshold"].notna()
    assert (
        high_cost.loc[valid_thresholds, "direction_threshold"]
        .ge(low_cost.loc[valid_thresholds, "direction_threshold"])
        .all()
    )
    assert high_cost["is_actionable"].sum() <= low_cost["is_actionable"].sum()
    assert (high_cost["round_trip_cost_bps"] == 50.0).all()
    assert (high_cost["actionable_cost_buffer_bps"] == 2.0).all()

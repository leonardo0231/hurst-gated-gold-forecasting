from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hge_gold.data import build_phase1, generate_sample_market_data, validate_source
from hge_gold.features import build_features, hurst_rs
from hge_gold.targets import bps_to_log_return, build_targets


def test_sample_data_is_deterministic() -> None:
    pd.testing.assert_frame_equal(
        generate_sample_market_data(500, 42), generate_sample_market_data(500, 42)
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda frame: frame.assign(
                date=frame["date"].where(frame.index != 2, frame["date"].iloc[1])
            ),
            "Duplicate",
        ),
        (lambda frame: frame.sort_values("date", ascending=False), "ordered"),
        (
            lambda frame: frame.assign(close=np.where(frame.index == 0, np.nan, frame["close"])),
            "NaN",
        ),
        (lambda frame: frame.assign(low=np.where(frame.index == 0, -1, frame["low"])), "positive"),
    ],
)
def test_data_validation_rejects_edge_cases(mutation, match: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=match):
        validate_source(mutation(generate_sample_market_data(500)))


def test_bps_conversion_uses_log_rule() -> None:
    assert bps_to_log_return(5) == pytest.approx(np.log1p(5 / 10_000))


def test_phase1_target_feature_contracts(config) -> None:  # type: ignore[no-untyped-def]
    outputs1 = build_phase1(config)
    outputs2 = build_targets(config)
    outputs3 = build_features(config)
    gold = pd.read_parquet(outputs1["gold"])
    targets = pd.read_parquet(outputs2["targets"])
    matrix = pd.read_parquet(outputs3["matrix"])
    base = pd.read_parquet(outputs3["modeling_base"])
    assert gold["date"].is_unique
    assert (gold["available_timestamp_utc"] <= gold["decision_timestamp_utc"]).all()
    assert not targets.duplicated(["row_id", "horizon", "target_policy_id"]).any()
    first = targets[(targets["horizon"] == 5) & targets["ret_fwd"].notna()].iloc[0]
    prices = gold.set_index("row_id")["gc_price_for_return"]
    assert first.ret_fwd == pytest.approx(
        np.log(prices.loc[first.row_id + 5] / prices.loc[first.row_id])
    )
    assert set(targets["horizon"].unique()) == {1, 5, 10, 20}
    forbidden = {
        "ret_fwd",
        "direction_label",
        "trade_label",
        "rv_fwd",
        "vol_fwd",
        "target_policy_id",
    }
    assert forbidden.isdisjoint(matrix.columns)
    assert not base.duplicated(["row_id", "horizon", "target_policy_id", "feature_set_id"]).any()
    leakage = pd.read_csv(outputs3["leakage"])
    assert leakage["leakage_check_passed"].all()
    registry = json.loads(outputs3["registry"].read_text())
    assert registry["garch_features_status"] == "DEFERRED"


def test_hurst_estimator_bounds() -> None:
    path = np.cumsum(np.random.default_rng(42).normal(size=126))
    value = hurst_rs(path)
    assert 0 <= value <= 1

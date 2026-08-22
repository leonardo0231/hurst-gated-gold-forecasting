from __future__ import annotations

import pandas as pd
import pytest

from hge_gold.availability import (
    FeatureAvailabilityRecord,
    SourceAvailabilityRecord,
    backward_asof_feature_join,
)


def _policy() -> SourceAvailabilityRecord:
    return SourceAvailabilityRecord(
        source_id="xagusd_mt5",
        observation_timestamp_column="bar_open",
        available_at_column="available_at",
        timezone="UTC",
        availability_policy="completed bar is available at the next observed bar open",
        revision_policy="captured broker history; revisions require a new immutable snapshot",
        vintage_policy="snapshot hash identifies the only admissible vintage",
        volume_semantics="broker tick volume; not centralized exchange volume",
        evidence_status="test_fixture",
        admissible_for_development_selection=True,
    )


def _features() -> tuple[FeatureAvailabilityRecord, ...]:
    return (
        FeatureAvailabilityRecord(
            feature_name="silver_return_1",
            source_id="xagusd_mt5",
            value_column="return_1",
            economic_rationale="precious-metal relative demand",
        ),
        FeatureAvailabilityRecord(
            feature_name="silver_volatility_20",
            source_id="xagusd_mt5",
            value_column="volatility_20",
            economic_rationale="precious-metal risk regime",
        ),
    )


def test_backward_join_uses_only_information_available_at_decision_time() -> None:
    decisions = pd.DataFrame(
        {
            "decision_timestamp": pd.to_datetime(
                ["2024-01-03T00:00:00Z", "2024-01-04T00:00:00Z", "2024-01-05T00:00:00Z"],
                utc=True,
            )
        }
    )
    source = pd.DataFrame(
        {
            "bar_open": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-03T00:00:00Z"], utc=True
            ),
            "available_at": pd.to_datetime(
                ["2024-01-02T00:00:00Z", "2024-01-04T00:00:00Z"], utc=True
            ),
            "return_1": [0.01, -0.02],
            "volatility_20": [0.10, 0.20],
        }
    )

    result = backward_asof_feature_join(
        decisions,
        source,
        policy=_policy(),
        features=_features(),
        namespace="silver__",
    )

    assert result["silver_return_1"].tolist() == [0.01, -0.02, -0.02]
    assert result["silver__staleness_days"].tolist() == [1.0, 0.0, 1.0]
    assert (result["silver__source_available_at"] <= result["decision_timestamp"]).all()
    assert result["silver__feature_coverage"].tolist() == [1.0, 1.0, 1.0]
    assert result["silver__revision_policy"].nunique() == 1
    assert result["silver__vintage_policy"].nunique() == 1
    assert result["silver__volume_semantics"].nunique() == 1


def test_join_never_backfills_an_earlier_decision_from_a_future_release() -> None:
    decisions = pd.DataFrame(
        {
            "decision_timestamp": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-03T00:00:00Z"], utc=True
            )
        }
    )
    source = pd.DataFrame(
        {
            "bar_open": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "available_at": pd.to_datetime(["2024-01-02T00:00:00Z"], utc=True),
            "return_1": [0.01],
            "volatility_20": [0.10],
        }
    )

    result = backward_asof_feature_join(
        decisions, source, policy=_policy(), features=_features(), namespace="silver__"
    )

    assert pd.isna(result.loc[0, "silver_return_1"])
    assert not result.loc[0, "silver__availability_matched"]
    assert result.loc[0, "silver__feature_coverage"] == 0.0
    assert result.loc[1, "silver_return_1"] == pytest.approx(0.01)


@pytest.mark.parametrize("naive_side", ["decision", "source"])
def test_join_rejects_timezone_naive_timestamps(naive_side: str) -> None:
    decisions = pd.DataFrame(
        {"decision_timestamp": pd.to_datetime(["2024-01-03T00:00:00Z"], utc=True)}
    )
    source = pd.DataFrame(
        {
            "bar_open": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "available_at": pd.to_datetime(["2024-01-02T00:00:00Z"], utc=True),
            "return_1": [0.01],
            "volatility_20": [0.10],
        }
    )
    if naive_side == "decision":
        decisions["decision_timestamp"] = pd.to_datetime(["2024-01-03"])
    else:
        source["available_at"] = pd.to_datetime(["2024-01-02"])

    with pytest.raises(ValueError, match="timezone-aware"):
        backward_asof_feature_join(
            decisions, source, policy=_policy(), features=_features(), namespace="silver__"
        )


def test_stale_matches_are_audited_but_feature_values_are_not_used() -> None:
    decisions = pd.DataFrame(
        {"decision_timestamp": pd.to_datetime(["2024-01-10T00:00:00Z"], utc=True)}
    )
    source = pd.DataFrame(
        {
            "bar_open": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "available_at": pd.to_datetime(["2024-01-02T00:00:00Z"], utc=True),
            "return_1": [0.01],
            "volatility_20": [0.10],
        }
    )

    result = backward_asof_feature_join(
        decisions,
        source,
        policy=_policy(),
        features=_features(),
        max_staleness="2D",
        namespace="silver__",
    )

    assert result.loc[0, "silver__availability_matched"]
    assert result.loc[0, "silver__is_stale"]
    assert result.loc[0, "silver__staleness_days"] == 8.0
    assert pd.isna(result.loc[0, "silver_return_1"])
    assert result.loc[0, "silver__feature_coverage"] == 0.0


def test_observation_after_availability_fails_closed() -> None:
    decisions = pd.DataFrame(
        {"decision_timestamp": pd.to_datetime(["2024-01-04T00:00:00Z"], utc=True)}
    )
    source = pd.DataFrame(
        {
            "bar_open": pd.to_datetime(["2024-01-03T00:00:00Z"], utc=True),
            "available_at": pd.to_datetime(["2024-01-02T00:00:00Z"], utc=True),
            "return_1": [0.01],
            "volatility_20": [0.10],
        }
    )

    with pytest.raises(ValueError, match="cannot occur after"):
        backward_asof_feature_join(
            decisions, source, policy=_policy(), features=_features(), namespace="silver__"
        )


def test_manifest_records_include_revision_vintage_and_volume_semantics() -> None:
    policy_record = _policy().to_manifest_record()
    feature_record = _features()[0].to_manifest_record()

    assert policy_record["revision_policy"]
    assert policy_record["vintage_policy"]
    assert policy_record["volume_semantics"] == (
        "broker tick volume; not centralized exchange volume"
    )
    assert feature_record["economic_rationale"] == "precious-metal relative demand"

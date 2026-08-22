"""Point-in-time source and feature availability contracts.

This module is intentionally independent of the modeling pipeline.  It provides a
fail-closed contract for future multi-source feature work without making any external
source admissible by itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceAvailabilityRecord:
    """Auditable availability and revision policy for one source."""

    source_id: str
    observation_timestamp_column: str
    available_at_column: str
    timezone: str
    availability_policy: str
    revision_policy: str
    vintage_policy: str
    volume_semantics: str
    evidence_status: str
    admissible_for_development_selection: bool

    def __post_init__(self) -> None:
        required = {
            "source_id": self.source_id,
            "observation_timestamp_column": self.observation_timestamp_column,
            "available_at_column": self.available_at_column,
            "timezone": self.timezone,
            "availability_policy": self.availability_policy,
            "revision_policy": self.revision_policy,
            "vintage_policy": self.vintage_policy,
            "volume_semantics": self.volume_semantics,
            "evidence_status": self.evidence_status,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"Source availability fields cannot be empty: {missing}")

    def to_manifest_record(self) -> dict[str, Any]:
        """Return a JSON-serializable record for immutable manifests."""

        return asdict(self)


@dataclass(frozen=True)
class FeatureAvailabilityRecord:
    """Static lineage for a feature copied from an availability-controlled source."""

    feature_name: str
    source_id: str
    value_column: str
    economic_rationale: str

    def __post_init__(self) -> None:
        required = {
            "feature_name": self.feature_name,
            "source_id": self.source_id,
            "value_column": self.value_column,
            "economic_rationale": self.economic_rationale,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"Feature availability fields cannot be empty: {missing}")

    def to_manifest_record(self) -> dict[str, str]:
        """Return a JSON-serializable record for immutable manifests."""

        return asdict(self)


def _aware_utc(series: pd.Series, field_name: str) -> pd.Series:
    """Parse timestamps while rejecting missing or timezone-naive values."""

    converted: list[pd.Timestamp] = []
    for value in series.tolist():
        if pd.isna(value):
            raise ValueError(f"{field_name} contains a missing timestamp")
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name} contains an invalid timestamp: {value!r}") from error
        if timestamp.tzinfo is None:
            raise ValueError(f"{field_name} must contain timezone-aware timestamps")
        converted.append(timestamp.tz_convert("UTC"))
    return pd.Series(pd.DatetimeIndex(converted), index=series.index, name=series.name)


def _validate_feature_records(
    features: Sequence[FeatureAvailabilityRecord],
    policy: SourceAvailabilityRecord,
) -> None:
    if not features:
        raise ValueError("At least one feature availability record is required")
    names = [feature.feature_name for feature in features]
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique")
    wrong_source = [
        feature.feature_name for feature in features if feature.source_id != policy.source_id
    ]
    if wrong_source:
        raise ValueError(
            f"Features do not belong to source {policy.source_id!r}: {wrong_source}"
        )


def backward_asof_feature_join(
    decisions: pd.DataFrame,
    source: pd.DataFrame,
    *,
    policy: SourceAvailabilityRecord,
    features: Sequence[FeatureAvailabilityRecord],
    decision_timestamp_column: str = "decision_timestamp",
    max_staleness: str | pd.Timedelta | None = None,
    namespace: str | None = None,
) -> pd.DataFrame:
    """Join the latest source record that was available at each decision timestamp.

    Both inputs must already be chronological.  Timestamps must carry an explicit
    timezone and are normalized to UTC.  The function uses only source rows satisfying
    ``source_available_at <= decision_timestamp``; a later source row is never used to
    fill an earlier decision.  If ``max_staleness`` is supplied, stale feature values are
    nulled while their matched timestamps remain available for audit.
    """

    _validate_feature_records(features, policy)
    required_decision = {decision_timestamp_column}
    required_source = {
        policy.observation_timestamp_column,
        policy.available_at_column,
        *(feature.value_column for feature in features),
    }
    missing_decision = required_decision - set(decisions.columns)
    missing_source = required_source - set(source.columns)
    if missing_decision:
        raise ValueError(f"Decision frame is missing columns: {sorted(missing_decision)}")
    if missing_source:
        raise ValueError(f"Source frame is missing columns: {sorted(missing_source)}")

    decision_time = _aware_utc(decisions[decision_timestamp_column], decision_timestamp_column)
    observation_time = _aware_utc(
        source[policy.observation_timestamp_column], policy.observation_timestamp_column
    )
    available_time = _aware_utc(source[policy.available_at_column], policy.available_at_column)
    if not decision_time.is_monotonic_increasing:
        raise ValueError("Decision timestamps must be chronological")
    if not available_time.is_monotonic_increasing:
        raise ValueError("Source availability timestamps must be chronological")
    if available_time.duplicated().any():
        raise ValueError("Source availability timestamps must be unique")
    if (observation_time > available_time).any():
        raise ValueError("A source observation cannot occur after its availability timestamp")

    maximum_age: pd.Timedelta | None = None
    if max_staleness is not None:
        maximum_age = pd.Timedelta(max_staleness)
        if maximum_age <= pd.Timedelta(0):
            raise ValueError("max_staleness must be positive")

    prefix = namespace if namespace is not None else f"{policy.source_id}__"
    if not prefix:
        raise ValueError("namespace cannot be empty")
    output_columns = [feature.feature_name for feature in features]
    audit_columns = [
        f"{prefix}source_observation_timestamp",
        f"{prefix}source_available_at",
        f"{prefix}staleness_seconds",
        f"{prefix}staleness_days",
        f"{prefix}availability_matched",
        f"{prefix}is_stale",
        f"{prefix}feature_coverage",
        f"{prefix}revision_policy",
        f"{prefix}vintage_policy",
        f"{prefix}volume_semantics",
        f"{prefix}evidence_status",
    ]
    conflicts = (set(output_columns) | set(audit_columns)) & set(decisions.columns)
    if conflicts:
        raise ValueError(f"Availability join would overwrite decision columns: {sorted(conflicts)}")

    left_key = "__availability_decision_time"
    right_key = "__availability_source_time"
    observation_key = "__availability_observation_time"
    left = decisions.copy()
    left[left_key] = decision_time.to_numpy()
    right_payload: dict[str, Any] = {
        right_key: available_time.to_numpy(),
        observation_key: observation_time.to_numpy(),
    }
    internal_value_columns: list[str] = []
    for index, feature in enumerate(features):
        internal_name = f"__availability_value_{index}"
        internal_value_columns.append(internal_name)
        right_payload[internal_name] = source[feature.value_column].to_numpy(copy=True)
    right = pd.DataFrame(right_payload)

    joined = pd.merge_asof(
        left,
        right,
        left_on=left_key,
        right_on=right_key,
        direction="backward",
        allow_exact_matches=True,
    )
    matched = joined[right_key].notna()
    if (joined.loc[matched, right_key] > joined.loc[matched, left_key]).any():
        raise AssertionError("Point-in-time join selected a source row from the future")
    staleness = joined[left_key] - joined[right_key]
    stale = pd.Series(False, index=joined.index)
    if maximum_age is not None:
        stale = matched & (staleness > maximum_age)

    for feature, internal_name in zip(features, internal_value_columns, strict=True):
        joined[feature.feature_name] = joined[internal_name].where(matched & ~stale)
    usable_values = joined[output_columns].notna()
    joined[f"{prefix}source_observation_timestamp"] = joined[observation_key]
    joined[f"{prefix}source_available_at"] = joined[right_key]
    joined[f"{prefix}staleness_seconds"] = staleness.dt.total_seconds()
    joined[f"{prefix}staleness_days"] = staleness.dt.total_seconds() / 86_400.0
    joined[f"{prefix}availability_matched"] = matched
    joined[f"{prefix}is_stale"] = stale
    joined[f"{prefix}feature_coverage"] = usable_values.mean(axis=1).where(matched, 0.0)
    joined[f"{prefix}revision_policy"] = policy.revision_policy
    joined[f"{prefix}vintage_policy"] = policy.vintage_policy
    joined[f"{prefix}volume_semantics"] = policy.volume_semantics
    joined[f"{prefix}evidence_status"] = policy.evidence_status

    joined = joined.drop(
        columns=[left_key, right_key, observation_key, *internal_value_columns]
    )
    joined.index = decisions.index
    # Coverage must be a finite [0, 1] audit field even when the source has no match.
    coverage = joined[f"{prefix}feature_coverage"].to_numpy(dtype=float)
    if not np.isfinite(coverage).all() or ((coverage < 0.0) | (coverage > 1.0)).any():
        raise AssertionError("Feature coverage is outside [0, 1]")
    return joined

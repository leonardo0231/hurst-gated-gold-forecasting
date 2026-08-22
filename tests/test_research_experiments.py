from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hge_gold.calibration import fit_past_only_sigmoid
from hge_gold.research_experiments import (
    DEVELOPMENT_BOUNDARY,
    DEVELOPMENT_ROW_LIMIT,
    add_robust_hurst_features,
    enumerate_preregistered_trials,
    load_development_source,
    select_arm_features,
)
from hge_gold.research_experiments import _model as build_registered_model


def _ohlcv(dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = np.linspace(1_800.0, 1_810.0, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.arange(len(dates), dtype=float) + 100.0,
        }
    )


def test_development_loader_reads_only_declared_prefix(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    frame = _ohlcv(pd.date_range("2023-06-28", periods=6, freq="D"))
    frame.to_csv(source, index=False)

    development = load_development_source(
        source,
        boundary_date="2023-07-01",
        row_limit=3,
        min_rows=3,
    )

    assert len(development) == 3
    assert development["date"].max() < pd.Timestamp("2023-07-01")


def test_development_loader_fails_closed_if_prefix_crosses_boundary(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    _ohlcv(pd.date_range("2023-06-29", periods=4, freq="D")).to_csv(source, index=False)

    with pytest.raises(ValueError, match="development boundary"):
        load_development_source(
            source,
            boundary_date="2023-07-01",
            row_limit=4,
            min_rows=3,
        )


def test_preregistered_trial_space_is_exactly_twelve_and_frozen() -> None:
    trials = enumerate_preregistered_trials()

    assert len(trials) == 12
    assert {(trial.arm, trial.horizon) for trial in trials} == {
        (arm, horizon)
        for arm in ("no_hurst", "current_dfa_hurst", "robust_hurst_regime")
        for horizon in (1, 5, 10, 20)
    }
    assert all(trial.logistic_c == 0.2 for trial in trials)
    assert all(trial.seed == 42 for trial in trials)
    assert DEVELOPMENT_BOUNDARY == "2023-07-03"
    assert DEVELOPMENT_ROW_LIMIT == 3213


def test_hurst_arms_share_base_features_and_exclude_legacy_rs() -> None:
    columns = [
        "return_lag_1",
        "momentum_20",
        "hurst_rs_single_scale_legacy_64",
        "hurst_dfa1_64",
        "hurst_dfa1_128",
        "hurst_dfa1_available",
        "hurst_regime",
        "hurst_regime_available",
        "hurst_robust_median",
        "hurst_robust_dispersion",
        "hurst_robust_regime",
        "hurst_robust_available",
    ]

    no_hurst = select_arm_features(columns, "no_hurst")
    current = select_arm_features(columns, "current_dfa_hurst")
    robust = select_arm_features(columns, "robust_hurst_regime")

    assert no_hurst == ["return_lag_1", "momentum_20"]
    assert set(no_hurst).issubset(current) and set(no_hurst).issubset(robust)
    assert "hurst_rs_single_scale_legacy_64" not in current + robust
    assert "hurst_dfa1_128" in current
    assert "hurst_robust_regime" in robust
    assert "hurst_dfa1_128" not in robust


def test_robust_hurst_regime_thresholds_use_past_values_only() -> None:
    frame = pd.DataFrame(
        {
            "hurst_dfa1_64": np.linspace(0.2, 0.8, 140),
            "hurst_dfa1_128": np.linspace(0.25, 0.75, 140),
        }
    )
    original = add_robust_hurst_features(frame, regime_window=100, min_periods=80)
    changed = frame.copy()
    changed.loc[100:, ["hurst_dfa1_64", "hurst_dfa1_128"]] = 9.0
    revised = add_robust_hurst_features(changed, regime_window=100, min_periods=80)

    pd.testing.assert_frame_equal(original.iloc[:100], revised.iloc[:100])
    assert revised.loc[100, "hurst_robust_high_threshold"] == pytest.approx(
        original.loc[100, "hurst_robust_high_threshold"]
    )


def test_sigmoid_calibrator_rejects_non_past_calibration_window() -> None:
    probabilities = np.linspace(0.1, 0.9, 20)
    y = np.array([0, 1] * 10)

    with pytest.raises(ValueError, match="strictly after"):
        fit_past_only_sigmoid(
            probabilities,
            y,
            calibration_row_ids=np.arange(20),
            prediction_row_ids=np.arange(10, 30),
            seed=42,
        )


def test_registered_transform_is_fitted_only_on_training_rows() -> None:
    training = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    labels = np.array([0, 1, 0, 1])
    validation = pd.DataFrame({"feature": [10_000.0, 20_000.0]})
    trial = enumerate_preregistered_trials()[0]

    fitted = build_registered_model(trial).fit(training, labels)
    fitted.predict_proba(validation)

    scaler = fitted.named_steps["scaler"]
    assert scaler.mean_[0] == pytest.approx(training["feature"].mean())
    assert scaler.mean_[0] != pytest.approx(validation["feature"].mean())


def test_registered_model_is_reproducible_from_frozen_trial_seed() -> None:
    features = pd.DataFrame(
        {
            "a": np.linspace(-2.0, 2.0, 40),
            "b": np.sin(np.arange(40, dtype=float)),
        }
    )
    labels = np.array([0, 1] * 20)
    trial = enumerate_preregistered_trials()[0]

    first = build_registered_model(trial).fit(features, labels).predict_proba(features)
    second = build_registered_model(trial).fit(features, labels).predict_proba(features)

    np.testing.assert_array_equal(first, second)

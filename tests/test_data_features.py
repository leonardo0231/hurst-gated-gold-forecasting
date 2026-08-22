from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hge_gold.config import FeatureConfig
from hge_gold.data import generate_research_sample, normalize_and_validate
from hge_gold.features import (
    _dfa1_scales,
    _hurst_dfa1,
    assert_causal_features,
    build_feature_matrix,
)


def _reference_dfa1(log_prices: np.ndarray) -> float:
    increments = np.diff(log_prices)
    profile = np.cumsum(increments - increments.mean())
    scale_values: list[float] = []
    fluctuation_values: list[float] = []
    for scale in _dfa1_scales(len(increments)):
        n_segments = len(profile) // scale
        remainder = len(profile) - n_segments * scale
        offsets = (0,) if remainder == 0 else (0, remainder)
        residuals: list[np.ndarray] = []
        for offset in offsets:
            for segment_index in range(n_segments):
                start = offset + segment_index * scale
                segment = profile[start : start + scale]
                fitted = np.polyval(np.polyfit(np.arange(scale), segment, 1), np.arange(scale))
                residuals.append(segment - fitted)
        fluctuation = np.sqrt(np.mean(np.concatenate(residuals) ** 2))
        if fluctuation > 1e-15:
            scale_values.append(float(scale))
            fluctuation_values.append(float(fluctuation))
    return float(np.polyfit(np.log(scale_values), np.log(fluctuation_values), 1)[0])


def test_source_validation_rejects_duplicate_dates() -> None:
    frame = generate_research_sample(700)
    frame.loc[10, "date"] = frame.loc[9, "date"]
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        normalize_and_validate(frame, min_rows=700)


def test_features_are_causal_under_future_mutation() -> None:
    source = normalize_and_validate(generate_research_sample(900), min_rows=700)
    original, columns = build_feature_matrix(source, FeatureConfig(regime_window=160))
    mutated_source = source.copy()
    cutoff = 620
    multiplier = np.linspace(1.25, 1.75, len(mutated_source) - cutoff - 1)
    for column in ["open", "high", "low", "close"]:
        mutated_source.loc[cutoff + 1 :, column] *= multiplier
    mutated_source.loc[cutoff + 1 :, "volume"] *= 2.0
    mutated, _ = build_feature_matrix(mutated_source, FeatureConfig(regime_window=160))
    assert_causal_features(original, mutated, columns, inclusive_row=cutoff)


def test_feature_matrix_contains_no_future_columns() -> None:
    source = normalize_and_validate(generate_research_sample(800), min_rows=700)
    matrix, columns = build_feature_matrix(source, FeatureConfig(regime_window=150))
    assert len(columns) >= 45
    assert "forward_log_return" not in matrix.columns
    assert matrix["row_id"].equals(pd.Series(range(len(matrix)), name="row_id"))


def test_dfa1_matches_independent_reference_and_fixed_scale_grid() -> None:
    rng = np.random.default_rng(20260820)
    log_prices = np.r_[0.0, np.cumsum(rng.normal(size=127))]

    assert _dfa1_scales(63) == (4, 8, 16)
    assert _dfa1_scales(127) == (4, 8, 16, 32)
    assert _hurst_dfa1(log_prices) == pytest.approx(_reference_dfa1(log_prices), abs=1e-12)


def test_dfa1_is_invariant_to_price_level_and_positive_amplitude_scale() -> None:
    rng = np.random.default_rng(11)
    log_prices = np.r_[0.0, np.cumsum(rng.normal(size=127))]
    expected = _hurst_dfa1(log_prices)

    assert _hurst_dfa1(log_prices + 7.5) == pytest.approx(expected, abs=1e-12)
    assert _hurst_dfa1(3.0 * log_prices - 2.0) == pytest.approx(expected, abs=1e-12)


def test_dfa1_white_noise_reference_is_centered_near_one_half() -> None:
    rng = np.random.default_rng(42)
    estimates = [_hurst_dfa1(np.r_[0.0, np.cumsum(rng.normal(size=256))]) for _ in range(80)]

    assert np.mean(estimates) == pytest.approx(0.5, abs=0.08)


def test_hurst_columns_are_explicit_and_regime_is_missing_until_ready() -> None:
    source = normalize_and_validate(generate_research_sample(800), min_rows=700)
    config = FeatureConfig(hurst_windows=(64, 128), regime_window=160)
    matrix, columns = build_feature_matrix(source, config)

    assert "hurst_rs_64" not in columns
    assert "hurst_rs_single_scale_legacy_64" in columns
    assert "hurst_dfa1_64" in columns
    assert matrix["hurst_dfa1_128"].first_valid_index() == 127
    assert matrix["hurst_regime"].first_valid_index() == 207
    assert matrix.loc[:206, "hurst_regime"].isna().all()
    assert (matrix.loc[:206, "hurst_regime_available"] == 0.0).all()
    assert (matrix.loc[207:, "hurst_regime_available"] == 1.0).all()

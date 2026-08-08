from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hge_gold.config import FeatureConfig
from hge_gold.data import generate_research_sample, normalize_and_validate
from hge_gold.features import assert_causal_features, build_feature_matrix


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

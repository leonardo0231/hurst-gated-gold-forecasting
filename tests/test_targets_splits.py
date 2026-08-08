from __future__ import annotations

from hge_gold.config import FeatureConfig, SplitConfig, TargetConfig
from hge_gold.data import generate_research_sample, normalize_and_validate
from hge_gold.features import build_feature_matrix
from hge_gold.splits import (
    assert_no_label_overlap,
    build_purged_walk_forward_folds,
    split_development_and_locked_test,
)
from hge_gold.targets import build_horizon_dataset


def _dataset(horizon: int = 5):
    source = normalize_and_validate(generate_research_sample(1200), min_rows=700)
    feature_config = FeatureConfig(regime_window=180)
    features, columns = build_feature_matrix(source, feature_config)
    dataset = build_horizon_dataset(
        source,
        features,
        columns,
        horizon,
        TargetConfig(horizons=(horizon,)),
        feature_config,
    )
    return dataset[dataset["is_modeling_eligible"]].reset_index(drop=True)


def test_target_alignment_matches_horizon() -> None:
    dataset = _dataset(10)
    assert (dataset["label_end_index"] - dataset["row_id"] == 10).all()
    assert dataset["direction_binary"].isin([0.0, 1.0]).all()
    assert (dataset["direction_threshold"] > 0).all()


def test_walk_forward_split_is_purged() -> None:
    dataset = _dataset(5)
    config = SplitConfig(
        locked_test_fraction=0.20,
        n_walk_forward_folds=4,
        min_train_rows=240,
        min_validation_rows=40,
    )
    development, locked, locked_start = split_development_and_locked_test(dataset, config)
    folds = build_purged_walk_forward_folds(development, config)
    assert_no_label_overlap(development, folds)
    assert (development["label_end_index"] < locked_start).all()
    assert (locked["row_id"] >= locked_start).all()


def test_non_actionable_rows_are_not_excluded_from_modeling() -> None:
    source = normalize_and_validate(
        generate_research_sample(1200),
        min_rows=700,
    )

    feature_config = FeatureConfig(
        regime_window=180,
    )

    features, columns = build_feature_matrix(source, feature_config)

    dataset = build_horizon_dataset(
        source,
        features,
        columns,
        5,
        TargetConfig(horizons=(5,)),
        feature_config,
    )

    eligible = dataset[dataset["is_modeling_eligible"]]

    non_actionable = eligible[~eligible["is_actionable"]]

    assert not non_actionable.empty

    assert non_actionable["direction_binary"].isin([0.0, 1.0]).all()


def test_binary_direction_matches_future_return_sign() -> None:
    dataset = _dataset(10)

    expected = (dataset["forward_log_return"] > 0.0).astype(float)

    assert (
        dataset["direction_binary"].reset_index(drop=True).equals(expected.reset_index(drop=True))
    )

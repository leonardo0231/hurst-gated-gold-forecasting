from __future__ import annotations

from hge_gold.v2.config import FeatureConfig, ModelConfig, SplitConfig, TargetConfig
from hge_gold.v2.data import generate_research_sample, normalize_and_validate
from hge_gold.v2.evaluation import classification_metrics
from hge_gold.v2.features import build_feature_matrix
from hge_gold.v2.modeling import train_and_predict
from hge_gold.v2.splits import build_purged_walk_forward_folds, split_development_and_locked_test
from hge_gold.v2.targets import build_horizon_dataset


def test_modeling_pipeline_can_learn_registered_synthetic_signal() -> None:
    source = normalize_and_validate(generate_research_sample(1500, seed=7), min_rows=700)
    feature_config = FeatureConfig(regime_window=180)
    features, columns = build_feature_matrix(source, feature_config)
    dataset = build_horizon_dataset(
        source,
        features,
        columns,
        5,
        TargetConfig(horizons=(5,), threshold_k=0.25),
        feature_config,
    )
    eligible = dataset[dataset["is_modeling_eligible"]].reset_index(drop=True)
    split_config = SplitConfig(
        locked_test_fraction=0.20,
        n_walk_forward_folds=4,
        min_train_rows=260,
        min_validation_rows=45,
    )
    development, locked, _ = split_development_and_locked_test(eligible, split_config)
    folds = build_purged_walk_forward_folds(development, split_config)
    result = train_and_predict(
        development,
        locked,
        columns,
        folds,
        ModelConfig(random_seed=7, fast_mode=True),
    )
    metrics = classification_metrics(
        locked["direction_binary"].to_numpy(dtype=int),
        result.locked_probability_up,
        result.threshold,
    )
    assert result.bundle["selection_uses_locked_test"] is False
    assert metrics["balanced_accuracy"] >= 0.60
    assert metrics["macro_f1"] >= 0.58

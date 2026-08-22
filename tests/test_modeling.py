from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hge_gold.config import FeatureConfig, ModelConfig, SplitConfig, TargetConfig
from hge_gold.data import generate_research_sample, normalize_and_validate
from hge_gold.evaluation import classification_metrics
from hge_gold.features import build_feature_matrix
from hge_gold.modeling import (
    _candidate_stability_key,
    _candidate_stability_summary,
    _purged_meta_train_mask,
    train_and_predict,
)
from hge_gold.splits import (
    WalkForwardFold,
    build_purged_walk_forward_folds,
    split_development_and_locked_test,
)
from hge_gold.targets import build_horizon_dataset


def test_candidate_stability_uses_preselection_folds() -> None:
    fold_metrics = [
        {
            "fold_id": "wf_01",
            "candidate": "stable",
            "balanced_accuracy": 0.55,
            "macro_f1": 0.54,
            "roc_auc": 0.56,
        },
        {
            "fold_id": "wf_02",
            "candidate": "stable",
            "balanced_accuracy": 0.56,
            "macro_f1": 0.55,
            "roc_auc": 0.57,
        },
        {
            "fold_id": "wf_03",
            "candidate": "stable",
            "balanced_accuracy": 0.57,
            "macro_f1": 0.56,
            "roc_auc": 0.58,
        },
        {
            "fold_id": "wf_01",
            "candidate": "volatile",
            "balanced_accuracy": 0.45,
            "macro_f1": 0.44,
            "roc_auc": 0.46,
        },
        {
            "fold_id": "wf_02",
            "candidate": "volatile",
            "balanced_accuracy": 0.56,
            "macro_f1": 0.55,
            "roc_auc": 0.57,
        },
        {
            "fold_id": "wf_03",
            "candidate": "volatile",
            "balanced_accuracy": 0.67,
            "macro_f1": 0.66,
            "roc_auc": 0.68,
        },
        {
            "fold_id": "wf_04",
            "candidate": "volatile",
            "balanced_accuracy": 0.99,
            "macro_f1": 0.99,
            "roc_auc": 0.99,
        },
    ]

    preselection_folds = {"wf_01", "wf_02", "wf_03"}

    stable = _candidate_stability_summary(
        fold_metrics,
        "stable",
        preselection_folds,
    )
    volatile = _candidate_stability_summary(
        fold_metrics,
        "volatile",
        preselection_folds,
    )

    assert stable["stability_median_balanced_accuracy"] == pytest.approx(0.56)
    assert volatile["stability_median_balanced_accuracy"] == pytest.approx(0.56)

    assert stable["stability_balanced_accuracy_std"] < volatile["stability_balanced_accuracy_std"]

    assert _candidate_stability_key(stable) > _candidate_stability_key(volatile)


def test_meta_training_labels_are_purged_before_selection_fold() -> None:
    development = pd.DataFrame({"label_end_index": [8, 10, 11, 12, 15]})
    common_mask = np.ones(5, dtype=bool)
    oof_fold = np.array(["wf_01", "wf_01", "wf_02", "wf_03", "wf_03"], dtype=object)
    selection_fold = WalkForwardFold(
        fold_id="wf_03",
        train_indices=np.array([0, 1]),
        validation_indices=np.array([3, 4]),
        validation_start_row=12,
        validation_end_row=20,
    )

    mask = _purged_meta_train_mask(development, common_mask, oof_fold, selection_fold)

    assert mask.tolist() == [True, True, True, False, False]
    assert development.loc[mask, "label_end_index"].max() < selection_fold.validation_start_row


def test_modeling_can_disable_stacked_meta_classifier() -> None:
    source = normalize_and_validate(generate_research_sample(900, seed=13), min_rows=700)
    feature_config = FeatureConfig(regime_window=160)
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
        min_train_rows=220,
        min_validation_rows=40,
    )
    development, locked, _ = split_development_and_locked_test(eligible, split_config)
    folds = build_purged_walk_forward_folds(development, split_config)

    result = train_and_predict(
        development,
        locked,
        columns,
        folds,
        ModelConfig(random_seed=13, fast_mode=True),
        allow_meta_model=False,
    )

    assert result.selected_strategy == "best_base_model"
    assert result.bundle["meta_model_allowed"] is False
    assert result.bundle["meta_model"] is None
    assert result.bundle["meta_training_rows"] == 0


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

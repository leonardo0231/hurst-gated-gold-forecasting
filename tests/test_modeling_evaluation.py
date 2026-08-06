from __future__ import annotations

import json

import numpy as np
import pandas as pd

from hge_gold.evaluation import moving_block_bootstrap_mean
from hge_gold.modeling import CLASS_ORDER, build_walk_forward_folds


def test_walk_forward_is_chronological_and_locked() -> None:
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=600))
    folds, locked_start = build_walk_forward_folds(dates, 3, 160, 0.15)
    assert all(
        fold.train_end < fold.validation_start <= fold.validation_end < locked_start
        for fold in folds
    )


def test_modeling_locked_predictions_and_selection(full_run) -> None:  # type: ignore[no-untyped-def]
    outputs = full_run["phase_outputs"][4]
    locked = pd.read_parquet(outputs["locked_predictions"])
    selected = json.loads(outputs["selected_map"].read_text())
    assert "y_true" not in locked.columns
    assert selected["selection_uses_locked_test"] is False
    assert len(selected["selected_models"]) == 16
    assert locked["selected_model_map_hash"].eq(selected["selected_model_map_hash"]).all()
    classification = locked[locked["task"].str.endswith("classification")]
    for raw in classification["y_pred_proba_json"]:
        probabilities = json.loads(raw)
        assert set(map(int, probabilities)) == set(CLASS_ORDER)
        assert np.isclose(sum(probabilities.values()), 1.0)


def test_phase5_join_and_costs(full_run) -> None:  # type: ignore[no-untyped-def]
    outputs = full_run["phase_outputs"][5]
    evaluation = pd.read_parquet(outputs["evaluation"])
    ledger = pd.read_parquet(outputs["ledger"])
    comparison = pd.read_csv(outputs["comparison"])
    assert evaluation["y_true"].notna().all()
    assert not evaluation.duplicated(["row_id", "horizon", "task", "selected_model_id"]).any()
    expected = ledger["position_change"] * ledger["round_trip_cost_bps"] / 10_000
    assert np.allclose(ledger["total_cost"], expected)
    assert not comparison["post_hoc_baseline_selection_used"].any()
    robustness = pd.read_csv(outputs["cost_robustness"])
    assert robustness["net_return"].is_monotonic_decreasing


def test_block_bootstrap_reproducible() -> None:
    values = np.random.default_rng(1).normal(0.1, 1, 100)
    first = moving_block_bootstrap_mean(values, 100, 10, 42)
    second = moving_block_bootstrap_mean(values, 100, 10, 42)
    assert first == second

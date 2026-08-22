from __future__ import annotations

from dataclasses import replace
from math import comb

import numpy as np
import pandas as pd

from hge_gold.config import SplitConfig
from hge_gold.splits import (
    assert_no_interval_overlap,
    build_purged_cpcv_splits,
    build_purged_walk_forward_folds,
    split_development_and_locked_test,
)
from hge_gold.statistics import deflated_sharpe_ratio, probability_of_backtest_overfitting


def _events(n_rows: int, horizon: int = 5) -> pd.DataFrame:
    row_id = np.arange(n_rows, dtype=int)
    return pd.DataFrame(
        {
            "row_id": row_id,
            "date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
            "label_end_index": row_id + horizon,
            "direction_binary": row_id % 2,
        }
    )


def test_configured_locked_boundary_is_append_invariant() -> None:
    original = _events(220)
    boundary = str(original.loc[160, "date"].date())
    config = SplitConfig(
        locked_test_start=boundary,
        n_walk_forward_folds=3,
        min_train_rows=60,
        min_validation_rows=20,
    )

    development, locked, locked_start = split_development_and_locked_test(original, config)
    appended = _events(280)
    appended_development, appended_locked, appended_start = split_development_and_locked_test(
        appended,
        config,
    )

    assert locked_start == appended_start == 160
    assert development["row_id"].tolist() == appended_development["row_id"].tolist()
    assert locked["row_id"].iloc[0] == appended_locked["row_id"].iloc[0] == 160


def test_fraction_split_remains_backward_compatible_when_boundary_is_null() -> None:
    frame = _events(200)
    config = SplitConfig(
        locked_test_fraction=0.20,
        locked_test_start=None,
        n_walk_forward_folds=3,
        min_train_rows=60,
        min_validation_rows=20,
    )

    _, _, locked_start = split_development_and_locked_test(frame, config)

    assert locked_start == 160


def test_walk_forward_builds_exact_requested_count_after_purge() -> None:
    development = _events(600, horizon=20)
    config = SplitConfig(
        n_walk_forward_folds=5,
        min_train_rows=100,
        min_validation_rows=50,
        embargo_rows=3,
    )

    folds = build_purged_walk_forward_folds(development, config)

    assert [fold.fold_id for fold in folds] == [f"wf_{number:02d}" for number in range(1, 6)]
    for fold in folds:
        assert fold.n_train == len(fold.train_indices)
        assert fold.n_validation == len(fold.validation_indices)
        assert fold.n_train_raw == fold.n_train + fold.purged_count + fold.embargoed_count
        assert fold.n_train >= config.min_train_rows
        assert fold.train_start_row is not None
        assert fold.train_end_row is not None
        assert fold.train_end_row < fold.validation_start_row
        assert_no_interval_overlap(
            development,
            fold.train_indices,
            fold.validation_indices,
            context=fold.fold_id,
        )


def test_cpcv_purges_event_overlap_and_applies_embargo() -> None:
    development = _events(240, horizon=8)

    splits = build_purged_cpcv_splits(
        development,
        n_groups=8,
        n_test_groups=2,
        embargo_rows=4,
    )

    assert len(splits) == comb(8, 2)
    assert any(split.purged_count > 0 for split in splits)
    assert any(split.embargoed_count > 0 for split in splits)
    for split in splits:
        assert not np.intersect1d(split.train_indices, split.test_indices).size
        assert split.n_train_raw == split.n_train + split.purged_count + split.embargoed_count
        assert_no_interval_overlap(
            development,
            split.train_indices,
            split.test_indices,
            context=split.split_id,
        )


def test_pbo_distinguishes_stable_signal_from_partition_overfit() -> None:
    rows_per_partition = 12
    n_partitions = 8
    n_rows = rows_per_partition * n_partitions
    phase = np.linspace(0.0, 8.0 * np.pi, n_rows)
    stable = np.column_stack(
        [
            0.010 + 0.002 * np.sin(phase),
            0.004 + 0.003 * np.cos(phase),
            -0.002 + 0.002 * np.sin(phase + 0.5),
        ]
    )
    overfit = np.full((n_rows, n_partitions), -0.010, dtype=float)
    overfit += 0.001 * np.sin(phase)[:, np.newaxis]
    for partition in range(n_partitions):
        start = partition * rows_per_partition
        stop = start + rows_per_partition
        overfit[start:stop, partition] += 0.080

    stable_result = probability_of_backtest_overfitting(stable, n_partitions=n_partitions)
    overfit_result = probability_of_backtest_overfitting(overfit, n_partitions=n_partitions)

    assert stable_result.pbo < overfit_result.pbo
    assert stable_result.median_logit > overfit_result.median_logit


def test_deflated_sharpe_probability_decreases_with_declared_trials() -> None:
    rng = np.random.default_rng(19)
    returns = 0.002 + rng.normal(0.0, 0.01, size=500)

    one_trial = deflated_sharpe_ratio(returns, declared_trials=1, periods_per_year=252.0)
    many_trials = deflated_sharpe_ratio(returns, declared_trials=100, periods_per_year=252.0)

    assert many_trials.benchmark_sharpe > one_trial.benchmark_sharpe
    assert many_trials.probability < one_trial.probability


def test_fixed_boundary_can_be_replaced_without_mutating_config() -> None:
    base = SplitConfig(locked_test_start=None)
    frozen = replace(base, locked_test_start="2020-05-01")

    assert base.locked_test_start is None
    assert frozen.locked_test_start == "2020-05-01"

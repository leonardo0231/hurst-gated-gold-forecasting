from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hge_gold.calibration import assess_sigmoid_eligibility, fit_past_only_sigmoid
from hge_gold.research_experiments_v3 import FAMILY_V3, enumerate_preregistered_trials_v3
from hge_gold.research_protocol import sha256_file
from hge_gold.research_validation import (
    assert_exact_paired_alignment,
    purge_calibration_against_prediction_window,
)
from hge_gold.statistics import (
    holm_adjust,
    paired_block_bootstrap_metric_difference,
    paired_block_sign_flip_test,
)


def test_calibration_purge_removes_label_overlap() -> None:
    calibration = pd.DataFrame(
        {
            "row_id": [100, 101, 102],
            "executable_label_end_index": [101, 105, 109],
            "raw_probability": [0.2, 0.4, 0.8],
            "y": [0, 1, 1],
        }
    )

    retained, metadata = purge_calibration_against_prediction_window(
        calibration,
        label_end_column="executable_label_end_index",
        prediction_start_row=106,
    )

    assert retained["row_id"].tolist() == [100, 101]
    assert metadata == {
        "raw_count": 3,
        "purged_count": 1,
        "retained_count": 2,
        "max_raw_label_end": 109,
        "max_retained_label_end": 105,
        "prediction_start_row": 106,
        "overlap_count_after_purge": 0,
    }


@pytest.mark.parametrize("horizon", [1, 5, 10, 20])
def test_calibration_overlap_zero_for_all_registered_horizons(horizon: int) -> None:
    calibration = pd.DataFrame(
        {
            "row_id": np.arange(100, 120),
            "executable_label_end_index": np.arange(100, 120) + horizon + 1,
        }
    )
    retained, metadata = purge_calibration_against_prediction_window(
        calibration,
        label_end_column="executable_label_end_index",
        prediction_start_row=115,
    )

    assert metadata["overlap_count_after_purge"] == 0
    assert (retained["executable_label_end_index"] < 115).all()


def test_sigmoid_rejects_label_overlap_even_when_row_order_is_valid() -> None:
    with pytest.raises(ValueError, match="label intervals overlap"):
        fit_past_only_sigmoid(
            np.asarray([0.2, 0.8]),
            np.asarray([0, 1]),
            calibration_row_ids=np.asarray([100, 101]),
            calibration_label_end_ids=np.asarray([100, 120]),
            prediction_row_ids=np.asarray([110, 111]),
            seed=42,
        )


def test_sigmoid_ineligible_after_single_class_purge() -> None:
    eligibility = assess_sigmoid_eligibility(
        np.asarray([0.2, 0.3, 0.4]),
        np.asarray([1, 1, 1]),
        minimum_samples=2,
    )

    assert not eligibility.eligible
    assert eligibility.reason == "single_class_after_purge"


def test_all_outer_selections_have_zero_secondary_overlap() -> None:
    checks = []
    for trial in enumerate_preregistered_trials_v3():
        for outer_fold in range(1, 6):
            start = 1000 + outer_fold * 100
            calibration = pd.DataFrame(
                {
                    "row_id": np.arange(start, start + 40),
                    "executable_label_end_index": (
                        np.arange(start, start + 40) + trial.horizon + 1
                    ),
                }
            )
            _, metadata = purge_calibration_against_prediction_window(
                calibration,
                label_end_column="executable_label_end_index",
                prediction_start_row=start + 40,
            )
            checks.append(metadata["overlap_count_after_purge"])
    assert len(checks) == 60
    assert max(checks) == 0


def _paired_frame(probability: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": np.arange(len(probability)),
            "executable_direction_binary": np.tile(np.asarray([0, 1]), len(probability) // 2),
            "probability_up": probability,
        }
    )


def test_paired_alignment_dfa1_no_hurst() -> None:
    reference = _paired_frame(np.tile(np.asarray([0.3, 0.7]), 60))
    candidate = reference.assign(probability_up=reference["probability_up"] + 0.01)

    result = assert_exact_paired_alignment(
        reference,
        candidate,
        reference_name="no_hurst",
        candidate_name="current_dfa_hurst",
    )

    assert result["row_count"] == 120
    assert result["row_ids_identical"]
    assert result["y_true_identical"]


def test_paired_alignment_robust_no_hurst_rejects_changed_target() -> None:
    reference = _paired_frame(np.tile(np.asarray([0.3, 0.7]), 60))
    candidate = reference.assign(
        executable_direction_binary=reference["executable_direction_binary"].iloc[::-1].to_numpy()
    )

    with pytest.raises(Exception, match="target alignment"):
        assert_exact_paired_alignment(reference, candidate, candidate_name="robust_hurst_regime")


def test_paired_bootstrap_is_reproducible() -> None:
    truth = np.tile(np.asarray([0, 1]), 120)
    no_hurst = np.where(truth == 1, 0.62, 0.38)
    hurst = np.where(truth == 1, 0.70, 0.30)
    no_hurst[::10] = 0.62

    first = paired_block_bootstrap_metric_difference(
        truth, hurst, no_hurst, block_length=8, n_resamples=100, seed=42
    )
    second = paired_block_bootstrap_metric_difference(
        truth, hurst, no_hurst, block_length=8, n_resamples=100, seed=42
    )

    assert first == second
    assert first.observed_delta > 0.0


def test_paired_block_sign_flip_is_reproducible() -> None:
    truth = np.tile(np.asarray([0, 1]), 120)
    no_hurst = np.where(truth == 1, 0.62, 0.38)
    hurst = np.where(truth == 1, 0.70, 0.30)

    first = paired_block_sign_flip_test(
        truth, hurst, no_hurst, block_length=10, n_permutations=200, seed=7
    )
    second = paired_block_sign_flip_test(
        truth, hurst, no_hurst, block_length=10, n_permutations=200, seed=7
    )

    assert first == second
    assert 0.0 <= first.p_raw <= 1.0


def test_holm_adjustment_is_applied() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.5])

    assert adjusted == pytest.approx((0.04, 0.09, 0.09, 0.5))


def test_v3_family_has_new_identity_and_twelve_trials() -> None:
    trials = enumerate_preregistered_trials_v3()

    assert FAMILY_V3 == "executable_direction_hurst_ablation_v3"
    assert len(trials) == 12
    assert len({trial.experiment_id for trial in trials}) == 12
    assert all("_v3-" in trial.experiment_id for trial in trials)


def test_v2_artifacts_are_not_overwritten() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = root / "artifacts/research/registry/executable_direction_hurst_ablation_v2.jsonl"
    receipt = (
        root
        / "artifacts/research/runs"
        / ("executable_direction_hurst_ablation_v2-20260822T165905Z/run_receipt.json")
    )

    assert sha256_file(registry) == (
        "2340ab37fa44e548d02f03c45718225a7630aa361493bb608065d82d0a460dc0"
    )
    assert sha256_file(receipt) == (
        "bd6bf37c85cd495cf2c3114b043d6733557feecdb765d78d538aa5f2b5b02482"
    )
    assert "executable_direction_hurst_ablation_v3" not in registry.read_text(encoding="utf-8")

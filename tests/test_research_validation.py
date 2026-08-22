from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from hge_gold.research_validation import (
    REQUIRED_ECONOMIC_BENCHMARKS,
    OuterFoldMetrics,
    PromotionEvidence,
    build_nested_purged_walk_forward_folds,
    evaluate_promotion_gate,
    joint_moving_block_bootstrap,
    nested_split_manifest,
    nested_split_manifest_sha256,
    serialize_nested_split_manifest,
)


def _events(n_rows: int, horizon: int) -> pd.DataFrame:
    row_ids = np.arange(n_rows, dtype=int)
    return pd.DataFrame(
        {
            "row_id": row_ids,
            "label_end_index": row_ids + horizon,
            "direction_binary": row_ids % 2,
        }
    )


def test_nested_builder_returns_exact_purged_5_outer_3_inner_folds() -> None:
    development = _events(1_000, horizon=5)
    nested = build_nested_purged_walk_forward_folds(
        development,
        outer_min_train_rows=300,
        inner_min_train_rows=100,
        outer_min_validation_rows=80,
        inner_min_validation_rows=40,
        pre_validation_gap_rows=3,
    )

    assert len(nested) == 5
    assert all(len(item.inner) == 3 for item in nested)
    for item in nested:
        outer = item.outer
        assert outer.train_end_row < outer.validation_start_row
        assert (
            development.iloc[outer.train_indices]["label_end_index"]
            < outer.validation_start_row
        ).all()
        for inner in item.inner:
            assert set(inner.train_indices).issubset(outer.train_indices)
            assert set(inner.validation_indices).issubset(outer.train_indices)
            assert inner.train_end_row < inner.validation_start_row
            assert (
                development.iloc[inner.train_indices]["label_end_index"]
                < inner.validation_start_row
            ).all()


def test_split_manifest_is_exact_canonical_and_hash_stable() -> None:
    development = _events(1_000, horizon=10)
    nested = build_nested_purged_walk_forward_folds(
        development,
        outer_min_train_rows=300,
        inner_min_train_rows=100,
        outer_min_validation_rows=80,
        inner_min_validation_rows=40,
        pre_validation_gap_rows=2,
    )

    manifest = nested_split_manifest(
        development,
        nested,
        horizon=10,
        pre_validation_gap_rows=2,
    )
    serialized = serialize_nested_split_manifest(manifest)

    assert manifest["outer_fold_count"] == 5
    assert manifest["inner_fold_count_per_outer"] == [3, 3, 3, 3, 3]
    assert manifest["purge_interval_convention"] == "closed_[row_id,label_end_index]"
    assert manifest["pre_validation_gap_rows"] == 2
    assert "train_row_ids" in manifest["outer_folds"][0]["outer"]
    assert serialized == serialize_nested_split_manifest(manifest)
    assert nested_split_manifest_sha256(manifest) == nested_split_manifest_sha256(manifest)


def test_joint_block_bootstrap_is_deterministic_and_paired() -> None:
    rng = np.random.default_rng(9)
    n_rows = 240
    truth = np.tile(np.asarray([0, 1], dtype=int), n_rows // 2)
    probability = np.where(truth == 1, 0.70, 0.30) + rng.normal(0.0, 0.04, n_rows)
    probability = np.clip(probability, 0.01, 0.99)
    benchmark = rng.normal(0.0, 0.003, n_rows)
    candidate = benchmark + 0.001

    first = joint_moving_block_bootstrap(
        truth,
        probability,
        candidate,
        benchmark,
        threshold=0.5,
        primary_block_length=8,
        sensitivity_block_lengths=(4, 16),
        iterations=120,
        seed=17,
    )
    second = joint_moving_block_bootstrap(
        truth,
        probability,
        candidate,
        benchmark,
        threshold=0.5,
        primary_block_length=8,
        sensitivity_block_lengths=(4, 16),
        iterations=120,
        seed=17,
    )

    assert first == second
    assert [estimate.block_length for estimate in first.estimates] == [8, 4, 16]
    paired = first.estimate_for(8).intervals["paired_mean_net_return"]
    assert np.isclose(paired.low, 0.001)
    assert np.isclose(paired.high, 0.001)
    assert first.estimate_for(8).intervals["balanced_accuracy"].low > 0.90


def _passing_evidence() -> PromotionEvidence:
    folds = tuple(
        OuterFoldMetrics(
            fold_id=f"outer_{number:02d}",
            balanced_accuracy=value,
            macro_f1=0.59,
            recall_down=0.58,
            recall_up=0.60,
            n_samples=200,
        )
        for number, value in enumerate([0.61, 0.62, 0.60, 0.63, 0.57], start=1)
    )
    return PromotionEvidence(
        outer_folds=folds,
        pooled_balanced_accuracy=0.61,
        pooled_macro_f1=0.59,
        pooled_recall_down=0.58,
        pooled_recall_up=0.60,
        balanced_accuracy_ci_low=0.54,
        balanced_accuracy_ci_high=0.66,
        balanced_accuracy_sensitivity_lows={"block_5": 0.53, "block_20": 0.52},
        candidate_brier=0.230,
        baseline_brier=0.232,
        candidate_ece=0.040,
        baseline_ece=0.045,
        calibration_intercept=0.02,
        calibration_slope=0.95,
        classification_baseline_deltas={"majority": 0.11, "logistic": 0.02},
        paired_net_return_ci_low=0.0001,
        cumulative_net_return_baseline_cost=0.08,
        cumulative_net_return_stress_cost=0.03,
        economic_benchmark_deltas={name: 0.01 for name in REQUIRED_ECONOMIC_BENCHMARKS},
        n_non_overlapping_trades=80,
        trial_return_registry_complete=True,
        pbo=0.15,
        dsr_probability=0.97,
        qa_flags={
            "leakage_free": True,
            "audit_isolated": True,
            "provenance_complete": True,
            "reproducible": True,
            "manifest_verified": True,
        },
    )


def test_promotion_gate_passes_only_complete_outer_development_evidence() -> None:
    decision = evaluate_promotion_gate(_passing_evidence())

    assert decision.status == "PASS"
    assert all(decision.checks.values())
    assert not decision.reasons


def test_promotion_gate_rejects_subchance_fold_and_missing_pbo_dsr() -> None:
    evidence = _passing_evidence()
    bad_folds = list(evidence.outer_folds)
    bad_folds[-1] = replace(bad_folds[-1], balanced_accuracy=0.49)
    failed = replace(
        evidence,
        outer_folds=tuple(bad_folds),
        trial_return_registry_complete=False,
        pbo=None,
        dsr_probability=None,
    )

    decision = evaluate_promotion_gate(failed)

    assert decision.status == "FAIL"
    assert not decision.checks["no_material_fold_below_chance"]
    assert not decision.checks["pbo_evaluable_and_passes"]
    assert not decision.checks["dsr_evaluable_and_passes"]


def test_promotion_gate_threshold_boundaries_are_inclusive_except_chance_ci() -> None:
    evidence = _passing_evidence()
    boundary_folds = tuple(
        replace(fold, balanced_accuracy=value)
        for fold, value in zip(evidence.outer_folds, [0.60, 0.60, 0.60, 0.55, 0.50], strict=True)
    )
    boundary = replace(
        evidence,
        outer_folds=boundary_folds,
        pooled_macro_f1=0.55,
        pooled_recall_down=0.50,
        pooled_recall_up=0.50,
        balanced_accuracy_ci_low=0.50,
    )

    decision = evaluate_promotion_gate(boundary)

    assert decision.checks["median_balanced_accuracy"]
    assert decision.checks["four_of_five_at_055"]
    assert decision.checks["no_material_fold_below_chance"]
    assert decision.checks["pooled_macro_f1"]
    assert decision.checks["pooled_recall_down"]
    assert decision.checks["pooled_recall_up"]
    assert not decision.checks["primary_ci_excludes_chance"]

"""Research-grade nested validation, uncertainty, and promotion utilities.

This module deliberately does not train models.  A model runner consumes the split
indices returned by :func:`build_nested_purged_walk_forward_folds`, performs all
selection on each outer fold's inner folds, and emits exactly one prediction for each
outer-validation observation.  The resulting pooled evidence can then be passed to
:func:`evaluate_promotion_gate`.

Terminology is intentionally precise:

* ``purged_count`` removes training events whose closed label-information interval
  intersects validation.
* ``pre_validation_gap_rows`` is an optional conservative gap immediately before a
  validation block.  It is not called an embargo.  In a strictly forward expanding
  split, no post-validation observation is in training, so a conventional post-test
  embargo is not applicable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, recall_score


@dataclass(frozen=True)
class ResearchFold:
    """One expanding chronological fold with indices into the full development frame."""

    fold_id: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_start_row: int
    train_end_row: int
    validation_start_row: int
    validation_end_row: int
    n_train_raw: int
    n_train: int
    n_validation: int
    purged_count: int
    pre_validation_gap_count: int


@dataclass(frozen=True)
class NestedOuterFold:
    """An outer evaluation fold and its development-only inner selection folds."""

    outer: ResearchFold
    inner: tuple[ResearchFold, ...]


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float


@dataclass(frozen=True)
class BootstrapEstimate:
    block_length: int
    n_resamples: int
    intervals: Mapping[str, ConfidenceInterval]


@dataclass(frozen=True)
class JointBootstrapResult:
    """Joint uncertainty estimates using identical time-block draws for all series."""

    seed: int
    iterations: int
    primary_block_length: int
    estimates: tuple[BootstrapEstimate, ...]

    def estimate_for(self, block_length: int) -> BootstrapEstimate:
        for estimate in self.estimates:
            if estimate.block_length == block_length:
                return estimate
        raise KeyError(f"No bootstrap estimate exists for block length {block_length}")


@dataclass(frozen=True)
class OuterFoldMetrics:
    fold_id: str
    balanced_accuracy: float
    macro_f1: float
    recall_down: float
    recall_up: float
    n_samples: int
    material: bool = True


@dataclass(frozen=True)
class PromotionEvidence:
    """Complete development-only evidence for one frozen candidate.

    Delta mappings are candidate minus benchmark, so non-negative values mean the
    candidate is not inferior.  Economic mappings must use the canonical keys in
    :data:`REQUIRED_ECONOMIC_BENCHMARKS`.
    """

    outer_folds: tuple[OuterFoldMetrics, ...]
    pooled_balanced_accuracy: float
    pooled_macro_f1: float
    pooled_recall_down: float
    pooled_recall_up: float
    balanced_accuracy_ci_low: float
    balanced_accuracy_ci_high: float
    balanced_accuracy_sensitivity_lows: Mapping[str, float]
    candidate_brier: float
    baseline_brier: float
    candidate_ece: float
    baseline_ece: float
    calibration_intercept: float
    calibration_slope: float
    classification_baseline_deltas: Mapping[str, float]
    paired_net_return_ci_low: float
    cumulative_net_return_baseline_cost: float
    cumulative_net_return_stress_cost: float
    economic_benchmark_deltas: Mapping[str, float]
    n_non_overlapping_trades: int
    trial_return_registry_complete: bool
    pbo: float | None
    dsr_probability: float | None
    qa_flags: Mapping[str, bool]


@dataclass(frozen=True)
class PromotionCriteria:
    expected_outer_folds: int = 5
    minimum_median_balanced_accuracy: float = 0.60
    minimum_pooled_macro_f1: float = 0.55
    minimum_pooled_class_recall: float = 0.50
    minimum_folds_at_055: int = 4
    minimum_fold_balanced_accuracy: float = 0.50
    chance_level: float = 0.50
    maximum_brier_increase: float = 0.005
    maximum_ece_increase: float = 0.01
    minimum_non_overlapping_trades: int = 30
    maximum_pbo: float = 0.20
    minimum_dsr_probability: float = 0.95


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
        }


REQUIRED_QA_FLAGS = frozenset(
    {
        "leakage_free",
        "audit_isolated",
        "provenance_complete",
        "reproducible",
        "manifest_verified",
    }
)

REQUIRED_ECONOMIC_BENCHMARKS = frozenset(
    {"cash", "always_long", "always_short", "momentum", "trend"}
)


def _ordered_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "label_end_index"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Event frame is missing required columns: {sorted(missing)}")
    ordered = frame.sort_values("row_id").reset_index(drop=True)
    row_ids = ordered["row_id"].to_numpy(dtype=int)
    label_ends = ordered["label_end_index"].to_numpy(dtype=int)
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_id values must be unique")
    if np.any(np.diff(row_ids) <= 0):
        raise ValueError("row_id values must be strictly increasing after ordering")
    if np.any(label_ends < row_ids):
        raise ValueError("Every label_end_index must be at or after its row_id")
    return ordered


def _closed_interval_overlap(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> bool:
    train = frame.iloc[np.asarray(train_indices, dtype=int)][
        ["row_id", "label_end_index"]
    ].to_numpy(dtype=int)
    validation = frame.iloc[np.asarray(validation_indices, dtype=int)][
        ["row_id", "label_end_index"]
    ].to_numpy(dtype=int)
    if train.size == 0 or validation.size == 0:
        return False
    return bool(
        np.any(
            (train[:, 0, np.newaxis] <= validation[:, 1])
            & (train[:, 1, np.newaxis] >= validation[:, 0])
        )
    )


def _training_positions(
    ordered: pd.DataFrame,
    validation_start_position: int,
    pre_validation_gap_rows: int,
) -> tuple[np.ndarray, int, int, int]:
    validation_start_row = int(ordered.iloc[validation_start_position]["row_id"])
    row_ids = ordered["row_id"].to_numpy(dtype=int)
    label_ends = ordered["label_end_index"].to_numpy(dtype=int)
    raw = row_ids < validation_start_row
    overlapping = label_ends >= validation_start_row
    gap = np.zeros(len(ordered), dtype=bool)
    if pre_validation_gap_rows:
        gap_start = max(0, validation_start_position - pre_validation_gap_rows)
        gap[gap_start:validation_start_position] = True
    train = raw & ~overlapping & ~gap
    purged_count = int(np.count_nonzero(raw & overlapping))
    gap_count = int(np.count_nonzero(raw & ~overlapping & gap))
    return np.flatnonzero(train), int(np.count_nonzero(raw)), purged_count, gap_count


def _build_expanding_folds(
    ordered: pd.DataFrame,
    *,
    n_folds: int,
    min_train_rows: int,
    min_validation_rows: int,
    pre_validation_gap_rows: int,
    prefix: str,
) -> tuple[ResearchFold, ...]:
    if n_folds < 1:
        raise ValueError("n_folds must be positive")
    if min_train_rows < 2 or min_validation_rows < 1:
        raise ValueError("Minimum train/validation sizes are invalid")
    if pre_validation_gap_rows < 0:
        raise ValueError("pre_validation_gap_rows cannot be negative")
    latest_start = len(ordered) - n_folds * min_validation_rows
    if latest_start <= 0:
        raise ValueError("Not enough observations for the requested folds")

    first_validation_start: int | None = None
    for position in range(min_train_rows, latest_start + 1):
        train, _, _, _ = _training_positions(
            ordered, position, pre_validation_gap_rows
        )
        if len(train) >= min_train_rows:
            first_validation_start = position
            break
    if first_validation_start is None:
        raise ValueError("No boundary retains the configured purged training minimum")

    validation_blocks = np.array_split(
        np.arange(first_validation_start, len(ordered), dtype=int), n_folds
    )
    if any(len(block) < min_validation_rows for block in validation_blocks):
        raise ValueError("Unable to construct exact minimum-size validation folds")

    folds: list[ResearchFold] = []
    for number, validation in enumerate(validation_blocks, start=1):
        validation_start_position = int(validation[0])
        train, n_train_raw, purged_count, gap_count = _training_positions(
            ordered, validation_start_position, pre_validation_gap_rows
        )
        if len(train) < min_train_rows:
            raise ValueError(f"{prefix}_{number:02d} has too few purged training rows")
        if _closed_interval_overlap(ordered, train, validation):
            raise AssertionError(f"Closed event intervals overlap in {prefix}_{number:02d}")
        folds.append(
            ResearchFold(
                fold_id=f"{prefix}_{number:02d}",
                train_indices=train,
                validation_indices=np.asarray(validation, dtype=int),
                train_start_row=int(ordered.iloc[int(train[0])]["row_id"]),
                train_end_row=int(ordered.iloc[int(train[-1])]["row_id"]),
                validation_start_row=int(
                    ordered.iloc[validation_start_position]["row_id"]
                ),
                validation_end_row=int(ordered.iloc[int(validation[-1])]["row_id"]),
                n_train_raw=n_train_raw,
                n_train=len(train),
                n_validation=len(validation),
                purged_count=purged_count,
                pre_validation_gap_count=gap_count,
            )
        )
    return tuple(folds)


def build_nested_purged_walk_forward_folds(
    development: pd.DataFrame,
    *,
    outer_min_train_rows: int,
    inner_min_train_rows: int,
    outer_min_validation_rows: int,
    inner_min_validation_rows: int,
    pre_validation_gap_rows: int = 0,
    outer_folds: int = 5,
    inner_folds: int = 3,
) -> tuple[NestedOuterFold, ...]:
    """Build exact expanding 5-outer/3-inner purged chronological folds.

    The defaults encode the research protocol.  Non-default fold counts are accepted only
    to make small synthetic contract tests possible; production callers must retain 5/3
    and the promotion gate independently requires five outer folds.

    Every returned index addresses the sorted full ``development`` frame.  A runner must:

    1. perform specification and threshold selection only on ``nested.inner``;
    2. refit that selected specification on ``nested.outer.train_indices``; and
    3. predict ``nested.outer.validation_indices`` exactly once.
    """

    ordered = _ordered_event_frame(development)
    outer = _build_expanding_folds(
        ordered,
        n_folds=outer_folds,
        min_train_rows=outer_min_train_rows,
        min_validation_rows=outer_min_validation_rows,
        pre_validation_gap_rows=pre_validation_gap_rows,
        prefix="outer",
    )
    nested: list[NestedOuterFold] = []
    for outer_fold in outer:
        outer_train_global = np.asarray(outer_fold.train_indices, dtype=int)
        inner_frame = ordered.iloc[outer_train_global].reset_index(drop=True)
        local_inner = _build_expanding_folds(
            inner_frame,
            n_folds=inner_folds,
            min_train_rows=inner_min_train_rows,
            min_validation_rows=inner_min_validation_rows,
            pre_validation_gap_rows=pre_validation_gap_rows,
            prefix=f"{outer_fold.fold_id}_inner",
        )
        mapped_inner: list[ResearchFold] = []
        for fold in local_inner:
            train_global = outer_train_global[fold.train_indices]
            validation_global = outer_train_global[fold.validation_indices]
            if not np.all(np.isin(train_global, outer_train_global)) or not np.all(
                np.isin(validation_global, outer_train_global)
            ):
                raise AssertionError("Inner split escaped its outer training partition")
            if _closed_interval_overlap(ordered, train_global, validation_global):
                raise AssertionError(f"Closed event intervals overlap in {fold.fold_id}")
            mapped_inner.append(
                ResearchFold(
                    fold_id=fold.fold_id,
                    train_indices=train_global,
                    validation_indices=validation_global,
                    train_start_row=fold.train_start_row,
                    train_end_row=fold.train_end_row,
                    validation_start_row=fold.validation_start_row,
                    validation_end_row=fold.validation_end_row,
                    n_train_raw=fold.n_train_raw,
                    n_train=fold.n_train,
                    n_validation=fold.n_validation,
                    purged_count=fold.purged_count,
                    pre_validation_gap_count=fold.pre_validation_gap_count,
                )
            )
        nested.append(NestedOuterFold(outer=outer_fold, inner=tuple(mapped_inner)))
    return tuple(nested)


def _fold_manifest(frame: pd.DataFrame, fold: ResearchFold) -> dict[str, Any]:
    train = frame.iloc[fold.train_indices]
    validation = frame.iloc[fold.validation_indices]
    return {
        "fold_id": fold.fold_id,
        "train_row_ids": train["row_id"].astype(int).tolist(),
        "train_label_end_indices": train["label_end_index"].astype(int).tolist(),
        "validation_row_ids": validation["row_id"].astype(int).tolist(),
        "validation_label_end_indices": validation["label_end_index"].astype(int).tolist(),
        "train_start_row": fold.train_start_row,
        "train_end_row": fold.train_end_row,
        "validation_start_row": fold.validation_start_row,
        "validation_end_row": fold.validation_end_row,
        "n_train_raw": fold.n_train_raw,
        "n_train": fold.n_train,
        "n_validation": fold.n_validation,
        "purged_count": fold.purged_count,
        "pre_validation_gap_count": fold.pre_validation_gap_count,
    }


def nested_split_manifest(
    development: pd.DataFrame,
    folds: Sequence[NestedOuterFold],
    *,
    horizon: int,
    pre_validation_gap_rows: int,
) -> dict[str, Any]:
    """Return a JSON-compatible exact split manifest, including every event endpoint."""

    ordered = _ordered_event_frame(development)
    payload: dict[str, Any] = {
        "protocol": "nested_purged_expanding_walk_forward_v1",
        "horizon": int(horizon),
        "outer_fold_count": len(folds),
        "inner_fold_count_per_outer": [len(item.inner) for item in folds],
        "purge_interval_convention": "closed_[row_id,label_end_index]",
        "pre_validation_gap_rows": int(pre_validation_gap_rows),
        "post_validation_embargo_rows": 0,
        "post_validation_embargo_rationale": (
            "not applicable to strictly forward expanding folds because future rows never train"
        ),
        "outer_folds": [],
    }
    outer_payload: list[dict[str, Any]] = []
    for item in folds:
        outer_payload.append(
            {
                "outer": _fold_manifest(ordered, item.outer),
                "inner": [_fold_manifest(ordered, fold) for fold in item.inner],
            }
        )
    payload["outer_folds"] = outer_payload
    return payload


def serialize_nested_split_manifest(manifest: Mapping[str, Any]) -> str:
    """Serialize a split manifest canonically for hashing and immutable receipts."""

    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def nested_split_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    encoded = serialize_nested_split_manifest(manifest).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_nested_split_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a canonical manifest; callers enforce artifact-directory immutability."""

    path.write_text(serialize_nested_split_manifest(manifest) + "\n", encoding="utf-8")


def _ece(y_true: np.ndarray, probability_up: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(probability_up, edges[1:-1]), n_bins - 1)
    result = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(y_true[mask].mean()) - float(probability_up[mask].mean())
            )
    return float(result)


def _bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> NDArray[np.int_]:
    starts = rng.integers(0, n - block_length + 1, size=ceil(n / block_length))
    blocks = [np.arange(start, start + block_length, dtype=int) for start in starts]
    return np.asarray(np.concatenate(blocks)[:n], dtype=int)


def _percentile_interval(values: Sequence[float], confidence: float) -> ConfidenceInterval:
    alpha = 1.0 - confidence
    low, high = np.quantile(np.asarray(values, dtype=float), [alpha / 2.0, 1.0 - alpha / 2.0])
    return ConfidenceInterval(low=float(low), high=float(high))


def joint_moving_block_bootstrap(
    y_true: np.ndarray,
    probability_up: np.ndarray,
    candidate_net_returns: np.ndarray,
    benchmark_net_returns: np.ndarray,
    *,
    threshold: float,
    primary_block_length: int,
    sensitivity_block_lengths: Sequence[int] = (),
    iterations: int = 5_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> JointBootstrapResult:
    """Joint moving-block CIs for classification, calibration, and paired returns.

    The same sampled indices are applied to labels, probabilities, candidate returns,
    and benchmark returns, preserving their time alignment.  Sensitivity lengths are
    evaluated with independent deterministic RNG streams derived from ``seed``.
    """

    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability_up, dtype=float)
    candidate = np.asarray(candidate_net_returns, dtype=float)
    benchmark = np.asarray(benchmark_net_returns, dtype=float)
    if not (truth.ndim == probability.ndim == candidate.ndim == benchmark.ndim == 1):
        raise ValueError("All bootstrap inputs must be one-dimensional")
    if len({len(truth), len(probability), len(candidate), len(benchmark)}) != 1:
        raise ValueError("All bootstrap inputs must have identical lengths")
    if len(truth) < 20 or np.unique(truth).size != 2:
        raise ValueError("Bootstrap requires at least 20 observations and both classes")
    if not np.isfinite(np.column_stack((probability, candidate, benchmark))).all():
        raise ValueError("Bootstrap inputs must be finite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1]")
    if not 0.0 < confidence < 1.0 or iterations < 1:
        raise ValueError("Invalid confidence or iteration count")

    lengths: list[int] = []
    for value in (primary_block_length, *sensitivity_block_lengths):
        block = int(value)
        if block < 1 or block > len(truth):
            raise ValueError("Every block length must be between 1 and the sample size")
        if block not in lengths:
            lengths.append(block)

    estimates: list[BootstrapEstimate] = []
    for block in lengths:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), block]))
        samples: dict[str, list[float]] = {
            "balanced_accuracy": [],
            "macro_f1": [],
            "recall_down": [],
            "recall_up": [],
            "brier_score": [],
            "expected_calibration_error": [],
            "candidate_mean_net_return": [],
            "paired_mean_net_return": [],
        }
        attempts = 0
        while len(samples["balanced_accuracy"]) < iterations:
            attempts += 1
            if attempts > iterations * 100:
                raise RuntimeError("Unable to draw enough two-class bootstrap samples")
            indices = _bootstrap_indices(len(truth), block, rng)
            sampled_truth = truth[indices]
            if np.unique(sampled_truth).size != 2:
                continue
            sampled_probability = probability[indices]
            prediction = (sampled_probability >= threshold).astype(int)
            samples["balanced_accuracy"].append(
                float(balanced_accuracy_score(sampled_truth, prediction))
            )
            samples["macro_f1"].append(
                float(f1_score(sampled_truth, prediction, average="macro", zero_division=0))
            )
            samples["recall_down"].append(
                float(recall_score(sampled_truth, prediction, pos_label=0, zero_division=0))
            )
            samples["recall_up"].append(
                float(recall_score(sampled_truth, prediction, pos_label=1, zero_division=0))
            )
            samples["brier_score"].append(
                float(brier_score_loss(sampled_truth, sampled_probability))
            )
            samples["expected_calibration_error"].append(
                _ece(sampled_truth, sampled_probability)
            )
            candidate_sample = candidate[indices]
            benchmark_sample = benchmark[indices]
            samples["candidate_mean_net_return"].append(float(candidate_sample.mean()))
            samples["paired_mean_net_return"].append(
                float((candidate_sample - benchmark_sample).mean())
            )
        estimates.append(
            BootstrapEstimate(
                block_length=block,
                n_resamples=iterations,
                intervals={
                    name: _percentile_interval(values, confidence)
                    for name, values in samples.items()
                },
            )
        )
    return JointBootstrapResult(
        seed=int(seed),
        iterations=iterations,
        primary_block_length=int(primary_block_length),
        estimates=tuple(estimates),
    )


def _finite(values: Sequence[float] | NDArray[np.float64]) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def evaluate_promotion_gate(
    evidence: PromotionEvidence,
    criteria: PromotionCriteria | None = None,
) -> PromotionDecision:
    """Apply the preregistered all-mandatory development promotion gate.

    Missing PBO/DSR inputs are deliberately a failure (``not_evaluable``), never a pass.
    A post-hoc regime explanation does not override a sub-chance material outer fold.
    """

    criteria = criteria or PromotionCriteria()
    folds = evidence.outer_folds
    fold_ba = np.asarray([fold.balanced_accuracy for fold in folds], dtype=float)
    fold_values = [
        value
        for fold in folds
        for value in (
            fold.balanced_accuracy,
            fold.macro_f1,
            fold.recall_down,
            fold.recall_up,
        )
    ]
    material = [fold for fold in folds if fold.material]
    required_economic_present = REQUIRED_ECONOMIC_BENCHMARKS.issubset(
        evidence.economic_benchmark_deltas
    )
    required_qa_present = REQUIRED_QA_FLAGS.issubset(evidence.qa_flags)
    sensitivity_lows = list(evidence.balanced_accuracy_sensitivity_lows.values())

    checks: dict[str, bool] = {
        "exact_outer_fold_count": len(folds) == criteria.expected_outer_folds,
        "unique_outer_fold_ids": len({fold.fold_id for fold in folds}) == len(folds),
        "outer_metrics_finite": _finite(fold_values) and _finite(fold_ba),
        "outer_samples_positive": all(fold.n_samples > 0 for fold in folds),
        "all_outer_folds_material": len(material) == criteria.expected_outer_folds,
        "median_balanced_accuracy": bool(
            len(fold_ba)
            and np.median(fold_ba) >= criteria.minimum_median_balanced_accuracy
        ),
        "four_of_five_at_055": bool(
            np.count_nonzero(fold_ba >= 0.55) >= criteria.minimum_folds_at_055
        ),
        "no_material_fold_below_chance": bool(
            material
            and all(
                fold.balanced_accuracy >= criteria.minimum_fold_balanced_accuracy
                for fold in material
            )
        ),
        "pooled_metrics_finite": _finite(
            [
                evidence.pooled_balanced_accuracy,
                evidence.pooled_macro_f1,
                evidence.pooled_recall_down,
                evidence.pooled_recall_up,
            ]
        ),
        "pooled_macro_f1": (
            evidence.pooled_macro_f1 >= criteria.minimum_pooled_macro_f1
        ),
        "pooled_recall_down": (
            evidence.pooled_recall_down >= criteria.minimum_pooled_class_recall
        ),
        "pooled_recall_up": (
            evidence.pooled_recall_up >= criteria.minimum_pooled_class_recall
        ),
        "primary_ci_excludes_chance": (
            evidence.balanced_accuracy_ci_low > criteria.chance_level
        ),
        "ci_sensitivity_excludes_chance": bool(
            sensitivity_lows
            and all(value > criteria.chance_level for value in sensitivity_lows)
        ),
        "calibration_reported": _finite(
            [
                evidence.candidate_brier,
                evidence.baseline_brier,
                evidence.candidate_ece,
                evidence.baseline_ece,
                evidence.calibration_intercept,
                evidence.calibration_slope,
            ]
        ),
        "brier_noninferiority": (
            evidence.candidate_brier - evidence.baseline_brier
            <= criteria.maximum_brier_increase
        ),
        "ece_noninferiority": (
            evidence.candidate_ece - evidence.baseline_ece
            <= criteria.maximum_ece_increase
        ),
        "classification_baselines_present": bool(
            evidence.classification_baseline_deltas
        ),
        "classification_baseline_noninferiority": bool(
            evidence.classification_baseline_deltas
            and _finite(list(evidence.classification_baseline_deltas.values()))
            and all(value >= 0.0 for value in evidence.classification_baseline_deltas.values())
        ),
        "paired_return_ci_nonnegative": evidence.paired_net_return_ci_low >= 0.0,
        "baseline_cost_net_positive": (
            evidence.cumulative_net_return_baseline_cost > 0.0
        ),
        "stress_cost_net_positive": evidence.cumulative_net_return_stress_cost > 0.0,
        "economic_benchmarks_present": required_economic_present,
        "economic_benchmark_noninferiority": bool(
            required_economic_present
            and _finite(list(evidence.economic_benchmark_deltas.values()))
            and all(
                evidence.economic_benchmark_deltas[name] >= 0.0
                for name in REQUIRED_ECONOMIC_BENCHMARKS
            )
        ),
        "minimum_trade_count": (
            evidence.n_non_overlapping_trades >= criteria.minimum_non_overlapping_trades
        ),
        "trial_return_registry_complete": evidence.trial_return_registry_complete,
        "pbo_evaluable_and_passes": bool(
            evidence.trial_return_registry_complete
            and evidence.pbo is not None
            and np.isfinite(evidence.pbo)
            and evidence.pbo <= criteria.maximum_pbo
        ),
        "dsr_evaluable_and_passes": bool(
            evidence.trial_return_registry_complete
            and evidence.dsr_probability is not None
            and np.isfinite(evidence.dsr_probability)
            and evidence.dsr_probability >= criteria.minimum_dsr_probability
        ),
        "required_qa_flags_present": required_qa_present,
        "all_qa_flags_pass": bool(
            required_qa_present
            and all(evidence.qa_flags[name] for name in REQUIRED_QA_FLAGS)
            and all(evidence.qa_flags.values())
        ),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return PromotionDecision(
        status="PASS" if not reasons else "FAIL",
        checks=checks,
        reasons=reasons,
    )

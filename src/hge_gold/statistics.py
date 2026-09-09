from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from math import ceil, e, log, sqrt
from statistics import NormalDist

import numpy as np
from sklearn.metrics import balanced_accuracy_score


@dataclass(frozen=True)
class BacktestOverfittingDiagnostic:
    """CSCV probability-of-backtest-overfitting summary.

    ``returns`` supplied to :func:`probability_of_backtest_overfitting` must contain only
    development-period, net strategy returns.  Each column is a fully specified trial and
    rows are in chronological order.
    """

    pbo: float
    n_combinations: int
    median_logit: float
    oos_rank_percentiles: tuple[float, ...]
    selected_strategy_indices: tuple[int, ...]


@dataclass(frozen=True)
class SharpeDiagnostic:
    """Probabilistic/deflated Sharpe result under the stated IID-moment approximation."""

    probability: float
    observed_sharpe: float
    benchmark_sharpe: float
    n_observations: int
    declared_trials: int
    skewness: float
    kurtosis: float
    periods_per_year: float


@dataclass(frozen=True)
class PairedBlockBootstrapResult:
    """Moving-block uncertainty for a paired metric difference."""

    metric: str
    block_length: int
    n_resamples: int
    confidence: float
    seed: int
    observed_a: float
    observed_b: float
    observed_delta: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PairedBlockSignFlipResult:
    """Block sign-flip inference for a paired metric contrast."""

    metric: str
    block_length: int
    block_count: int
    n_permutations: int
    seed: int
    observed_delta: float
    p_raw: float


def _validate_paired_probabilities(
    y_true: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    *,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if metric != "balanced_accuracy":
        raise ValueError("Only the preregistered balanced_accuracy metric is supported")
    truth = np.asarray(y_true, dtype=int)
    first = np.asarray(probability_a, dtype=float)
    second = np.asarray(probability_b, dtype=float)
    if truth.ndim != 1 or first.ndim != 1 or second.ndim != 1:
        raise ValueError("Paired inference inputs must be one-dimensional")
    if len({len(truth), len(first), len(second)}) != 1 or len(truth) < 4:
        raise ValueError("Paired inference inputs must have equal length and at least four rows")
    if np.unique(truth).size != 2:
        raise ValueError("Balanced-accuracy inference requires both target classes")
    if not np.isfinite(np.column_stack((first, second))).all():
        raise ValueError("Paired probabilities must be finite")
    if np.any((first < 0.0) | (first > 1.0)) or np.any((second < 0.0) | (second > 1.0)):
        raise ValueError("Paired probabilities must lie in [0, 1]")
    return truth, first, second


def _moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length < 1 or block_length > n:
        raise ValueError("block_length must be between one and the sample size")
    starts = rng.integers(0, n - block_length + 1, size=ceil(n / block_length))
    indices = [index for start in starts for index in range(int(start), int(start) + block_length)]
    return np.asarray(indices[:n], dtype=int)


def _paired_metric(truth: np.ndarray, probability: np.ndarray, *, metric: str) -> float:
    if metric != "balanced_accuracy":
        raise ValueError("Only the preregistered balanced_accuracy metric is supported")
    return float(balanced_accuracy_score(truth, probability >= 0.5))


def paired_block_bootstrap_metric_difference(
    y_true: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    block_length: int,
    n_resamples: int,
    seed: int,
    metric: str = "balanced_accuracy",
    confidence: float = 0.95,
) -> PairedBlockBootstrapResult:
    """Compute a paired moving-block bootstrap CI for ``A - B``.

    One moving-block index draw is applied to both arms and the common target vector. Draws
    containing only one target class are rejected because balanced accuracy is undefined for
    the missing class. The retry limit is deterministic and fail-closed.
    """

    truth, first, second = _validate_paired_probabilities(
        y_true, probability_a, probability_b, metric=metric
    )
    if n_resamples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("n_resamples must be positive and confidence must lie in (0, 1)")
    n = len(truth)
    if block_length < 1 or block_length > n:
        raise ValueError("block_length must be between one and the sample size")
    observed_a = _paired_metric(truth, first, metric=metric)
    observed_b = _paired_metric(truth, second, metric=metric)
    observed_delta = observed_a - observed_b
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(block_length), 701]))
    deltas: list[float] = []
    attempts = 0
    while len(deltas) < n_resamples:
        attempts += 1
        if attempts > n_resamples * 100:
            raise RuntimeError("Unable to draw enough two-class paired bootstrap samples")
        indices = _moving_block_indices(n, block_length, rng)
        sampled_truth = truth[indices]
        if np.unique(sampled_truth).size != 2:
            continue
        deltas.append(
            _paired_metric(sampled_truth, first[indices], metric=metric)
            - _paired_metric(sampled_truth, second[indices], metric=metric)
        )
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(np.asarray(deltas), [alpha / 2.0, 1.0 - alpha / 2.0])
    return PairedBlockBootstrapResult(
        metric=metric,
        block_length=int(block_length),
        n_resamples=int(n_resamples),
        confidence=float(confidence),
        seed=int(seed),
        observed_a=observed_a,
        observed_b=observed_b,
        observed_delta=observed_delta,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )


def paired_block_sign_flip_test(
    y_true: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    block_length: int,
    n_permutations: int,
    seed: int,
    metric: str = "balanced_accuracy",
) -> PairedBlockSignFlipResult:
    """Run a two-sided paired block sign-flip test for ``A - B``.

    Blocks are consecutive, non-overlapping chronological groups. Each block receives one
    random sign, preserving within-block dependence. Contributions are class-normalized so
    their sum equals the global balanced-accuracy contrast exactly.
    """

    truth, first, second = _validate_paired_probabilities(
        y_true, probability_a, probability_b, metric=metric
    )
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    n = len(truth)
    if block_length < 1 or block_length > n:
        raise ValueError("block_length must be between one and the sample size")
    if len(np.unique(truth)) != 2:
        raise ValueError("Sign-flip inference requires both target classes")

    prediction_a = first >= 0.5
    prediction_b = second >= 0.5
    class_totals = np.asarray([(truth == label).sum() for label in (0, 1)], dtype=float)
    blocks = [
        np.arange(start, min(start + block_length, n), dtype=int)
        for start in range(0, n, block_length)
    ]
    contributions: list[float] = []
    for block in blocks:
        contribution = 0.0
        for class_index, label in enumerate((0, 1)):
            mask = truth[block] == label
            # For class 0 the balanced-accuracy contribution is the true-negative
            # rate, so a prediction of 0 is correct.  Class 1 uses the true-positive
            # rate, where a prediction of 1 is correct.
            correct_difference = (
                prediction_b[block][mask].astype(int) - prediction_a[block][mask].astype(int)
                if label == 0
                else prediction_a[block][mask].astype(int)
                - prediction_b[block][mask].astype(int)
            )
            contribution += (
                0.5
                * float(
                    correct_difference.sum()
                )
                / class_totals[class_index]
            )
        contributions.append(contribution)
    contribution_array = np.asarray(contributions, dtype=float)
    observed_delta = _paired_metric(truth, first, metric=metric) - _paired_metric(
        truth, second, metric=metric
    )
    if not np.isclose(contribution_array.sum(), observed_delta, atol=1e-12, rtol=0.0):
        raise AssertionError("Paired block contributions do not reproduce observed metric delta")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(block_length), 907]))
    exceedances = 0
    observed_abs = abs(observed_delta)
    for _ in range(n_permutations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(contribution_array))
        permuted = abs(float(np.dot(signs, contribution_array)))
        if permuted >= observed_abs - 1e-15:
            exceedances += 1
    p_raw = (1.0 + exceedances) / (n_permutations + 1.0)
    return PairedBlockSignFlipResult(
        metric=metric,
        block_length=int(block_length),
        block_count=len(blocks),
        n_permutations=int(n_permutations),
        seed=int(seed),
        observed_delta=float(observed_delta),
        p_raw=float(p_raw),
    )


def holm_adjust(p_values: Sequence[float]) -> tuple[float, ...]:
    """Return Holm step-down adjusted p-values in the original order."""

    values = np.asarray(tuple(float(value) for value in p_values), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("p_values must be a non-empty finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values))))
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return tuple(float(value) for value in adjusted)


def _validate_returns(returns: np.ndarray, *, minimum_columns: int = 1) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < minimum_columns:
        raise ValueError("Returns must be a finite observations-by-strategies matrix")
    if not np.isfinite(values).all():
        raise ValueError("Returns must contain only finite values")
    return values


def _strategy_sharpes(returns: np.ndarray) -> np.ndarray:
    means = np.mean(returns, axis=0)
    standard_deviations = np.std(returns, axis=0, ddof=1)
    sharpes = np.zeros(returns.shape[1], dtype=float)
    nonzero = standard_deviations > 0.0
    sharpes[nonzero] = means[nonzero] / standard_deviations[nonzero]
    sharpes[~nonzero & (means > 0.0)] = np.inf
    sharpes[~nonzero & (means < 0.0)] = -np.inf
    return sharpes


def probability_of_backtest_overfitting(
    returns: np.ndarray,
    *,
    n_partitions: int = 8,
) -> BacktestOverfittingDiagnostic:
    """Estimate PBO with combinatorially symmetric cross-validation (CSCV).

    Assumptions: rows are ordered, synchronous net returns for all declared trials; the
    number of contiguous partitions is even; and all strategy selection occurred only on
    development data.  PBO is the fraction of in-sample winners whose out-of-sample rank
    is at or below the median.  It is an overfitting diagnostic, not a performance estimate.
    """

    values = _validate_returns(returns, minimum_columns=2)
    if n_partitions < 2 or n_partitions % 2:
        raise ValueError("n_partitions must be an even integer of at least 2")
    if values.shape[0] < n_partitions * 2:
        raise ValueError("At least two return observations are required per partition")

    partitions = [
        np.asarray(partition, dtype=int)
        for partition in np.array_split(np.arange(values.shape[0]), n_partitions)
    ]
    all_groups = set(range(n_partitions))
    rank_percentiles: list[float] = []
    logits: list[float] = []
    selected_indices: list[int] = []
    for train_groups in combinations(range(n_partitions), n_partitions // 2):
        test_groups = sorted(all_groups.difference(train_groups))
        train_indices = np.sort(np.concatenate([partitions[index] for index in train_groups]))
        test_indices = np.sort(np.concatenate([partitions[index] for index in test_groups]))
        in_sample_sharpes = _strategy_sharpes(values[train_indices])
        selected = int(np.argmax(in_sample_sharpes))
        out_of_sample_sharpes = _strategy_sharpes(values[test_indices])
        selected_score = out_of_sample_sharpes[selected]
        lower = int(np.count_nonzero(out_of_sample_sharpes < selected_score))
        equal = int(np.count_nonzero(out_of_sample_sharpes == selected_score))
        percentile = (lower + 0.5 * equal) / values.shape[1]
        percentile = float(np.clip(percentile, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
        rank_percentiles.append(percentile)
        logits.append(log(percentile / (1.0 - percentile)))
        selected_indices.append(selected)

    logit_array = np.asarray(logits, dtype=float)
    return BacktestOverfittingDiagnostic(
        pbo=float(np.mean(logit_array <= 0.0)),
        n_combinations=len(logits),
        median_logit=float(np.median(logit_array)),
        oos_rank_percentiles=tuple(rank_percentiles),
        selected_strategy_indices=tuple(selected_indices),
    )


def _sample_moments(values: np.ndarray) -> tuple[float, float, float]:
    centered = values - float(np.mean(values))
    second_moment = float(np.mean(centered**2))
    if second_moment <= 0.0:
        raise ValueError("Sharpe diagnostics require non-constant returns")
    skewness = float(np.mean(centered**3) / second_moment**1.5)
    kurtosis = float(np.mean(centered**4) / second_moment**2)
    sample_sharpe = float(np.mean(values) / np.std(values, ddof=1))
    return sample_sharpe, skewness, kurtosis


def _probabilistic_sharpe_probability(
    sample_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
) -> float:
    variance_term = 1.0 - skewness * sample_sharpe + ((kurtosis - 1.0) / 4.0) * sample_sharpe**2
    if variance_term <= 0.0:
        raise ValueError("Return moments imply a non-positive Sharpe variance approximation")
    statistic = (sample_sharpe - benchmark_sharpe) * sqrt(n_observations - 1) / sqrt(variance_term)
    return float(NormalDist().cdf(statistic))


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    *,
    benchmark_sharpe: float = 0.0,
    periods_per_year: float = 1.0,
) -> SharpeDiagnostic:
    """Probability that annualized Sharpe exceeds a fixed annualized benchmark.

    The Bailey-Lopez de Prado moment approximation is conditional on the frozen return
    series and does not account for strategy selection or serial correlation.  Callers must
    use non-overlapping or otherwise dependence-adjusted returns.
    """

    values = _validate_returns(returns)[:, 0]
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    sample_sharpe, skewness, kurtosis = _sample_moments(values)
    annualization = sqrt(periods_per_year)
    probability = _probabilistic_sharpe_probability(
        sample_sharpe,
        benchmark_sharpe / annualization,
        len(values),
        skewness,
        kurtosis,
    )
    return SharpeDiagnostic(
        probability=probability,
        observed_sharpe=sample_sharpe * annualization,
        benchmark_sharpe=float(benchmark_sharpe),
        n_observations=len(values),
        declared_trials=1,
        skewness=skewness,
        kurtosis=kurtosis,
        periods_per_year=float(periods_per_year),
    )


def _expected_maximum_sharpe(
    declared_trials: int,
    trial_sharpe_mean: float,
    trial_sharpe_std: float,
) -> float:
    if declared_trials == 1:
        return trial_sharpe_mean
    normal = NormalDist()
    euler_mascheroni = 0.5772156649015329
    first_quantile = normal.inv_cdf(1.0 - 1.0 / declared_trials)
    second_quantile = normal.inv_cdf(1.0 - 1.0 / (declared_trials * e))
    return trial_sharpe_mean + trial_sharpe_std * (
        (1.0 - euler_mascheroni) * first_quantile + euler_mascheroni * second_quantile
    )


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    declared_trials: int,
    periods_per_year: float = 1.0,
    trial_sharpe_mean: float = 0.0,
    trial_sharpe_std: float | None = None,
) -> SharpeDiagnostic:
    """Deflate Sharpe for the declared number of strategy trials.

    ``trial_sharpe_mean`` and ``trial_sharpe_std`` are annualized moments of the Sharpe
    distribution across *all* genuinely attempted development trials.  When the standard
    deviation is unavailable, the function uses ``sqrt(periods_per_year/(n-1))``, the IID
    Gaussian-null approximation.  This fallback and the independent-trials approximation
    must be disclosed; a complete append-only trial ledger is still required.
    """

    values = _validate_returns(returns)[:, 0]
    if declared_trials < 1:
        raise ValueError("declared_trials must be at least 1")
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    sample_sharpe, skewness, kurtosis = _sample_moments(values)
    annualization = sqrt(periods_per_year)
    if trial_sharpe_std is None:
        trial_sharpe_std = sqrt(periods_per_year / (len(values) - 1))
    if trial_sharpe_std < 0.0:
        raise ValueError("trial_sharpe_std cannot be negative")
    benchmark = _expected_maximum_sharpe(
        declared_trials,
        float(trial_sharpe_mean),
        float(trial_sharpe_std),
    )
    probability = _probabilistic_sharpe_probability(
        sample_sharpe,
        benchmark / annualization,
        len(values),
        skewness,
        kurtosis,
    )
    return SharpeDiagnostic(
        probability=probability,
        observed_sharpe=sample_sharpe * annualization,
        benchmark_sharpe=benchmark,
        n_observations=len(values),
        declared_trials=int(declared_trials),
        skewness=skewness,
        kurtosis=kurtosis,
        periods_per_year=float(periods_per_year),
    )

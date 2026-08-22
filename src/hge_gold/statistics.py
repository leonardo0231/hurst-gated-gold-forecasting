from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import e, log, sqrt
from statistics import NormalDist

import numpy as np


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

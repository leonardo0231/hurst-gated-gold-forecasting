from __future__ import annotations

from dataclasses import asdict
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import EvaluationConfig


class ClassificationMetrics(TypedDict):
    n_samples: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    precision_down: float
    precision_up: float
    recall_down: float
    recall_up: float
    mcc: float
    roc_auc: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    threshold: float
    confusion_matrix: list[list[int]]


def _expected_calibration_error(
    y_true: np.ndarray, probability_up: np.ndarray, n_bins: int = 10
) -> float:
    """Return equal-width expected calibration error for binary probabilities."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Include probability 1.0 in the final bin while keeping all other bins left-closed.
    bin_ids = np.minimum(np.digitize(probability_up, edges[1:-1], right=False), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        weight = float(mask.mean())
        observed = float(y_true[mask].mean())
        predicted = float(probability_up[mask].mean())
        error += weight * abs(observed - predicted)
    return float(error)


def classification_metrics(
    y_true: np.ndarray, probability_up: np.ndarray, threshold: float
) -> ClassificationMetrics:
    y_true = np.asarray(y_true, dtype=int)
    probability_up = np.asarray(probability_up, dtype=float)
    y_pred = (probability_up >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_down": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_up": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_down": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_up": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, probability_up))
        if np.unique(y_true).size == 2
        else float("nan"),
        "brier_score": float(brier_score_loss(y_true, probability_up)),
        "log_loss": float(log_loss(y_true, probability_up, labels=[0, 1])),
        "expected_calibration_error": _expected_calibration_error(y_true, probability_up),
        "threshold": float(threshold),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def tune_probability_threshold(
    y_true: np.ndarray,
    probability_up: np.ndarray,
    minimum: float,
    maximum: float,
    steps: int,
) -> tuple[float, ClassificationMetrics]:
    best_threshold = 0.5
    best_metrics = classification_metrics(y_true, probability_up, best_threshold)
    best_key = (
        float(best_metrics["balanced_accuracy"]),
        float(best_metrics["macro_f1"]),
        -abs(best_threshold - 0.5),
    )
    for threshold in np.linspace(minimum, maximum, steps):
        metrics = classification_metrics(y_true, probability_up, float(threshold))
        key = (
            float(metrics["balanced_accuracy"]),
            float(metrics["macro_f1"]),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def moving_block_bootstrap_ci(
    y_true: np.ndarray,
    probability_up: np.ndarray,
    threshold: float,
    iterations: int,
    block_length: int,
    seed: int,
) -> dict[str, float]:
    n = len(y_true)
    if n < max(20, block_length * 2):
        return {"balanced_accuracy_ci_low": float("nan"), "balanced_accuracy_ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    starts = np.arange(0, n - block_length + 1)
    for _ in range(iterations):
        selected: list[int] = []
        while len(selected) < n:
            start = int(rng.choice(starts))
            selected.extend(range(start, start + block_length))
        idx = np.asarray(selected[:n], dtype=int)
        if np.unique(y_true[idx]).size < 2:
            continue
        pred = (probability_up[idx] >= threshold).astype(int)
        values.append(float(balanced_accuracy_score(y_true[idx], pred)))
    if not values:
        return {"balanced_accuracy_ci_low": float("nan"), "balanced_accuracy_ci_high": float("nan")}
    low, high = np.quantile(values, [0.025, 0.975])
    return {"balanced_accuracy_ci_low": float(low), "balanced_accuracy_ci_high": float(high)}


def acceptance_status(
    metrics: ClassificationMetrics, config: EvaluationConfig
) -> dict[str, object]:
    checks = {
        "minimum_test_samples": int(metrics["n_samples"]) >= config.min_test_samples,
        "balanced_accuracy": float(metrics["balanced_accuracy"]) >= config.primary_threshold,
        "macro_f1": float(metrics["macro_f1"]) >= config.macro_f1_threshold,
        "recall_down": float(metrics["recall_down"]) >= config.minimum_class_recall,
        "recall_up": float(metrics["recall_up"]) >= config.minimum_class_recall,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "registered_thresholds": asdict(config),
    }


def _select_non_overlapping_predictions(
    predictions: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    ordered = predictions.sort_values("row_id").reset_index(drop=True)

    if ordered.empty:
        return ordered

    minimum_gap = max(1, int(horizon))
    selected_positions: list[int] = []
    last_row_id: int | None = None

    for position, row_id in enumerate(ordered["row_id"].to_numpy(dtype=int)):
        current_row_id = int(row_id)

        if last_row_id is None or current_row_id - last_row_id >= minimum_gap:
            selected_positions.append(position)
            last_row_id = current_row_id

    return ordered.iloc[selected_positions].reset_index(drop=True)


def backtest_summary(
    predictions: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    slippage_bps: float,
    probability_no_trade_band: float = 0.0,
) -> dict[str, float | int]:
    if not 0.0 <= probability_no_trade_band < 0.5:
        raise ValueError("probability_no_trade_band must be in [0, 0.5)")
    selected, signal = _select_active_non_overlapping_predictions(
        predictions, horizon, probability_no_trade_band
    )
    summary = _strategy_summary(
        selected["forward_log_return"].to_numpy(dtype=float),
        signal,
        _observed_trades_per_year(predictions, len(selected)),
        transaction_cost_bps,
        slippage_bps,
    )
    summary["n_decision_opportunities"] = int(len(predictions))
    summary["trade_coverage"] = float(len(selected) / len(predictions)) if len(predictions) else 0.0
    return summary


def _select_active_non_overlapping_predictions(
    predictions: pd.DataFrame,
    horizon: int,
    probability_no_trade_band: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply abstention first, then take the next signal whenever the strategy is flat."""
    ordered = predictions.sort_values("row_id").reset_index(drop=True)
    if probability_no_trade_band > 0.0 and "probability_up" not in ordered:
        raise ValueError("probability_up is required when a no-trade band is enabled")
    selected_positions: list[int] = []
    selected_signals: list[float] = []
    last_row_id: int | None = None
    for position, row in enumerate(ordered.itertuples(index=False)):
        if probability_no_trade_band > 0.0 and (
            abs(float(row.probability_up) - 0.5) < probability_no_trade_band
        ):
            continue
        row_id = int(row.row_id)
        if last_row_id is not None and row_id - last_row_id < max(1, int(horizon)):
            continue
        selected_positions.append(position)
        selected_signals.append(1.0 if int(row.y_pred) == 1 else -1.0)
        last_row_id = row_id
    return (
        ordered.iloc[selected_positions].reset_index(drop=True),
        np.asarray(selected_signals, dtype=float),
    )


def _strategy_summary(
    forward_log_return: np.ndarray,
    signal: np.ndarray,
    periods_per_year: float,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> dict[str, float | int]:
    forward_log_return = np.asarray(forward_log_return, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if forward_log_return.shape != signal.shape:
        raise ValueError("forward returns and signals must have identical shapes")
    active = signal != 0.0
    gross = signal * forward_log_return
    cost = np.log1p((transaction_cost_bps + slippage_bps) / 10_000.0)
    net = gross - cost * active.astype(float)
    equity = np.concatenate(([1.0], np.exp(np.cumsum(net))))
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    sharpe = (
        np.sqrt(periods_per_year) * float(np.mean(net)) / float(np.std(net, ddof=1))
        if len(net) > 1 and periods_per_year > 0.0 and np.std(net, ddof=1) > 0
        else float("nan")
    )
    active_net = net[active]
    return {
        "n_non_overlapping_observations": int(len(net)),
        "n_non_overlapping_trades": int(active.sum()),
        "trade_coverage": float(active.mean()) if len(active) else 0.0,
        "hit_rate": float((active_net > 0).mean()) if len(active_net) else float("nan"),
        "mean_net_log_return": float(np.mean(active_net)) if len(active_net) else float("nan"),
        "cumulative_net_return": float(equity[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "annualized_sharpe": float(sharpe),
        "annualization_periods_per_year": float(periods_per_year),
    }


def _observed_trades_per_year(predictions: pd.DataFrame, selected_count: int) -> float:
    """Annualize selected-trade returns using their actual decision-span frequency."""
    if selected_count == 0 or predictions.empty:
        return 0.0
    row_ids = predictions["row_id"].to_numpy(dtype=int)
    observed_span = max(1, int(row_ids.max() - row_ids.min() + 1))
    return float(252.0 * selected_count / observed_span)


def economic_benchmark_summaries(
    predictions: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    slippage_bps: float,
    probability_no_trade_band: float = 0.0,
) -> dict[str, dict[str, float | int]]:
    """Evaluate model and simple benchmarks on the exact same non-overlapping rows."""
    selected, model_signal = _select_active_non_overlapping_predictions(
        predictions, horizon, probability_no_trade_band
    )
    returns = selected["forward_log_return"].to_numpy(dtype=float)
    if not 0.0 <= probability_no_trade_band < 0.5:
        raise ValueError("probability_no_trade_band must be in [0, 0.5)")

    required_benchmark_features = {"return_lag_1", "momentum_20", "volatility_zscore"}
    missing = required_benchmark_features - set(selected.columns)
    if missing:
        raise ValueError(f"Missing causal benchmark features: {sorted(missing)}")
    momentum_1 = np.where(selected["return_lag_1"].to_numpy(dtype=float) >= 0.0, 1.0, -1.0)
    momentum_20 = np.where(selected["momentum_20"].to_numpy(dtype=float) >= 0.0, 1.0, -1.0)
    volatility_filtered = np.where(
        selected["volatility_zscore"].to_numpy(dtype=float) <= 0.0,
        momentum_20,
        0.0,
    )
    signals = {
        "model": model_signal,
        "always_long": np.ones(len(selected), dtype=float),
        "always_short": -np.ones(len(selected), dtype=float),
        "cash": np.zeros(len(selected), dtype=float),
        "last_return_momentum": momentum_1,
        "moving_average_20": momentum_20,
        "volatility_filtered_momentum": volatility_filtered,
    }
    summaries = {
        name: _strategy_summary(
            returns,
            signal,
            _observed_trades_per_year(predictions, len(selected)),
            transaction_cost_bps,
            slippage_bps,
        )
        for name, signal in signals.items()
    }
    for summary in summaries.values():
        summary["n_decision_opportunities"] = int(len(predictions))
        summary["trade_coverage"] = (
            float(int(summary["n_non_overlapping_trades"]) / len(predictions))
            if len(predictions)
            else 0.0
        )
    full_period = _select_non_overlapping_predictions(predictions, horizon)
    summaries["full_period_buy_and_hold"] = _strategy_summary(
        full_period["forward_log_return"].to_numpy(dtype=float),
        np.ones(len(full_period), dtype=float),
        _observed_trades_per_year(predictions, len(full_period)),
        transaction_cost_bps,
        slippage_bps,
    )
    summaries["full_period_buy_and_hold"]["n_decision_opportunities"] = int(len(predictions))
    summaries["full_period_buy_and_hold"]["trade_coverage"] = (
        float(len(full_period) / len(predictions)) if len(predictions) else 0.0
    )
    return summaries

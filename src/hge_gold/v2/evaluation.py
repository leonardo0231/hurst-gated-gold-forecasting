from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import EvaluationConfig


def classification_metrics(y_true: np.ndarray, probability_up: np.ndarray, threshold: float) -> dict[str, float | int | list[list[int]]]:
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
        "roc_auc": float(roc_auc_score(y_true, probability_up)) if np.unique(y_true).size == 2 else float("nan"),
        "brier_score": float(brier_score_loss(y_true, probability_up)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def tune_probability_threshold(
    y_true: np.ndarray,
    probability_up: np.ndarray,
    minimum: float,
    maximum: float,
    steps: int,
) -> tuple[float, dict[str, float | int | list[list[int]]]]:
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


def acceptance_status(metrics: dict[str, float | int | list[list[int]]], config: EvaluationConfig) -> dict[str, object]:
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


def backtest_summary(
    predictions: pd.DataFrame,
    horizon: int,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> dict[str, float | int]:
    ordered = predictions.sort_values("row_id").reset_index(drop=True)
    non_overlapping = ordered.iloc[::max(1, horizon)].copy()
    signal = np.where(non_overlapping["y_pred"].to_numpy() == 1, 1.0, -1.0)
    gross = signal * non_overlapping["forward_log_return"].to_numpy(dtype=float)
    cost = np.log1p((transaction_cost_bps + slippage_bps) / 10_000.0)
    net = gross - cost
    equity = np.exp(np.cumsum(net))
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    periods_per_year = 252.0 / max(1, horizon)
    sharpe = (
        np.sqrt(periods_per_year) * float(np.mean(net)) / float(np.std(net, ddof=1))
        if len(net) > 1 and np.std(net, ddof=1) > 0
        else float("nan")
    )
    return {
        "n_non_overlapping_trades": int(len(net)),
        "hit_rate": float((net > 0).mean()) if len(net) else float("nan"),
        "mean_net_log_return": float(np.mean(net)) if len(net) else float("nan"),
        "cumulative_net_return": float(equity[-1] - 1.0) if len(equity) else float("nan"),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else float("nan"),
        "annualized_sharpe": float(sharpe),
    }
